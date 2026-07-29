"""Canonical FastAPI service facade for unified MCP runtime.

This module provides a stable HTTP entrypoint in ``ipfs_accelerate_py.mcp_server``
without routing back through the legacy ``ipfs_accelerate_py.mcp`` facade.
"""

from __future__ import annotations

import inspect
import json
import logging
import re
from collections.abc import Mapping
from typing import Any, Callable, Dict, List, Optional

from .fastapi_config import UnifiedFastAPIConfig
from .server import create_server

try:
    # FastAPI resolves postponed endpoint annotations against module globals.
    # Keep this optional so importing the facade still works without FastAPI.
    from starlette.requests import Request
except ImportError:  # pragma: no cover - exercised only without FastAPI
    Request = Any  # type: ignore[misc,assignment]


logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
_PROFILE_G_REST_ROUTES = (
    ("/mcp/risk/profile", ("GET",)),
    ("/mcp/goals", ("GET", "POST")),
    ("/mcp/goals/{goal_cid}", ("GET",)),
    ("/mcp/goals/{goal_cid}/decompose", ("POST",)),
    ("/mcp/goals/{goal_cid}/select", ("POST",)),
    ("/mcp/tasks", ("GET", "POST")),
    ("/mcp/tasks/ready", ("GET",)),
    ("/mcp/tasks/{task_cid}", ("GET",)),
    ("/mcp/risk/assess", ("POST",)),
    ("/mcp/risk/evidence", ("GET",)),
    ("/mcp/risk/history", ("GET",)),
    ("/mcp/neighborhood/query", ("POST",)),
    ("/mcp/neighborhood/attest", ("POST",)),
    ("/mcp/schedule/frontier", ("GET",)),
    ("/mcp/schedule/proposals", ("POST",)),
    ("/mcp/schedule/claims", ("POST",)),
    ("/mcp/schedule/claims/{claim_cid}/renew", ("POST",)),
    ("/mcp/schedule/claims/{claim_cid}/release", ("POST",)),
    ("/mcp/schedule/resolutions", ("POST",)),
    ("/mcp/schedule/reconcile", ("POST",)),
    ("/mcp/schedule/status/{task_cid}", ("GET",)),
)


class _MCPDispatchError(ValueError):
    """A request could not be dispatched through the local MCP registry."""

    def __init__(self, code: int, message: str, *, data: Any = None) -> None:
        super().__init__(message)
        self.code = int(code)
        self.message = str(message)
        self.data = data


def _jsonrpc_result(request_id: Any, result: Any) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "result": result}


def _jsonrpc_error(
    request_id: Any,
    code: int,
    message: str,
    *,
    data: Any = None,
) -> dict[str, Any]:
    error: dict[str, Any] = {"code": int(code), "message": str(message)}
    if data is not None:
        error["data"] = data
    return {"jsonrpc": "2.0", "id": request_id, "error": error}


def _mcp_tool_registry(server: Any) -> Mapping[str, Any]:
    registry = getattr(server, "tools", None)
    if isinstance(registry, Mapping):
        return registry
    registry = getattr(getattr(server, "mcp", None), "tools", None)
    return registry if isinstance(registry, Mapping) else {}


def _mcp_tool_descriptors(server: Any) -> list[dict[str, Any]]:
    descriptors: list[dict[str, Any]] = []
    for raw_name, raw_spec in sorted(
        _mcp_tool_registry(server).items(),
        key=lambda item: str(item[0]),
    ):
        name = str(raw_name)
        spec = raw_spec if isinstance(raw_spec, Mapping) else {}
        input_schema = spec.get("input_schema", spec.get("inputSchema", {}))
        if not isinstance(input_schema, Mapping):
            input_schema = {}
        descriptors.append(
            {
                "name": name,
                "description": str(spec.get("description") or ""),
                "inputSchema": dict(input_schema),
            }
        )
    return descriptors


def _normalize_tool_arguments(
    tool_name: str,
    arguments: Mapping[str, Any],
) -> dict[str, Any]:
    normalized = dict(arguments)
    if tool_name == "tools_list_categories":
        # The canonical registry currently returns names only. The client may
        # request counts, but that optional display hint must not break calls.
        normalized.pop("include_count", None)
    elif tool_name == "tools_get_schema":
        if "tool_name" not in normalized and "tool" in normalized:
            normalized["tool_name"] = normalized.pop("tool")
    elif tool_name == "tools_dispatch":
        if "tool_name" not in normalized and "tool" in normalized:
            normalized["tool_name"] = normalized.pop("tool")
        if "parameters" not in normalized and "params" in normalized:
            normalized["parameters"] = normalized.pop("params")
    return normalized


def _mcp_tool_result(value: Any) -> dict[str, Any]:
    if (
        isinstance(value, Mapping)
        and isinstance(value.get("content"), list)
    ):
        return dict(value)
    return {
        "content": [
            {
                "type": "text",
                "text": json.dumps(
                    value,
                    sort_keys=True,
                    separators=(",", ":"),
                    ensure_ascii=False,
                    default=str,
                ),
            }
        ],
        "isError": False,
    }


async def _call_mcp_tool(
    server: Any,
    tool_name: Any,
    arguments: Any,
) -> dict[str, Any]:
    if not isinstance(tool_name, str) or not tool_name.strip():
        raise _MCPDispatchError(-32602, "tool name must be a non-empty string")
    name = tool_name.strip()
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, Mapping):
        raise _MCPDispatchError(-32602, "tool arguments must be an object")

    registry = _mcp_tool_registry(server)
    spec = registry.get(name)
    if spec is not None:
        function = (
            spec.get("function")
            if isinstance(spec, Mapping)
            else spec
        )
        if not callable(function):
            raise _MCPDispatchError(
                -32603,
                "registered tool is not callable",
                data={"tool": name},
            )
        normalized = _normalize_tool_arguments(name, arguments)
        try:
            result = function(**normalized)
            if inspect.isawaitable(result):
                result = await result
        except TypeError as error:
            raise _MCPDispatchError(
                -32602,
                "tool arguments do not match the registered schema",
                data={"tool": name, "error_type": type(error).__name__},
            ) from error
        except Exception as error:
            raise _MCPDispatchError(
                -32603,
                "tool execution failed",
                data={"tool": name, "error_type": type(error).__name__},
            ) from error
        return _mcp_tool_result(result)

    manager = getattr(server, "_unified_tool_manager", None)
    separator = "." if "." in name else "/" if "/" in name else ""
    if separator and callable(getattr(manager, "dispatch", None)):
        category, leaf = name.split(separator, 1)
        if category and leaf:
            try:
                result = manager.dispatch(category, leaf, dict(arguments))
                if inspect.isawaitable(result):
                    result = await result
            except Exception as error:
                raise _MCPDispatchError(
                    -32603,
                    "tool execution failed",
                    data={"tool": name, "error_type": type(error).__name__},
                ) from error
            return _mcp_tool_result(result)

    raise _MCPDispatchError(-32601, "tool not found", data={"tool": name})


def _profile_g_rest_binding(http_method: str, path: str) -> tuple[str, dict[str, Any]] | None:
    """Resolve a normative Profile G REST path to its JSON-RPC method."""
    static = {
        ("GET", "/mcp/risk/profile"): "mcp++/risk/profile",
        ("POST", "/mcp/goals"): "mcp++/goals/create",
        ("GET", "/mcp/goals"): "mcp++/goals/list",
        ("POST", "/mcp/tasks"): "mcp++/tasks/create",
        ("GET", "/mcp/tasks"): "mcp++/tasks/list",
        ("GET", "/mcp/tasks/ready"): "mcp++/tasks/ready",
        ("POST", "/mcp/risk/assess"): "mcp++/risk/assess",
        ("GET", "/mcp/risk/evidence"): "mcp++/risk/evidence",
        ("GET", "/mcp/risk/history"): "mcp++/risk/history",
        ("POST", "/mcp/neighborhood/query"): "mcp++/neighborhood/query",
        ("POST", "/mcp/neighborhood/attest"): "mcp++/neighborhood/attest",
        ("GET", "/mcp/schedule/frontier"): "mcp++/schedule/frontier",
        ("POST", "/mcp/schedule/proposals"): "mcp++/schedule/propose",
        ("POST", "/mcp/schedule/claims"): "mcp++/schedule/claim",
        ("POST", "/mcp/schedule/resolutions"): "mcp++/schedule/resolve",
        ("POST", "/mcp/schedule/reconcile"): "mcp++/schedule/reconcile",
    }
    method = static.get((http_method, path))
    if method:
        return method, {}
    patterns = (
        ("GET", r"/mcp/goals/([^/]+)$", "mcp++/goals/get", "goal_cid"),
        ("POST", r"/mcp/goals/([^/]+)/(decompose|select)$", None, "goal_cid"),
        ("GET", r"/mcp/tasks/([^/]+)$", "mcp++/tasks/get", "task_cid"),
        ("GET", r"/mcp/schedule/status/([^/]+)$", "mcp++/schedule/status", "task_cid"),
        ("POST", r"/mcp/schedule/claims/([^/]+)/(renew|release)$", None, "claim_cid"),
    )
    for verb, pattern, rpc_method, cid_key in patterns:
        if verb != http_method:
            continue
        match = re.fullmatch(pattern, path)
        if match:
            if rpc_method is None:
                rpc_method = f"mcp++/{'goals' if cid_key == 'goal_cid' else 'schedule'}/{match.group(2)}"
            return rpc_method, {cid_key: match.group(1)}
    return None


class _FallbackStandaloneApp:
    """Minimal fallback app used when FastAPI is unavailable."""

    def __init__(self, title: str, description: str):
        self.title = title
        self.description = description
        self.mounts: list[dict[str, Any]] = []
        self.routes: list[dict[str, Any]] = []

    def mount(self, path: str, app: Any, name: Optional[str] = None) -> None:
        self.mounts.append({"path": path, "app": app, "name": name})

    def add_route(
        self,
        path: str,
        endpoint: Callable[..., Any],
        methods: Optional[List[str]] = None,
    ) -> None:
        self.routes.append({"path": path, "endpoint": endpoint, "methods": methods or ["GET"]})


_DEFAULT_CONFIG: UnifiedFastAPIConfig | None = None
_DEFAULT_APP: Any | None = None


def get_fastapi_config() -> UnifiedFastAPIConfig:
    """Return cached canonical FastAPI settings for import-compatible callers."""
    global _DEFAULT_CONFIG
    if _DEFAULT_CONFIG is None:
        _DEFAULT_CONFIG = UnifiedFastAPIConfig.from_env()
    return _DEFAULT_CONFIG


def get_fastapi_app() -> Any:
    """Return cached canonical FastAPI app for import-compatible callers."""
    global _DEFAULT_APP
    if _DEFAULT_APP is None:
        _DEFAULT_APP = create_fastapi_app(get_fastapi_config())
    return _DEFAULT_APP


def create_fastapi_app(config: UnifiedFastAPIConfig | None = None) -> Any:
    """Create a standalone FastAPI-compatible app for MCP endpoints."""
    resolved = config or UnifiedFastAPIConfig.from_env()
    title = "IPFS Accelerate MCP API"
    mcp_server = create_server(
        name=resolved.name,
        description=resolved.description,
        mount_path="",
    )

    try:
        from fastapi import FastAPI, HTTPException

        app: Any = FastAPI(
            title=title,
            description=resolved.description,
            version="0.1.0",
            docs_url="/docs",
            redoc_url="/redoc",
        )

        @app.get("/healthz")
        async def _healthz() -> Dict[str, Any]:
            return {"status": "ok", "service": resolved.name}

        @app.get("/mcp/health")
        async def _mcp_health() -> Dict[str, Any]:
            return {
                "status": "ok",
                "server": resolved.name,
                "protocol_version": MCP_PROTOCOL_VERSION,
                "tool_count": len(_mcp_tool_registry(mcp_server)),
            }

        async def _request_payload(request: Request) -> dict[str, Any]:
            try:
                payload = await request.json()
            except Exception as error:
                raise HTTPException(
                    status_code=400,
                    detail="request body must be JSON",
                ) from error
            if not isinstance(payload, dict):
                raise HTTPException(
                    status_code=400,
                    detail="request body must be an object",
                )
            return payload

        async def _optional_request_payload(
            request: Request,
        ) -> dict[str, Any]:
            body = await request.body()
            if not body:
                return {}
            return await _request_payload(request)

        def _tool_call_fields(
            payload: Mapping[str, Any],
        ) -> tuple[Any, Any, Any]:
            params = payload.get("params", payload)
            if not isinstance(params, Mapping):
                raise _MCPDispatchError(-32602, "tool call params must be an object")
            return (
                payload.get("id"),
                params.get("name"),
                params.get("arguments", params.get("args", {})),
            )

        async def _dispatch_jsonrpc(
            payload: Mapping[str, Any],
        ) -> dict[str, Any]:
            request_id = payload.get("id")
            method = payload.get("method")
            params = payload.get("params", {})
            if not isinstance(method, str) or not method:
                return _jsonrpc_error(request_id, -32600, "invalid JSON-RPC request")
            if params is None:
                params = {}
            if not isinstance(params, Mapping):
                return _jsonrpc_error(request_id, -32602, "params must be an object")

            if method == "initialize":
                requested_version = params.get("protocolVersion")
                protocol_version = (
                    requested_version
                    if isinstance(requested_version, str) and requested_version
                    else MCP_PROTOCOL_VERSION
                )
                supported = set(
                    getattr(mcp_server, "_unified_supported_profiles", ()) or ()
                )
                # Only advertise the profile implemented on this HTTP boundary.
                # Other optional profiles remain available through their own
                # transports and must not trigger unsupported client behavior.
                experimental = {
                    "mcp++/risk-scheduling": True
                } if "mcp++/risk-scheduling" in supported else {}
                return _jsonrpc_result(
                    request_id,
                    {
                        "protocolVersion": protocol_version,
                        "capabilities": {
                            "tools": {"listChanged": False},
                            "experimental": experimental,
                        },
                        "serverInfo": {
                            "name": resolved.name,
                            "version": "0.1.0",
                        },
                    },
                )
            if method == "tools/list":
                return _jsonrpc_result(
                    request_id,
                    {"tools": _mcp_tool_descriptors(mcp_server)},
                )
            if method == "tools/call":
                try:
                    result = await _call_mcp_tool(
                        mcp_server,
                        params.get("name"),
                        params.get("arguments", {}),
                    )
                except _MCPDispatchError as error:
                    return _jsonrpc_error(
                        request_id,
                        error.code,
                        error.message,
                        data=error.data,
                    )
                return _jsonrpc_result(request_id, result)
            if method == "shutdown":
                return _jsonrpc_result(request_id, None)

            from .mcplusplus.profile_g_transport import (
                ProfileGTransportError,
                get_profile_g_dispatcher,
                is_profile_g_method,
                jsonrpc_error,
            )

            if is_profile_g_method(method):
                try:
                    return _jsonrpc_result(
                        request_id,
                        get_profile_g_dispatcher().dispatch(method, dict(params)),
                    )
                except ProfileGTransportError as error:
                    return jsonrpc_error(request_id, error)
            return _jsonrpc_error(request_id, -32601, "method not found")

        @app.get("/mcp/tools/list")
        async def _mcp_tools_list_get() -> Dict[str, Any]:
            return _jsonrpc_result(
                None,
                {"tools": _mcp_tool_descriptors(mcp_server)},
            )

        @app.post("/mcp/tools/list")
        async def _mcp_tools_list_post(request: Request) -> Dict[str, Any]:
            payload = await _optional_request_payload(request)
            return _jsonrpc_result(
                payload.get("id"),
                {"tools": _mcp_tool_descriptors(mcp_server)},
            )

        @app.post("/mcp/tools/call")
        async def _mcp_tools_call(request: Request) -> Dict[str, Any]:
            payload = await _request_payload(request)
            try:
                request_id, tool_name, arguments = _tool_call_fields(payload)
                result = await _call_mcp_tool(
                    mcp_server,
                    tool_name,
                    arguments,
                )
            except _MCPDispatchError as error:
                return _jsonrpc_error(
                    payload.get("id"),
                    error.code,
                    error.message,
                    data=error.data,
                )
            return _jsonrpc_result(request_id, result)

        @app.post("/mcp")
        @app.post("/mcp/")
        async def _mcp_jsonrpc(request: Request) -> Dict[str, Any]:
            return await _dispatch_jsonrpc(await _request_payload(request))

        @app.post("/mcp/policy/evaluate")
        async def _evaluate_profile_d_policy(request: Request) -> Dict[str, Any]:
            """Canonical Profile D REST evaluation endpoint.

            The mounted MCP application retains normal tool traffic; this
            explicit route provides the Profile D REST surface specified by
            MCP++ and shares its evaluator with the libp2p dispatcher.
            """
            from .mcplusplus.policy_engine import evaluate_profile_d_execution_policy

            try:
                payload = await request.json()
                if not isinstance(payload, dict):
                    raise ValueError("request body must be an object")
                return evaluate_profile_d_execution_policy(
                    actor=payload.get("actor", ""),
                    action=payload.get("action", ""),
                    resource=payload.get("resource"),
                    policy=payload.get("policy") if isinstance(payload.get("policy"), dict) else None,
                    policy_text=payload.get("policy_text"),
                    evaluated_at=payload.get("evaluated_at"),
                    intent_cid=payload.get("intent_cid"),
                    request_zkp_certificate=bool(payload.get("request_zkp_certificate", False)),
                )
            except ValueError as error:
                raise HTTPException(status_code=400, detail=str(error)) from error

        async def _profile_g_rest(request: Request) -> Any:
            from .mcplusplus.profile_g_transport import (
                ERROR_NUMBERS, ProfileGTransportError, get_profile_g_dispatcher,
            )
            binding = _profile_g_rest_binding(request.method, request.url.path)
            if binding is None:
                raise HTTPException(status_code=404, detail="unknown MCP++ REST operation")
            method, path_params = binding
            params = dict(request.query_params)
            for integer_name in ("limit", "at_ms"):
                if integer_name in params:
                    try:
                        params[integer_name] = int(params[integer_name])
                    except ValueError as error:
                        raise HTTPException(status_code=400, detail=f"{integer_name} must be an integer") from error
            if request.method == "POST":
                try:
                    body = await request.json()
                except Exception as error:
                    raise HTTPException(status_code=400, detail="request body must be JSON") from error
                if not isinstance(body, dict):
                    raise HTTPException(status_code=400, detail="request body must be an object")
                params.update(body)
            params.update(path_params)  # the path is authoritative
            try:
                return get_profile_g_dispatcher().dispatch(method, params)
            except ProfileGTransportError as error:
                status = 400 if ERROR_NUMBERS.get(error.code) == -32602 else (
                    403 if error.code in {"G_AUTHORITY_DENIED", "G_POLICY_DENIED", "G_REDACTED"}
                    else 409 if error.code in {"G_NOT_READY", "G_IDEMPOTENCY_CONFLICT", "G_CLAIM_CONFLICT", "G_LEASE_EXPIRED"}
                    else 422 if error.code in {"G_CID_MISMATCH", "G_EVIDENCE_INVALID"} else 503
                )
                from fastapi.responses import JSONResponse
                return JSONResponse(status_code=status, content={
                    "code": ERROR_NUMBERS.get(error.code, -32603),
                    "message": error.message, "data": error.data(),
                })

        for profile_path, methods in _PROFILE_G_REST_ROUTES:
            app.add_api_route(
                profile_path,
                _profile_g_rest,
                methods=list(methods),
            )

    except ImportError:
        logger.warning("FastAPI is not installed; using fallback standalone app")
        app = _FallbackStandaloneApp(title=title, description=resolved.description)

        async def _healthz() -> Dict[str, Any]:
            return {"status": "ok", "service": resolved.name}

        app.add_route("/healthz", _healthz, methods=["GET"])

    mountable = getattr(mcp_server, "app", None)
    app.mount(resolved.mount_path, mountable if mountable is not None else mcp_server, name="mcp_server")
    setattr(app, "_mcp_server", mcp_server)
    return app


def run_standalone_app(app: Any, host: str = "localhost", port: int = 8000, verbose: bool = False) -> None:
    """Run a standalone FastAPI app using uvicorn."""
    try:
        import uvicorn
    except ImportError as exc:
        raise RuntimeError("uvicorn is required to run the standalone MCP app") from exc

    uvicorn.run(app, host=host, port=int(port), log_level="debug" if verbose else "info")


def run_fastapi_server(config: UnifiedFastAPIConfig | None = None) -> None:
    """Run canonical MCP FastAPI service using integration runner."""
    resolved = config or UnifiedFastAPIConfig.from_env()
    app = create_fastapi_app(resolved)
    run_standalone_app(
        app,
        host=resolved.host,
        port=resolved.port,
        verbose=resolved.verbose,
    )


def main() -> None:
    """Entrypoint for `python -m ipfs_accelerate_py.mcp_server.fastapi_service`."""
    run_fastapi_server()


def __getattr__(name: str) -> Any:
    """Provide lazy import-compatible `settings` and `app` module attributes."""
    if name == "settings":
        return get_fastapi_config()
    if name == "app":
        return get_fastapi_app()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


if __name__ == "__main__":
    main()
