"""Canonical FastAPI service facade for unified MCP runtime.

This module provides a stable HTTP entrypoint in ``ipfs_accelerate_py.mcp_server``
without routing back through the legacy ``ipfs_accelerate_py.mcp`` facade.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Callable, Dict, List, Optional

from .fastapi_config import UnifiedFastAPIConfig
from .server import create_server


logger = logging.getLogger(__name__)


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

    try:
        from fastapi import FastAPI, HTTPException, Request

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

        @app.api_route("/mcp/{profile_g_path:path}", methods=["GET", "POST"])
        async def _profile_g_rest(profile_g_path: str, request: Request) -> Any:
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

    except ImportError:
        logger.warning("FastAPI is not installed; using fallback standalone app")
        app = _FallbackStandaloneApp(title=title, description=resolved.description)

        async def _healthz() -> Dict[str, Any]:
            return {"status": "ok", "service": resolved.name}

        app.add_route("/healthz", _healthz, methods=["GET"])

    mcp_server = create_server(
        name=resolved.name,
        description=resolved.description,
        mount_path="",
    )

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
