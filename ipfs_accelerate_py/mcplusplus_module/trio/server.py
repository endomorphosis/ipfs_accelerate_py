"""
Trio-native MCP Server for MCP++ (Model Context Protocol Plus Plus).

This module provides a Trio-native implementation of the MCP server that runs
P2P operations without asyncio-to-Trio bridging overhead.

Module: ipfs_accelerate_py.mcplusplus_module.trio.server

Key features:
- Native Trio event loop (no asyncio bridges)
- Structured concurrency with nurseries
- Graceful shutdown with cancel scopes
- Hypercorn-compatible ASGI application
- Full P2P tool integration (20 tools)

Usage:
    import trio
    from ipfs_accelerate_py.mcplusplus_module.trio import TrioMCPServer

    async def main():
        server = TrioMCPServer(name="ipfs-accelerate-p2p")
        await server.run()

    if __name__ == "__main__":
        trio.run(main)

For Hypercorn deployment:
    hypercorn --worker-class trio ipfs_accelerate_py.mcplusplus_module.trio.server:create_app
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections import defaultdict
from typing import Any, Optional
from dataclasses import dataclass, field

import trio

from ..cid_ucan import compute_cid

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.trio.server")


# ---------------------------------------------------------------------------
# Rate Limiter (Token Bucket)
# ---------------------------------------------------------------------------

class TokenBucketRateLimiter:
    """Per-IP token bucket rate limiter for HTTP endpoints.

    Configurable via environment variables:
        MCPPP_RATE_LIMIT_RPS: Requests per second per IP (default: 50, max: 10000)
        MCPPP_RATE_LIMIT_BURST: Max burst size (default: 100, max: 100000)
    """

    def __init__(self):
        self.rps = self._safe_int("MCPPP_RATE_LIMIT_RPS", 50, 1, 10000)
        self.burst = self._safe_int("MCPPP_RATE_LIMIT_BURST", 100, 1, 100000)
        self._buckets: dict = defaultdict(lambda: {"tokens": self.burst, "last": time.time()})

    @staticmethod
    def _safe_int(key: str, default: int, min_val: int, max_val: int) -> int:
        """Parse env var as int with bounds validation."""
        raw = os.environ.get(key, "")
        if not raw:
            return default
        try:
            val = int(raw)
            if val < min_val or val > max_val:
                logger.warning("%s=%d out of range [%d, %d], using %d", key, val, min_val, max_val, default)
                return default
            return val
        except (ValueError, OverflowError):
            logger.warning("%s=%r not a valid integer, using %d", key, raw, default)
            return default

    def allow(self, client_ip: str) -> bool:
        """Check if a request from client_ip is allowed."""
        now = time.time()
        bucket = self._buckets[client_ip]
        elapsed = now - bucket["last"]
        bucket["last"] = now
        # Refill tokens
        bucket["tokens"] = min(self.burst, bucket["tokens"] + elapsed * self.rps)
        # Periodic auto-cleanup to prevent memory leak from many unique IPs
        if len(self._buckets) > 10000:
            self.cleanup()
        if bucket["tokens"] >= 1.0:
            bucket["tokens"] -= 1.0
            return True
        return False

    def cleanup(self) -> None:
        """Remove stale entries (older than 60s) to prevent memory leak."""
        now = time.time()
        stale = [ip for ip, b in self._buckets.items() if now - b["last"] > 60]
        for ip in stale:
            del self._buckets[ip]


@dataclass
class ServerConfig:
    """Configuration for TrioMCPServer.

    Attributes:
        name: Server name (default: "ipfs-accelerate-mcp-trio")
        host: Host to bind to (default: "0.0.0.0")
        port: Port to bind to (default: 8000)
        mount_path: API mount path (default: "/mcp")
        debug: Enable debug logging (default: False)
        enable_p2p_tools: Enable all P2P tools (default: True)
        enable_workflow_tools: Enable workflow scheduler tools (default: True)
        enable_taskqueue_tools: Enable taskqueue tools (default: True)
    """
    name: str = "ipfs-accelerate-mcp-trio"
    host: str = "0.0.0.0"
    port: int = 8000
    mount_path: str = "/mcp"
    debug: bool = False
    enable_p2p_tools: bool = True
    enable_workflow_tools: bool = True
    enable_taskqueue_tools: bool = True

    @classmethod
    def from_env(cls) -> ServerConfig:
        """Create configuration from environment variables.

        Environment variables:
            MCP_SERVER_NAME: Server name
            MCP_HOST: Host to bind to
            MCP_PORT: Port to bind to (default: 8000)
            MCP_MOUNT_PATH: API mount path
            MCP_DEBUG: Enable debug logging (1/true/yes)
            MCP_DISABLE_P2P: Disable P2P tools (1/true/yes)
            MCPPP_RATE_LIMIT_RPS: Rate limit requests/sec per IP (default: 50)
            MCPPP_RATE_LIMIT_BURST: Rate limit burst size (default: 100)
            MCPPP_EPOCH_SIZE: ZK compaction epoch size (default: 1000)
            MCPPP_HOT_TIER_MAX: Hot tier max events (default: 2000)
            MCPPP_STORAGE_DIR: Storage directory for cold tier + revocations
            MCPPP_EXEC_TIMEOUT_S: Tool execution timeout seconds (default: 30)
            MCPPP_ALLOW_UNSIGNED_DELEGATIONS: Allow unsigned UCAN (0/1, default: 0)
            MCPPP_REQUIRE_SERVICE_SIGNATURES: Require signed service announces (0/1)
            MCP_CORS_ORIGINS: Comma-separated allowed CORS origins
            MCP_LOG_FORMAT: Log format - "text" (default) or "json"

        Returns:
            ServerConfig instance with values from environment
        """
        def _safe_int(env_key: str, default: int) -> int:
            val = os.getenv(env_key, "")
            try:
                return int(val) if val else default
            except ValueError:
                logger.warning(f"Invalid integer for {env_key}={val!r}, using default {default}")
                return default

        return cls(
            name=os.getenv("MCP_SERVER_NAME", cls.name),
            host=os.getenv("MCP_HOST", cls.host),
            port=_safe_int("MCP_PORT", cls.port),
            mount_path=os.getenv("MCP_MOUNT_PATH", cls.mount_path),
            debug=os.getenv("MCP_DEBUG", "").lower() in ("1", "true", "yes"),
            enable_p2p_tools=os.getenv("MCP_DISABLE_P2P", "").lower() not in ("1", "true", "yes"),
        )


class TrioMCPServer:
    """Trio-native MCP server for P2P operations.

    This server runs entirely on Trio's event loop, eliminating the need for
    asyncio-to-Trio bridges that add latency to P2P operations.

    The server supports:
    - All 20 P2P tools (14 taskqueue + 6 workflow)
    - Structured concurrency with Trio nurseries
    - Graceful shutdown with cancel scopes
    - Hypercorn ASGI integration

    Example:
        >>> import trio
        >>> from ipfs_accelerate_py.mcplusplus_module.trio import TrioMCPServer
        >>>
        >>> async def main():
        ...     server = TrioMCPServer()
        ...     await server.run()
        ...
        >>> trio.run(main)
    """

    def __init__(self, config: Optional[ServerConfig] = None, name: Optional[str] = None):
        """Initialize the Trio MCP server.

        Args:
            config: Server configuration (uses defaults if None)
            name: Server name (overrides config.name if provided)
        """
        self.config = config or ServerConfig()
        if name:
            self.config.name = name

        # Configure logging
        log_format = os.environ.get("MCP_LOG_FORMAT", "text")
        if log_format == "json":
            import json as _json

            class JSONFormatter(logging.Formatter):
                def format(self, record):
                    return _json.dumps({
                        "ts": self.formatTime(record),
                        "level": record.levelname,
                        "logger": record.name,
                        "msg": record.getMessage(),
                        "module": record.module,
                    })

            handler = logging.StreamHandler()
            handler.setFormatter(JSONFormatter())
            logging.getLogger("ipfs_accelerate_mcp").addHandler(handler)
            logging.getLogger("ipfs_accelerate_mcp").propagate = False

        if self.config.debug:
            logging.getLogger("ipfs_accelerate_mcp.mcplusplus").setLevel(logging.DEBUG)

        # Server state
        self.mcp = None
        self.fastapi_app = None
        self._nursery: Optional[trio.Nursery] = None
        self._cancel_scope: Optional[trio.CancelScope] = None
        self._started = False

        logger.info(f"Initialized TrioMCPServer: {self.config.name}")

    def setup(self) -> None:
        """Set up the MCP server with tools and resources.

        This initializes the MCP instance and registers all configured tools.
        Must be called before run().
        """
        logger.info(f"Setting up TrioMCPServer: {self.config.name}")

        try:
            # Try to import FastMCP
            try:
                from fastmcp import FastMCP

                self.mcp = FastMCP(name=self.config.name)
                logger.info("Using FastMCP for Trio server")
            except ImportError:
                # Fallback to standalone implementation from the main mcp module
                logger.warning("FastMCP not available, using standalone implementation")
                from ipfs_accelerate_py.mcp.server import StandaloneMCP

                self.mcp = StandaloneMCP(name=self.config.name)

            # Register P2P tools if enabled
            if self.config.enable_p2p_tools:
                self._register_p2p_tools()

            # Register the broader IPFS Accelerate MCP tools for feature parity
            # (unified kit wrappers + legacy tools). Skip p2p taskqueue tools
            # here to avoid duplicate registrations; MCP++ already registers
            # the dedicated P2P tool set above.
            try:
                from ipfs_accelerate_py.mcp.tools import register_all_tools

                register_all_tools(self.mcp, include_p2p_taskqueue_tools=False)
                logger.info("Registered core ipfs_accelerate_py MCP tools")
            except Exception as e:
                logger.warning(f"Core MCP tools not registered: {e}")

            # Register core resources for parity with the primary MCP server.
            try:
                from ipfs_accelerate_py.mcp.resources import register_all_resources

                register_all_resources(self.mcp)
                logger.info("Registered core ipfs_accelerate_py MCP resources")
            except Exception as e:
                logger.warning(f"Core MCP resources not registered: {e}")

            # Register default prompts for parity (prompts are optional).
            try:
                if self.mcp is None:
                    raise RuntimeError("MCP server not initialized")

                try:
                    from ipfs_accelerate_py.mcp.fastmcp_compat import ensure_register_prompt_compat

                    ensure_register_prompt_compat(self.mcp)
                except Exception as e:
                    logger.debug(f"FastMCP prompt compatibility shim not applied: {e}")

                self.mcp.register_prompt(
                    name="ipfs_help",
                    template="""
                    # IPFS Accelerate Help

                    IPFS Accelerate provides tools and resources for working with IPFS and accelerating AI models.

                    ## Available Tools

                    {% for tool_name, tool in server.tools.items() %}
                    - **{{ tool_name }}**: {{ tool.description }}
                    {% endfor %}

                    ## Available Resources

                    {% for uri, resource in server.resources.items() %}
                    - **{{ uri }}**: {{ resource.description }}
                    {% endfor %}
                    """,
                    description="Get help with IPFS Accelerate",
                    input_schema={
                        "type": "object",
                        "properties": {},
                        "required": [],
                    },
                )
                logger.info("Registered core ipfs_accelerate_py MCP prompts")
            except Exception as e:
                logger.debug(f"Core MCP prompts not registered: {e}")

            # Create FastAPI app for ASGI
            self.fastapi_app = self._create_fastapi_app()

            logger.info(f"TrioMCPServer setup complete: {self.config.name}")

        except Exception as e:
            logger.error(f"Error setting up TrioMCPServer: {e}")
            raise

    def _register_p2p_tools(self) -> None:
        """Register all P2P tools with the MCP server."""
        logger.info("Registering P2P tools for Trio server")

        try:
            register_p2p_taskqueue_tools, register_p2p_workflow_tools = self._resolve_p2p_registrars()

            # Register tools based on configuration
            if self.config.enable_taskqueue_tools and self.config.enable_workflow_tools:
                register_p2p_taskqueue_tools(self.mcp)
                register_p2p_workflow_tools(self.mcp)
                logger.info("Registered all 20 P2P tools")
            else:
                # Register selectively
                if self.config.enable_taskqueue_tools:
                    register_p2p_taskqueue_tools(self.mcp)
                    logger.info("Registered 14 taskqueue tools")

                if self.config.enable_workflow_tools:
                    register_p2p_workflow_tools(self.mcp)
                    logger.info("Registered 6 workflow tools")

        except Exception as e:
            logger.error(f"Error registering P2P tools: {e}")
            raise

    def _resolve_p2p_registrars(self):
        """Resolve P2P registrar callables used by Trio MCP server.

        Kept as a dedicated hook so registration wiring can be validated via
        targeted unit tests without requiring full runtime setup.
        """
        from .. import tools as tools_module

        return tools_module._resolve_p2p_registrars()

    def _create_fastapi_app(self) -> Any:
        """Create the FastAPI application for ASGI.

        Returns:
            FastAPI application instance
        """
        try:
            if self.mcp is None:
                raise RuntimeError("MCP server not initialized")

            mcp = self.mcp

            from fastapi import FastAPI
            from fastapi.middleware.cors import CORSMiddleware

            # Create FastAPI app
            if hasattr(mcp, "create_fastapi_app"):
                # FastMCP provides this method
                app = mcp.create_fastapi_app(
                    title="IPFS Accelerate MCP++ API (Trio)",
                    description="Trio-native MCP server with P2P capabilities",
                    version="0.1.0",
                    docs_url="/docs",
                    redoc_url="/redoc",
                    mount_path=self.config.mount_path,
                )
            else:
                # Create manually for standalone
                app = FastAPI(
                    title="IPFS Accelerate MCP++ API (Trio)",
                    description="Trio-native MCP server with P2P capabilities",
                    version="0.1.0",
                    docs_url="/docs",
                    redoc_url="/redoc",
                )

            # Enable CORS — default to localhost origins for security;
            # set MCP_CORS_ORIGINS=* in production if needed
            allowed_origins = os.getenv(
                "MCP_CORS_ORIGINS",
                "http://localhost:8765,http://localhost:3000,http://127.0.0.1:8765"
            )
            origins = [o.strip() for o in allowed_origins.split(",") if o.strip()]

            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            logger.info(f"CORS enabled for origins: {origins}")

            # Rate limiting middleware
            rate_limiter = TokenBucketRateLimiter()

            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.responses import JSONResponse as StarletteJSONResponse

            class RateLimitMiddleware(BaseHTTPMiddleware):
                async def dispatch(self, request, call_next):
                    # Skip rate limiting for health endpoints
                    if request.url.path in ("/health", "/ready", "/live", "/metrics"):
                        return await call_next(request)
                    client_ip = request.client.host if request.client else "unknown"
                    if not rate_limiter.allow(client_ip):
                        try:
                            from ..metrics import get_metrics
                            get_metrics().rate_limit_rejected.inc()
                        except Exception:
                            pass
                        return StarletteJSONResponse(
                            status_code=429,
                            content={"error": "Too Many Requests", "retry_after_ms": 1000},
                        )
                    return await call_next(request)

            app.add_middleware(RateLimitMiddleware)

            # Health/readiness/liveness endpoints
            @app.get("/health")
            async def health_check():
                """Liveness probe — server is running."""
                return {
                    "status": "healthy",
                    "timestamp": time.time(),
                    "version": "1.0.0",
                    "service": self.config.name,
                }

            @app.get("/ready")
            async def readiness_check():
                """Readiness probe — server is ready to accept traffic."""
                from ..cid_ucan import get_event_dag
                dag = get_event_dag()
                dag_state = dag.to_dict()

                p2p_ready = False
                try:
                    from ..p2p_transport import get_p2p_node
                    node = get_p2p_node()
                    p2p_ready = getattr(node, '_operational', node._started)
                except Exception:
                    pass

                return {
                    "status": "ready" if self._started else "starting",
                    "service": self.config.name,
                    "dag_events": {
                        "hot": dag_state.get("hot_events", dag_state.get("total_events", 0)),
                        "total": dag_state.get("total_events", 0),
                    },
                    "services": {
                        "http": "available",
                        "p2p": "available" if p2p_ready else "degraded",
                    },
                }

            @app.get("/live")
            async def liveness_check():
                """Kubernetes liveness — process is alive."""
                return {"status": "alive"}

            @app.get("/api/mcp/status")
            async def api_mcp_status():
                """Frontend-compatible status endpoint (Hallucinate App / SwissKnife).

                Returns combined health + readiness for Electron dashboard.
                """
                from ..cid_ucan import get_event_dag
                dag = get_event_dag()
                dag_state = dag.to_dict()

                p2p_status = "disabled"
                try:
                    from ..p2p_transport import get_p2p_node
                    node = get_p2p_node()
                    p2p_status = "active" if getattr(node, '_operational', False) else "degraded"
                except Exception:
                    pass

                return {
                    "status": "ready" if self._started else "starting",
                    "service": self.config.name,
                    "version": "0.1.0",
                    "uptime": time.time() - self._start_time if hasattr(self, '_start_time') else 0,
                    "dag_events": dag_state.get("total_events", 0),
                    "p2p": p2p_status,
                    "tools": len(self.mcp.tools) if hasattr(self.mcp, 'tools') else 0,
                }

            @app.get("/tools/list")
            async def tools_list():
                """List all registered tools (frontend-compatible endpoint).

                Returns tool names with descriptions for UI rendering.
                """
                tools = []
                if hasattr(self.mcp, 'tools'):
                    for name, tool in self.mcp.tools.items():
                        tool_info = {"name": name}
                        if hasattr(tool, 'description'):
                            tool_info["description"] = tool.description
                        if hasattr(tool, 'inputSchema'):
                            tool_info["inputSchema"] = tool.inputSchema
                        elif hasattr(tool, 'input_schema'):
                            tool_info["inputSchema"] = tool.input_schema
                        tools.append(tool_info)
                return {"tools": tools}

            @app.get("/metrics")
            async def prometheus_metrics():
                """Prometheus metrics endpoint."""
                from starlette.responses import Response
                from ..metrics import get_metrics
                metrics = get_metrics()

                # Update dynamic gauges
                try:
                    from ..cid_ucan import get_event_dag, get_evaluator
                    dag = get_event_dag()
                    dag_state = dag.to_dict()
                    metrics.dag_events_total.set(dag_state.get("total_events", 0))
                    metrics.dag_hot_events.set(dag_state.get("hot_events", dag_state.get("total_events", 0)))
                    compaction = dag_state.get("compaction", {})
                    metrics.dag_compaction_epochs.set(compaction.get("epochs_compacted", 0))

                    evaluator = get_evaluator()
                    metrics.ucan_delegations_total.set(len(evaluator._store))
                    metrics.ucan_revocations_total.set(len(evaluator._revoked))
                except Exception:
                    pass

                try:
                    from ..p2p_transport import get_p2p_node
                    node = get_p2p_node()
                    metrics.p2p_peers_connected.set(len(node._peers))
                except Exception:
                    pass

                return Response(
                    content=metrics.collect_all(),
                    media_type="text/plain; version=0.0.4; charset=utf-8",
                )

            # Register MCP++ profile endpoints
            self._register_mcppp_routes(app)

            return app

        except Exception as e:
            logger.error(f"Error creating FastAPI app: {e}")
            raise

    def _register_mcppp_routes(self, app: Any) -> None:
        """Register MCP++ Profile B/C/D HTTP endpoints on the FastAPI app."""
        from fastapi import Request
        from fastapi.responses import JSONResponse

        @app.get("/mcp/interfaces")
        async def list_interfaces():
            """Profile A: List all registered interface descriptors with full schemas."""
            from ..interface_descriptor import InterfaceDescriptor, MethodDescriptor, get_interface_repository

            repo = get_interface_repository()
            tools = list(self.mcp.tools.keys()) if hasattr(self.mcp, 'tools') else []

            # Auto-populate repository if empty
            if len(repo._descriptors) == 0 and tools:
                for tool_name in tools:
                    tool_fn = self.mcp.tools[tool_name] if hasattr(self.mcp, 'tools') else None
                    # Extract schema from tool function if available
                    input_schema = {}
                    if tool_fn and hasattr(tool_fn, '__annotations__'):
                        input_schema = {
                            k: str(v) for k, v in tool_fn.__annotations__.items()
                            if k != 'return'
                        }
                    method = MethodDescriptor(
                        name=tool_name,
                        description=getattr(tool_fn, '__doc__', '') or '',
                        input_schema=input_schema,
                    )
                    descriptor = InterfaceDescriptor(
                        name=tool_name,
                        methods=[method],
                        author="ipfs_accelerate_py",
                    )
                    repo.register(descriptor)

            return {
                "interfaces": [d.to_dict() for d in repo._descriptors.values()],
                "count": len(repo._descriptors),
            }

        @app.post("/mcp/execute")
        async def execute_envelope(request: Request):
            """Profile B: Execute with CID-native envelope + UCAN + Policy enforcement."""
            from ..cid_ucan import execute_with_envelope, get_evaluator
            from ..temporal_policy import get_policy_evaluator

            body = await request.json()
            method = body.get("method", "")
            params = body.get("params", {})
            requester = body.get("requester", "")
            delegation_cid = body.get("delegation_cid")
            policy_cid = body.get("policy_cid")

            # Profile D: Evaluate temporal policy BEFORE execution
            if policy_cid:
                try:
                    policy_eval = get_policy_evaluator()
                    decision = policy_eval.evaluate(
                        method=method, actor=requester or "*",
                        resource=f"mcp://tool/{method}", policy_cid=policy_cid,
                    )
                    if decision.verdict not in ("allow", "allow_with_obligations"):
                        return JSONResponse(
                            status_code=403,
                            content={"error": f"Policy denied: {decision.justification}",
                                     "decision": decision.to_dict()},
                        )
                except Exception as e:
                    # Fail-closed: if policy evaluation crashes, deny the request
                    logger.error("Policy evaluation error (fail-closed): %s", e)
                    return JSONResponse(
                        status_code=403,
                        content={"error": "Policy evaluation failed (fail-closed)",
                                 "detail": str(e)},
                    )

            async def _execute(m, p):
                if hasattr(self.mcp, 'tools') and m in self.mcp.tools:
                    return await self.mcp.tools[m](**p)
                raise ValueError(f"Tool not found: {m}")

            envelope = await execute_with_envelope(
                method=method, params=params, requester=requester,
                delegation_cid=delegation_cid, executor_fn=_execute,
            )
            return envelope.to_dict()

        @app.get("/mcp/dag/frontier")
        async def dag_frontier():
            """Profile B: Get Event DAG frontier (leaf nodes)."""
            from ..cid_ucan import get_event_dag
            dag = get_event_dag()
            frontier = dag.frontier()
            return {"frontier": [{"cid": e.cid, "type": e.event_type, "timestamp": e.timestamp} for e in frontier]}

        @app.get("/mcp/dag/history")
        async def dag_history(limit: int = 50):
            """Profile B: Get recent Event DAG history."""
            from ..cid_ucan import get_event_dag
            dag = get_event_dag()
            events = dag.history(limit=limit)
            return {"events": [{"cid": e.cid, "type": e.event_type, "parents": e.parent_cids, "timestamp": e.timestamp} for e in events]}

        @app.get("/mcp/dag/provenance/{cid}")
        async def dag_provenance(cid: str):
            """Profile B: Trace provenance for a CID."""
            from ..cid_ucan import get_event_dag
            dag = get_event_dag()
            chain = dag.provenance(cid)
            return {"provenance": [{"cid": e.cid, "type": e.event_type, "parents": e.parent_cids} for e in chain]}

        @app.post("/mcp/ucan/delegate")
        async def ucan_delegate(request: Request):
            """Profile C: Create a new UCAN delegation with signature verification."""
            from ..cid_ucan import Delegation, Capability, get_evaluator

            body = await request.json()
            caps = [Capability(resource=c["resource"], ability=c["ability"]) for c in body.get("capabilities", [])]
            delegation = Delegation(
                issuer=body.get("issuer", ""),
                audience=body.get("audience", ""),
                capabilities=caps,
                expiry=body.get("expiry", 0.0),
                not_before=body.get("not_before", 0.0),
                proof_cids=body.get("proof_cids", []),
                signature=body.get("signature"),
            )

            evaluator = get_evaluator()

            # Verify signature if present (fail-closed: signed delegations must verify)
            if delegation.signature:
                if not evaluator._verify_signature(delegation):
                    return JSONResponse(
                        status_code=401,
                        content={"error": "Invalid signature on delegation", "issuer": delegation.issuer},
                    )

            # Verify proof chain (parent delegations must exist)
            for proof_cid in delegation.proof_cids:
                if proof_cid not in evaluator._store:
                    return JSONResponse(
                        status_code=400,
                        content={"error": f"Proof CID not found: {proof_cid}"},
                    )

            evaluator.add(delegation)
            # Write-through: persist immediately to survive crashes
            try:
                storage_dir = os.environ.get("MCPPP_STORAGE_DIR",
                                             os.path.expanduser("~/.ipfs_accelerate/state"))
                await trio.to_thread.run_sync(
                    lambda: evaluator.save_delegations(os.path.join(storage_dir, "delegations.json"))
                )
            except Exception:
                pass  # Best-effort; will be saved on shutdown
            return {"cid": delegation.cid, "delegation": delegation.to_dict()}

        @app.post("/mcp/ucan/revoke")
        async def ucan_revoke(request: Request):
            """Profile C: Revoke a UCAN delegation."""
            from ..cid_ucan import get_evaluator

            body = await request.json()
            delegation_cid = body.get("delegation_cid", "")
            if not delegation_cid:
                return JSONResponse(status_code=400, content={"error": "Missing delegation_cid"})

            evaluator = get_evaluator()
            evaluator.revoke(delegation_cid)
            return {"revoked": delegation_cid, "status": "ok"}

        @app.post("/mcp/ucan/validate")
        async def ucan_validate(request: Request):
            """Profile C: Validate a UCAN delegation chain."""
            from ..cid_ucan import get_evaluator

            body = await request.json()
            evaluator = get_evaluator()
            authorized, reason = evaluator.can_invoke(
                leaf_cid=body.get("delegation_cid", ""),
                resource=body.get("resource", ""),
                ability=body.get("ability", "invoke"),
                actor=body.get("actor"),
            )
            return {"authorized": authorized, "reason": reason}

        @app.get("/mcp/services")
        async def list_services():
            """Service registry: List all known local and remote services."""
            try:
                from ..service_registry import get_service_registry
                registry = get_service_registry()
                return registry.to_dict()
            except Exception as e:
                return {"local_services": {}, "remote_services": {}, "error": str(e)}

        @app.get("/mcp/p2p/peers")
        async def p2p_peers():
            """Profile E: List connected P2P peers."""
            from ..p2p_transport import get_p2p_node
            node = get_p2p_node()
            return node.to_dict()

        @app.post("/mcp/p2p/call")
        async def p2p_call_tool(request: Request):
            """Profile E: Call a tool on a remote peer with UCAN+Policy enforcement."""
            from ..p2p_transport import get_p2p_node
            from ..cid_ucan import get_evaluator

            body = await request.json()
            peer_id = body.get("peer_id", "")
            method = body.get("method", "")
            params = body.get("params", {})
            delegation_cid = body.get("delegation_cid")
            policy_cid = body.get("policy_cid")
            actor = body.get("actor", "")

            # Enforce UCAN delegation if provided
            if delegation_cid:
                evaluator = get_evaluator()
                try:
                    authorized, reason = evaluator.can_invoke(
                        delegation_cid, f"mcp://tool/{method}", "invoke", actor=actor or None
                    )
                except ValueError as e:
                    logger.warning("UCAN chain validation failed (possible DoS): %s", e)
                    return JSONResponse(
                        status_code=400,
                        content={"error": "Invalid delegation chain", "code": "CHAIN_INVALID",
                                 "detail": str(e)},
                    )
                if not authorized:
                    return JSONResponse(
                        status_code=403,
                        content={"error": f"UCAN unauthorized: {reason}"},
                    )

            # Enforce policy if provided
            if policy_cid:
                from ..temporal_policy import get_policy_evaluator
                policy_eval = get_policy_evaluator()
                decision = policy_eval.evaluate(
                    method=method, actor=actor or "*",
                    resource=f"mcp://tool/{method}", policy_cid=policy_cid,
                )
                if decision.verdict not in ("allow", "allow_with_obligations"):
                    return JSONResponse(
                        status_code=403,
                        content={"error": f"Policy denied: {decision.justification}"},
                    )

            node = get_p2p_node()
            if not node._started or not getattr(node, '_operational', False):
                return JSONResponse(
                    status_code=503,
                    content={"error": "P2P service not available",
                             "code": "SERVICE_DEGRADED",
                             "fallback": "Use /mcp/execute for local tool invocation"},
                )
            try:
                result = await node.call_tool(
                    peer_id=peer_id,
                    method=method,
                    params=params,
                    timeout=body.get("timeout", 30.0),
                )
                return {"result": result, "peer_id": peer_id, "method": method}
            except TimeoutError as e:
                return JSONResponse(status_code=504, content={"error": str(e), "code": "TIMEOUT"})
            except ConnectionError as e:
                return JSONResponse(status_code=502, content={"error": str(e), "code": "CONNECTION_ERROR"})
            except RuntimeError as e:
                return JSONResponse(status_code=502, content={"error": str(e), "code": "REMOTE_ERROR"})
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": str(e), "code": "INTERNAL_ERROR"})

        @app.post("/mcp/policy/evaluate")
        async def policy_evaluate(request: Request):
            """Profile D: Evaluate a temporal deontic policy."""
            from ..temporal_policy import get_policy_evaluator
            body = await request.json()
            evaluator = get_policy_evaluator()
            decision = evaluator.evaluate(
                method=body.get("method", ""),
                actor=body.get("actor", "*"),
                resource=body.get("resource"),
                policy_cid=body.get("policy_cid"),
            )
            return decision.to_dict()

        @app.post("/mcp/policy/register")
        async def policy_register(request: Request):
            """Profile D: Register a new temporal deontic policy."""
            from ..temporal_policy import get_policy_evaluator, PolicyObject, PolicyClause
            body = await request.json()
            clauses = [
                PolicyClause(
                    clause_type=c.get("clause_type", "permission"),
                    actor=c.get("actor", "*"),
                    action=c.get("action", "*"),
                    resource=c.get("resource"),
                    valid_from=c.get("valid_from"),
                    valid_until=c.get("valid_until"),
                    obligation_deadline=c.get("obligation_deadline"),
                    metadata=c.get("metadata", {}),
                )
                for c in body.get("clauses", [])
            ]
            policy = PolicyObject(
                name=body.get("name", "unnamed"),
                clauses=clauses,
                description=body.get("description", ""),
            )
            evaluator = get_policy_evaluator()
            cid = evaluator.register(policy)
            # Write-through: persist immediately to survive crashes
            try:
                storage_dir = os.environ.get("MCPPP_STORAGE_DIR",
                                             os.path.expanduser("~/.ipfs_accelerate/state"))
                await trio.to_thread.run_sync(
                    lambda: evaluator.save_policies(os.path.join(storage_dir, "policies.json"))
                )
            except Exception:
                pass  # Best-effort; will be saved on shutdown
            return {"cid": cid, "policy": policy.to_dict()}

        @app.get("/mcp/discover")
        async def mcp_discover():
            """Discovery endpoint: returns server capabilities, version, available tools."""
            from ..interface_descriptor import get_interface_repository

            tools = list(self.mcp.tools.keys()) if hasattr(self.mcp, 'tools') else []
            repo = get_interface_repository()

            profiles = {
                "A": "MCP-IDL (Interface Descriptors)",
                "B": "CID-Native Execution (Intent/Decision/Receipt)",
                "C": "UCAN Authorization (Delegation Chains)",
                "D": "Temporal Deontic Policy (Permission/Prohibition/Obligation)",
                "E": "mcp+p2p (libp2p Transport)",
            }

            p2p_status = "disabled"
            peer_id = None
            try:
                from ..p2p_transport import get_p2p_node
                node = get_p2p_node()
                if node._started:
                    p2p_status = "active"
                    peer_id = node.peer_id
            except Exception:
                pass

            return {
                "server": self.config.name,
                "version": "0.1.0",
                "protocol": "mcp++",
                "profiles": profiles,
                "tools": tools,
                "interfaces": len(repo._descriptors),
                "p2p": {"status": p2p_status, "peer_id": peer_id},
                "endpoints": {
                    "jsonrpc": "/mcp",
                    "execute": "/mcp/execute",
                    "interfaces": "/mcp/interfaces",
                    "ucan_delegate": "/mcp/ucan/delegate",
                    "ucan_revoke": "/mcp/ucan/revoke",
                    "policy_evaluate": "/mcp/policy/evaluate",
                    "policy_register": "/mcp/policy/register",
                    "p2p_peers": "/mcp/p2p/peers",
                    "p2p_call": "/mcp/p2p/call",
                    "services": "/mcp/services",
                    "events": "/mcp/events/stream",
                    "health": "/health",
                    "metrics": "/metrics",
                },
                "auth": {
                    "ucan_required": not bool(os.environ.get("MCPPP_ALLOW_UNSIGNED_DELEGATIONS")),
                },
            }

        @app.get("/mcp/events/stream")
        async def event_stream(request: Request):
            """SSE endpoint: streams EventDAG changes and server events in real-time.

            Frontend connects here for live updates (tool executions, peer events, etc).
            Uses Server-Sent Events (SSE) for broad compatibility with Electron/browsers.
            """
            from starlette.responses import StreamingResponse
            from ..cid_ucan import get_event_dag

            async def _generate_events():
                """Async generator yielding SSE events with disconnect detection."""
                dag = get_event_dag()
                last_count = len(dag._events)

                # Send initial connection event
                yield f"event: connected\ndata: {{\"server\": \"{self.config.name}\", \"events\": {last_count}}}\n\n"

                # Poll for new events (every 1s) — SSE keepalive
                try:
                    while True:
                        await trio.sleep(1.0)
                        # Check if client disconnected
                        if await request.is_disconnected():
                            break
                        current_count = len(dag._events)
                        if current_count > last_count:
                            # New events appended
                            new_events = list(dag._events.values())[last_count:current_count]
                            for event in new_events:
                                event_data = json.dumps({
                                    "cid": event.cid,
                                    "type": event.event_type,
                                    "timestamp": event.timestamp,
                                    "parents": event.parent_cids,
                                })
                                yield f"event: dag_event\ndata: {event_data}\n\n"
                            last_count = current_count
                        else:
                            # Keepalive
                            yield f": keepalive\n\n"
                except (trio.Cancelled, GeneratorExit):
                    pass

            return StreamingResponse(
                _generate_events(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-Accel-Buffering": "no",
                },
            )

        @app.post("/mcp")
        async def jsonrpc_handler(request: Request):
            """JSON-RPC 2.0 endpoint with MCP++ negotiation."""
            body = await request.json()
            result = await self._handle_jsonrpc(body)
            return JSONResponse(content=result)

    async def _startup(self) -> None:
        """Server startup hook.

        Called when the server starts. Use this for initialization that
        requires Trio context (e.g., opening resources, starting background tasks).
        """
        logger.info(f"Starting TrioMCPServer on {self.config.host}:{self.config.port}")
        self._started = True

        # Recover persisted EventDAG state
        try:
            import os
            import json as _json
            state_dir = os.path.expanduser("~/.ipfs_accelerate/state")
            dag_path = os.path.join(state_dir, "event_dag.json")
            if os.path.isfile(dag_path):
                def _load_dag():
                    with open(dag_path, "r") as f:
                        return _json.load(f)

                from ..cid_ucan import get_event_dag, DAGEvent
                data = await trio.to_thread.run_sync(_load_dag)
                dag = get_event_dag()
                events = data.get("events", {})
                loaded = 0
                for cid, info in events.items():
                    if cid not in dag._events:
                        dag.append(DAGEvent(
                            cid=cid,
                            event_type=info.get("type", "unknown"),
                            parent_cids=info.get("parents", []),
                            timestamp=info.get("timestamp", 0),
                        ))
                        loaded += 1
                if loaded > 0:
                    logger.info(f"EventDAG recovered: {loaded} events from disk")
        except Exception as e:
            logger.warning(f"EventDAG recovery failed (non-fatal): {e}")

        # Recover persisted revocation list
        try:
            import os
            state_dir = os.path.expanduser("~/.ipfs_accelerate/state")
            revoc_path = os.path.join(state_dir, "revocations.json")
            if os.path.isfile(revoc_path):
                from ..cid_ucan import get_evaluator

                def _load_revocations():
                    evaluator = get_evaluator()
                    return evaluator.load_revocations(revoc_path)

                count = await trio.to_thread.run_sync(_load_revocations)
                if count > 0:
                    logger.info(f"Revocations recovered: {count} entries from disk")
        except Exception as e:
            logger.debug(f"Revocation recovery: {e}")

        # Start P2P node if enabled
        if self.config.enable_p2p_tools and self._nursery:
            try:
                from ..p2p_transport import get_p2p_node
                node = get_p2p_node()
                await node.start(self._nursery)

                # Wait briefly for P2P to become operational
                deadline = time.time() + 5.0
                while not getattr(node, '_operational', False) and time.time() < deadline:
                    await trio.sleep(0.1)

                if not getattr(node, '_operational', False):
                    logger.warning("P2P node started but not yet operational (will continue in background)")

                # Register our MCP tools as the P2P tool handler
                if hasattr(self.mcp, 'tools'):
                    tools_list = list(self.mcp.tools.keys())

                    async def _handle_p2p_tool(method, params):
                        # Handle service announcements
                        if method == "_mcppp_service_announce":
                            from ..service_registry import get_service_registry
                            registry = get_service_registry()
                            sender = params.get("_sender_peer_id", "")
                            return registry.handle_announce(params, sender_peer_id=sender)
                        if method in self.mcp.tools:
                            return await self.mcp.tools[method](**params)
                        raise ValueError(f"Tool not found: {method}")
                    node.set_tool_handler(_handle_p2p_tool)

                    # Register in service registry for cross-server discovery
                    try:
                        from ..service_registry import get_service_registry, ServiceRecord
                        registry = get_service_registry()
                        registry.register_local(ServiceRecord(
                            service_name="ipfs-accelerate-mcp",
                            peer_id=node.peer_id or "",
                            multiaddrs=node.multiaddrs or [],
                            tools=tools_list,
                            metadata={"port": self.config.port, "server": self.config.name},
                        ))
                        # Start service advertise loop
                        self._nursery.start_soon(registry.advertise_loop, node, self._nursery)
                    except Exception as e:
                        logger.debug(f"Service registry setup: {e}")

                logger.info(f"P2P node started: {node.peer_id}")

                # Start periodic peer discovery and health checks
                self._nursery.start_soon(self._p2p_maintenance_loop, node)
            except Exception as e:
                logger.warning(f"P2P node startup failed (non-fatal): {e}")

        # Start obligation enforcement loop (Profile D)
        if self._nursery:
            self._nursery.start_soon(self._obligation_enforcement_loop)

    async def _p2p_maintenance_loop(self, node) -> None:
        """Background loop for peer discovery, health checks, and reconnection."""
        backoff = 5.0  # Start with 5s between cycles
        max_backoff = 120.0

        while self._started:
            try:
                await trio.sleep(backoff)

                # Attempt peer discovery
                try:
                    discovered = await node.discover_peers()
                    if discovered:
                        logger.info(f"Discovered {len(discovered)} peers via mDNS")
                        backoff = 5.0  # Reset backoff on success
                except Exception as e:
                    logger.debug(f"Peer discovery cycle: {e}")

                # Health-check existing peers (remove stale ones)
                stale_peers = []
                peers_snapshot = list(node._peers.items())
                for peer_id, info in peers_snapshot:
                    if time.time() - info.last_seen > 300:  # 5 min stale threshold
                        stale_peers.append(peer_id)
                for peer_id in stale_peers:
                    node._peers.pop(peer_id, None)
                    logger.debug(f"Removed stale peer: {peer_id}")

                # Reconnect to bootstrap peers if we have no active peers
                if not node._peers:
                    for peer_addr in node._bootstrap_peers:
                        try:
                            await node._connect_bootstrap(peer_addr)
                        except Exception:
                            pass
                    backoff = min(backoff * 1.5, max_backoff)

            except trio.Cancelled:
                break
            except Exception as e:
                logger.debug(f"P2P maintenance error: {e}")
                backoff = min(backoff * 2, max_backoff)

    async def _obligation_enforcement_loop(self) -> None:
        """Background loop that checks for overdue obligations (Profile D enforcement)."""
        from ..temporal_policy import get_obligation_tracker
        tracker = get_obligation_tracker()

        while self._started:
            try:
                await trio.sleep(30.0)
                overdue = tracker.get_overdue()
                if overdue:
                    for ob in overdue:
                        logger.warning(
                            "OBLIGATION VIOLATION: id=%s action=%s deadline=%s actor=%s",
                            ob.obligation_id[:16], ob.action, ob.deadline, ob.actor,
                        )
                    # Emit metrics
                    try:
                        from ..metrics import get_metrics_registry
                        metrics = get_metrics_registry()
                        if hasattr(metrics, 'obligations_overdue'):
                            metrics.obligations_overdue.set(len(overdue))
                    except Exception:
                        pass
            except trio.Cancelled:
                break
            except Exception as e:
                logger.debug(f"Obligation enforcement error: {e}")

    async def _shutdown(self) -> None:
        """Server shutdown hook.

        Called when the server is shutting down. Use this for cleanup
        (e.g., closing resources, stopping background tasks).
        """
        logger.info("Shutting down TrioMCPServer")

        # Stop P2P node
        try:
            from ..p2p_transport import get_p2p_node
            node = get_p2p_node()
            if node._started:
                await node.stop()
        except Exception as e:
            logger.debug(f"P2P node shutdown: {e}")

        # Persist EventDAG to disk (non-blocking)
        try:
            from ..cid_ucan import get_event_dag
            dag = get_event_dag()
            dag_state = dag.to_dict(include_events=True)
            if dag_state["total_events"] > 0:
                import os
                import json as _json
                state_dir = os.path.expanduser("~/.ipfs_accelerate/state")

                def _persist_dag():
                    os.makedirs(state_dir, exist_ok=True)
                    with open(os.path.join(state_dir, "event_dag.json"), "w") as f:
                        _json.dump(dag_state, f)

                with trio.move_on_after(10):  # 10s timeout for persistence
                    await trio.to_thread.run_sync(_persist_dag)
                    logger.info(f"EventDAG persisted: {dag_state['total_events']} events")
        except Exception as e:
            logger.warning(f"EventDAG persistence failed: {e}")

        # Persist revocation list
        try:
            from ..cid_ucan import get_evaluator
            import os
            state_dir = os.path.expanduser("~/.ipfs_accelerate/state")

            def _persist_revocations():
                os.makedirs(state_dir, exist_ok=True)
                evaluator = get_evaluator()
                evaluator.save_revocations(os.path.join(state_dir, "revocations.json"))
                evaluator.save_delegations(os.path.join(state_dir, "delegations.json"))

            with trio.move_on_after(10):  # 10s timeout for persistence
                await trio.to_thread.run_sync(_persist_revocations)
                logger.info("Revocations and delegations persisted to disk")
        except Exception as e:
            logger.warning(f"UCAN state persistence failed: {e}")

        # Persist policies
        try:
            from ..temporal_policy import get_policy_evaluator
            import os
            state_dir = os.path.expanduser("~/.ipfs_accelerate/state")

            def _persist_policies():
                os.makedirs(state_dir, exist_ok=True)
                evaluator = get_policy_evaluator()
                evaluator.save_policies(os.path.join(state_dir, "policies.json"))

            with trio.move_on_after(10):
                await trio.to_thread.run_sync(_persist_policies)
                logger.info("Policies persisted to disk")
        except Exception as e:
            logger.warning(f"Policy persistence failed: {e}")

        self._started = False

    async def run(self, *, task_status=trio.TASK_STATUS_IGNORED) -> None:
        """Run the Trio MCP server with HTTP serving.

        This method runs the server using Trio's structured concurrency.
        It serves HTTP using Hypercorn (if available) or a built-in Trio TCP
        server for JSON-RPC over HTTP.

        Args:
            task_status: For use with nursery.start() to signal readiness

        Example standalone:
            >>> trio.run(server.run)

        Example with nursery:
            >>> async with trio.open_nursery() as nursery:
            ...     await nursery.start(server.run)
        """
        if not self.mcp:
            self.setup()

        async with trio.open_nursery() as nursery:
            self._nursery = nursery

            # Set up signal handling for graceful shutdown
            async def _wait_for_signal():
                import signal
                try:
                    with trio.open_signal_receiver(signal.SIGTERM, signal.SIGINT) as signal_aiter:
                        async for sig_num in signal_aiter:
                            logger.info(f"Received signal {sig_num}, initiating graceful shutdown...")
                            nursery.cancel_scope.cancel()
                            break
                except (OSError, NotImplementedError, AttributeError):
                    await trio.sleep_forever()  # Signals not available

            nursery.start_soon(_wait_for_signal)

            # Run startup hook
            await self._startup()

            try:
                # Try Hypercorn (preferred for production)
                try:
                    from hypercorn.trio import serve as hypercorn_serve
                    from hypercorn.config import Config as HypercornConfig

                    asgi_app = self.create_asgi_app()
                    hconfig = HypercornConfig()
                    hconfig.bind = [f"{self.config.host}:{self.config.port}"]
                    hconfig.worker_class = "trio"

                    logger.info(
                        "TrioMCPServer serving via Hypercorn at http://%s:%s%s",
                        self.config.host, self.config.port, self.config.mount_path,
                    )
                    # Signal ready just before serving (Hypercorn binds synchronously)
                    task_status.started()
                    await hypercorn_serve(asgi_app, hconfig)

                except ImportError:
                    # Fallback: built-in Trio TCP server with JSON-RPC handler
                    logger.info(
                        "Hypercorn not available. Starting built-in Trio JSON-RPC server at http://%s:%s%s",
                        self.config.host, self.config.port, self.config.mount_path,
                    )
                    # _serve_jsonrpc_trio signals started after socket binds
                    await self._serve_jsonrpc_trio(nursery, task_status)

            except trio.Cancelled:
                logger.info("TrioMCPServer cancelled")
                raise
            finally:
                # Run shutdown hook
                await self._shutdown()
                self._nursery = None

    async def _serve_jsonrpc_trio(self, nursery: trio.Nursery, task_status=None) -> None:
        """Built-in Trio TCP server for MCP JSON-RPC over HTTP.
        
        Handles basic HTTP/1.1 POST requests with JSON-RPC payloads.
        This is a minimal implementation for when Hypercorn is not available.
        Signals task_status.started() only after the socket is bound.
        """
        import json as _json

        async def handle_connection(stream: trio.SocketStream) -> None:
            try:
                data = b""
                while not data.endswith(b"\r\n\r\n"):
                    chunk = await stream.receive_some(4096)
                    if not chunk:
                        return
                    data += chunk

                # Parse HTTP request
                header_end = data.find(b"\r\n\r\n")
                headers_raw = data[:header_end].decode("utf-8", errors="replace")
                body_start = data[header_end + 4:]

                # Get content-length
                content_length = 0
                for line in headers_raw.split("\r\n"):
                    if line.lower().startswith("content-length:"):
                        content_length = int(line.split(":")[1].strip())

                # Read remaining body
                body = body_start
                while len(body) < content_length:
                    chunk = await stream.receive_some(4096)
                    if not chunk:
                        break
                    body += chunk

                # Route request
                first_line = headers_raw.split("\r\n")[0]
                method, path, _ = first_line.split(" ", 2)

                if method == "GET" and path == "/api/mcp/status":
                    response_body = _json.dumps({"status": "ok", "server": self.config.name, "protocol": "/mcp+p2p/1.0.0"})
                elif method == "GET" and path == "/api/mcp/tools":
                    tools = list(self.mcp.tools.keys()) if hasattr(self.mcp, 'tools') else []
                    response_body = _json.dumps({"tools": tools})
                elif method == "POST" and path == self.config.mount_path:
                    # JSON-RPC handler
                    request = _json.loads(body.decode("utf-8"))
                    response_body = _json.dumps(await self._handle_jsonrpc(request))
                elif method == "GET" and path == "/mcp/interfaces":
                    tools = list(self.mcp.tools.keys()) if hasattr(self.mcp, 'tools') else []
                    interfaces = [{"cid": compute_cid({"name": t}), "name": t, "version": "1.0.0"} for t in tools]
                    response_body = _json.dumps({"interfaces": interfaces, "count": len(interfaces)})
                elif method == "GET" and path == "/mcp/dag/frontier":
                    from ..cid_ucan import get_event_dag
                    dag = get_event_dag()
                    frontier = dag.frontier()
                    response_body = _json.dumps({"frontier": [{"cid": e.cid, "type": e.event_type, "timestamp": e.timestamp} for e in frontier]})
                elif method == "GET" and path == "/mcp/dag/history":
                    from ..cid_ucan import get_event_dag
                    dag = get_event_dag()
                    events = dag.history(limit=50)
                    response_body = _json.dumps({"events": [{"cid": e.cid, "type": e.event_type, "parents": e.parent_cids, "timestamp": e.timestamp} for e in events]})
                elif method == "GET" and path.startswith("/mcp/dag/provenance/"):
                    from ..cid_ucan import get_event_dag
                    target_cid = path.split("/mcp/dag/provenance/")[1]
                    dag = get_event_dag()
                    chain = dag.provenance(target_cid)
                    response_body = _json.dumps({"provenance": [{"cid": e.cid, "type": e.event_type, "parents": e.parent_cids} for e in chain]})
                elif method == "POST" and path == "/mcp/execute":
                    from ..cid_ucan import execute_with_envelope
                    req = _json.loads(body.decode("utf-8"))
                    async def _exec(m, p):
                        if hasattr(self.mcp, 'tools') and m in self.mcp.tools:
                            return await self.mcp.tools[m](**p)
                        raise ValueError(f"Tool not found: {m}")
                    envelope = await execute_with_envelope(
                        method=req.get("method", ""), params=req.get("params", {}),
                        requester=req.get("requester", ""), delegation_cid=req.get("delegation_cid"),
                        executor_fn=_exec,
                    )
                    response_body = _json.dumps(envelope.to_dict())
                elif method == "GET" and path == "/mcp/p2p/peers":
                    response_body = _json.dumps({"peers": [], "protocol": "/mcp+p2p/1.0.0"})
                else:
                    response_body = _json.dumps({"error": "Not found"})

                # Send HTTP response
                http_response = (
                    f"HTTP/1.1 200 OK\r\n"
                    f"Content-Type: application/json\r\n"
                    f"Content-Length: {len(response_body)}\r\n"
                    f"Access-Control-Allow-Origin: *\r\n"
                    f"\r\n"
                    f"{response_body}"
                )
                await stream.send_all(http_response.encode("utf-8"))
            except Exception as e:
                logger.debug(f"Connection handler error: {e}")
            finally:
                await stream.aclose()

        listeners = await nursery.start(
            trio.serve_tcp, handle_connection, self.config.port, host=self.config.host
        )
        logger.info(f"Trio TCP server listening on {self.config.host}:{self.config.port}")
        self._started = True
        # Signal readiness AFTER socket is bound
        if task_status is not None:
            task_status.started()
        await trio.sleep_forever()

    async def _handle_jsonrpc(self, request: dict) -> dict:
        """Handle a single JSON-RPC request with input validation and metrics."""
        # Validate request structure
        if not isinstance(request, dict):
            return {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32700, "message": "Parse error: request must be a JSON object"}}

        method = request.get("method")
        if not method or not isinstance(method, str):
            return {"jsonrpc": "2.0", "id": request.get("id"),
                    "error": {"code": -32600, "message": "Invalid Request: missing or non-string 'method'"}}

        req_id = request.get("id", 1)
        if req_id is not None and not isinstance(req_id, (str, int, float)):
            return {"jsonrpc": "2.0", "id": None,
                    "error": {"code": -32600, "message": "Invalid Request: 'id' must be string, number, or null"}}

        params = request.get("params", {})
        if not isinstance(params, (dict, list)):
            params = {}

        # Metrics instrumentation
        start_time = time.time()
        try:
            result = await self._dispatch_jsonrpc(method, params, req_id)
            status = "error" if "error" in result else "ok"
        except Exception as e:
            status = "error"
            result = {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(e)}}

        duration = time.time() - start_time
        try:
            from ..metrics import get_metrics
            metrics = get_metrics()
            metrics.requests_total.inc(method=method, status=status)
            metrics.request_duration.observe(duration, method=method)
        except Exception:
            pass

        return result

    async def _dispatch_jsonrpc(self, method: str, params: dict, req_id: Any) -> dict:
        """Dispatch a JSON-RPC method call."""

        if method == "initialize":
            client_caps = params.get("capabilities", {}).get("experimental", {})
            server_caps = {k: True for k in client_caps if k.startswith("mcp++")}
            return {
                "jsonrpc": "2.0", "id": req_id,
                "result": {
                    "protocolVersion": "2024-11-05",
                    "capabilities": {"tools": {"listChanged": True}, "experimental": server_caps},
                    "serverInfo": {"name": self.config.name, "version": "1.0.0"},
                },
            }
        elif method == "tools/call":
            tool_name = params.get("name", "")
            arguments = params.get("arguments", {})
            delegation_cid = params.get("_delegation_cid")  # MCP++ extension

            if hasattr(self.mcp, 'tools') and tool_name in self.mcp.tools:
                # Profile A: Validate params against declared schema
                try:
                    from ..interface_descriptor import validate_params
                    validation_error = validate_params(tool_name, arguments)
                    if validation_error:
                        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32602, "message": f"Schema validation: {validation_error}"}}
                except ImportError:
                    pass

                # Enforce UCAN delegation if provided
                if delegation_cid:
                    from ..cid_ucan import get_evaluator
                    evaluator = get_evaluator()
                    actor = params.get("_actor", "")
                    authorized, reason = evaluator.can_invoke(
                        delegation_cid, f"mcp://tool/{tool_name}", "invoke", actor=actor or None
                    )
                    if not authorized:
                        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32003, "message": f"Unauthorized: {reason}"}}

                try:
                    import inspect
                    tool_fn = self.mcp.tools[tool_name]
                    if inspect.iscoroutinefunction(tool_fn):
                        result = await tool_fn(**arguments)
                    else:
                        result = tool_fn(**arguments)
                    return {"jsonrpc": "2.0", "id": req_id, "result": result}
                except Exception as e:
                    return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(e)}}
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Tool not found: {tool_name}"}}
        elif method == "tools/list":
            tools = list(self.mcp.tools.keys()) if hasattr(self.mcp, 'tools') else []
            return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": tools}}
        elif method == "mcp++/p2p/peers":
            return {"jsonrpc": "2.0", "id": req_id, "result": {"peers": [], "protocol": "/mcp+p2p/1.0.0"}}
        elif method == "shutdown":
            return {"jsonrpc": "2.0", "id": req_id, "result": None}
        else:
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32601, "message": f"Method not found: {method}"}}

    def create_asgi_app(self) -> Any:
        """Create the ASGI application for Hypercorn.

        This method is used by Hypercorn to get the ASGI app.

        Returns:
            ASGI application (FastAPI app)

        Example:
            # In your deployment script:
            from ipfs_accelerate_py.mcplusplus_module.trio import TrioMCPServer

            server = TrioMCPServer()
            server.setup()
            app = server.create_asgi_app()

            # Then run with Hypercorn:
            # hypercorn --worker-class trio module:app
        """
        if not self.mcp:
            self.setup()

        return self.fastapi_app


# Factory function for Hypercorn deployment
def create_app() -> Any:
    """Factory function to create the ASGI app for Hypercorn.

    This is the entry point for Hypercorn deployment:
        hypercorn --worker-class trio ipfs_accelerate_py.mcplusplus_module.trio.server:create_app

    Configuration is loaded from environment variables (see ServerConfig.from_env).

    Returns:
        ASGI application ready for Hypercorn
    """
    config = ServerConfig.from_env()
    server = TrioMCPServer(config=config)
    server.setup()
    return server.create_asgi_app()


# Main entry point for standalone execution
async def main():
    """Main entry point for standalone Trio server execution.

    Example:
        python -m ipfs_accelerate_py.mcplusplus_module.trio.server

    Or:
        from ipfs_accelerate_py.mcplusplus_module.trio.server import main
        import trio
        trio.run(main)
    """
    config = ServerConfig.from_env()
    server = TrioMCPServer(config=config)
    await server.run()


if __name__ == "__main__":
    # Run the server when module is executed directly
    import sys

    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        stream=sys.stdout,
    )

    logger.info("Starting Trio MCP Server...")
    trio.run(main)


__all__ = [
    "TrioMCPServer",
    "ServerConfig",
    "create_app",
    "main",
]
