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

import logging
import os
from typing import Any, Optional
from dataclasses import dataclass

import trio

logger = logging.getLogger("ipfs_accelerate_mcp.mcplusplus.trio.server")


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
            MCP_PORT: Port to bind to
            MCP_MOUNT_PATH: API mount path
            MCP_DEBUG: Enable debug logging (1/true/yes)
            MCP_DISABLE_P2P: Disable P2P tools (1/true/yes)

        Returns:
            ServerConfig instance with values from environment
        """
        return cls(
            name=os.getenv("MCP_SERVER_NAME", cls.name),
            host=os.getenv("MCP_HOST", cls.host),
            port=int(os.getenv("MCP_PORT", str(cls.port))),
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

            # Enable CORS
            allowed_origins = os.getenv("MCP_CORS_ORIGINS", "*")
            origins = [o.strip() for o in allowed_origins.split(",") if o.strip()] or ["*"]

            app.add_middleware(
                CORSMiddleware,
                allow_origins=origins,
                allow_credentials=True,
                allow_methods=["*"],
                allow_headers=["*"],
            )

            logger.info(f"CORS enabled for origins: {origins}")

            return app

        except Exception as e:
            logger.error(f"Error creating FastAPI app: {e}")
            raise

    async def _startup(self) -> None:
        """Server startup hook.

        Called when the server starts. Use this for initialization that
        requires Trio context (e.g., opening resources, starting background tasks).
        """
        logger.info(f"Starting TrioMCPServer on {self.config.host}:{self.config.port}")
        self._started = True

    async def _shutdown(self) -> None:
        """Server shutdown hook.

        Called when the server is shutting down. Use this for cleanup
        (e.g., closing resources, stopping background tasks).
        """
        logger.info("Shutting down TrioMCPServer")
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
                    task_status.started()
                    await hypercorn_serve(asgi_app, hconfig)

                except ImportError:
                    # Fallback: built-in Trio TCP server with JSON-RPC handler
                    logger.info(
                        "Hypercorn not available. Starting built-in Trio JSON-RPC server at http://%s:%s%s",
                        self.config.host, self.config.port, self.config.mount_path,
                    )
                    task_status.started()
                    await self._serve_jsonrpc_trio(nursery)

            except trio.Cancelled:
                logger.info("TrioMCPServer cancelled")
                raise
            finally:
                # Run shutdown hook
                await self._shutdown()
                self._nursery = None

    async def _serve_jsonrpc_trio(self, nursery: trio.Nursery) -> None:
        """Built-in Trio TCP server for MCP JSON-RPC over HTTP.
        
        Handles basic HTTP/1.1 POST requests with JSON-RPC payloads.
        This is a minimal implementation for when Hypercorn is not available.
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
                    response_body = _json.dumps({"interfaces": [], "count": 0})
                elif method == "GET" and path.startswith("/mcp/dag"):
                    response_body = _json.dumps({"events": [], "frontier": []})
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
        await trio.sleep_forever()

    async def _handle_jsonrpc(self, request: dict) -> dict:
        """Handle a single JSON-RPC request."""
        method = request.get("method", "")
        params = request.get("params", {})
        req_id = request.get("id", 1)

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
            if hasattr(self.mcp, 'tools') and tool_name in self.mcp.tools:
                try:
                    result = await self.mcp.tools[tool_name](**arguments)
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
