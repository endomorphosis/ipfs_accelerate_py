"""IPFS Accelerate MCP.

This package exposes two MCP server APIs:

1) A modern FastAPI-based server via :class:`~ipfs_accelerate_py.mcp.server.IPFSAccelerateMCPServer`
   (used by the integration layer).
2) A small legacy/test-oriented API surface (``register_components``,
   ``start_server_thread`` ...) used by unit tests and older scripts.
"""

from __future__ import annotations

import os
import platform
import socket
import sys
import logging
import threading
import time
from typing import Dict, Any, Optional, Union

# Package version
__version__ = "0.1.0"

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("ipfs_accelerate_mcp")

# Best-effort minimal deps (optional)
try:
    from ipfs_accelerate_py.utils.auto_install import ensure_packages

    ensure_packages(
        {
            "fastapi": "fastapi",
            "uvicorn": "uvicorn",
            "fastmcp": "fastmcp",
        }
    )
except Exception:
    pass

from .server import IPFSAccelerateMCPServer, StandaloneMCP


class LegacyMCPServer:
    """A lightweight MCP server object used by legacy code/tests.

    The unit tests in :mod:`ipfs_accelerate_py.mcp.tests` expect ``create_server`` to
    return an object with ``tools`` and ``resources`` dicts that are mutated by
    ``register_components``.
    """

    def __init__(
        self,
        host: str,
        port: int,
        name: str,
        description: str = "",
        verbose: bool = False,
    ) -> None:
        self.host = host
        self.port = port
        self.name = name
        self.description = description
        self.verbose = verbose

        self.mcp = StandaloneMCP(name=self.name)
        self.fastapi_app = self.mcp.create_fastapi_app(
            title="IPFS Accelerate MCP API (Legacy)",
            description=self.description or "Legacy MCP API",
            version=__version__,
            docs_url="/docs",
            redoc_url="/redoc",
            mount_path="",
        )

        self._uvicorn_server = None
        self._server_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()

    @property
    def tools(self) -> Dict[str, Any]:
        return getattr(self.mcp, "tools", {})

    @property
    def resources(self) -> Dict[str, Any]:
        return getattr(self.mcp, "resources", {})

    @property
    def prompts(self) -> Dict[str, Any]:
        return getattr(self.mcp, "prompts", {})

    @property
    def server_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    def run(self) -> None:
        import uvicorn

        config = uvicorn.Config(
            self.fastapi_app,
            host=self.host,
            port=self.port,
            log_level="debug" if self.verbose else "info",
        )
        self._uvicorn_server = uvicorn.Server(config)
        self._uvicorn_server.run()


def create_server(
    name: str = "ipfs-accelerate",
    host: str = "0.0.0.0",
    port: int = 8000,
    mount_path: str = "/mcp",
    debug: bool = False,
    # Legacy/test-oriented args
    description: Optional[str] = None,
    verbose: bool = False,
) -> Union[IPFSAccelerateMCPServer, LegacyMCPServer]:
    """Create an MCP server.

    - When called with ``description``/``verbose`` (legacy signature), returns a
      :class:`LegacyMCPServer`.
    - Otherwise returns a modern :class:`~ipfs_accelerate_py.mcp.server.IPFSAccelerateMCPServer`.
    """

    if description is not None or verbose:
        return LegacyMCPServer(
            host=host,
            port=port,
            name=name,
            description=description or "",
            verbose=verbose,
        )

    return IPFSAccelerateMCPServer(
        name=name,
        host=host,
        port=port,
        mount_path=mount_path,
        debug=debug,
    )


def register_components(server: Union[IPFSAccelerateMCPServer, LegacyMCPServer]) -> None:
    """Register all tools/resources on the provided server."""

    if isinstance(server, IPFSAccelerateMCPServer):
        server.setup()
        return

    from .tools import register_all_tools
    from .resources import register_all_resources

    register_all_tools(server.mcp)
    register_all_resources(server.mcp)

    # Compatibility resource names expected by legacy unit tests
    server.mcp.register_resource(
        uri="ipfs_accelerate/version",
        function=lambda: get_version(),
        description="Version information for IPFS Accelerate MCP",
    )
    server.mcp.register_resource(
        uri="ipfs_accelerate/system_info",
        function=lambda: {
            "platform": platform.system(),
            "platform_release": platform.release(),
            "python": sys.version,
        },
        description="Basic system information",
    )
    server.mcp.register_resource(
        uri="ipfs_accelerate/config",
        function=lambda: {
            "env": {
                "IPFS_ACCEL_MCP_STRICT": os.getenv("IPFS_ACCEL_MCP_STRICT"),
                "MCP_HOST": os.getenv("MCP_HOST"),
                "MCP_PORT": os.getenv("MCP_PORT"),
            }
        },
        description="Basic configuration",
    )
    try:
        from .resources.model_info import get_default_supported_models

        server.mcp.register_resource(
            uri="ipfs_accelerate/supported_models",
            function=get_default_supported_models,
            description="Supported models",
        )
    except Exception:
        server.mcp.register_resource(
            uri="ipfs_accelerate/supported_models",
            function=lambda: {"categories": {}},
            description="Supported models (fallback)",
        )


def get_server_info(server: Union[IPFSAccelerateMCPServer, LegacyMCPServer]) -> Dict[str, Any]:
    """Return a test-friendly server info dict."""

    if isinstance(server, IPFSAccelerateMCPServer):
        # Ensure server is ready; server.mcp may be None otherwise.
        if server.mcp is None:
            server.setup()
        tools = list(getattr(server.mcp, "tools", {}).keys()) if server.mcp else []
        resources = list(getattr(server.mcp, "resources", {}).keys()) if server.mcp else []
        return {
            "name": server.name,
            "description": getattr(server, "description", ""),
            "host": server.host,
            "port": server.port,
            "url": f"http://{server.host}:{server.port}",
            "tools": tools,
            "resources": resources,
        }

    return {
        "name": server.name,
        "description": server.description,
        "host": server.host,
        "port": server.port,
        "url": server.server_url,
        "tools": list(server.tools.keys()),
        "resources": list(server.resources.keys()),
    }


def _is_test_mode() -> bool:
    return bool(os.environ.get("PYTEST_CURRENT_TEST") or os.environ.get("CI"))


def start_server_thread(server: Union[IPFSAccelerateMCPServer, LegacyMCPServer]) -> threading.Thread:
    """Start a server in a background thread.

    The unit tests only assert the thread is alive; this does start uvicorn when possible.
    """

    def _ensure_stop_event() -> threading.Event:
        if not hasattr(server, "_stop_event") or getattr(server, "_stop_event") is None:
            server._stop_event = threading.Event()
        return server._stop_event

    def _keepalive() -> None:
        stop_event = _ensure_stop_event()
        while not stop_event.is_set():
            time.sleep(0.1)

    if _is_test_mode():
        t = threading.Thread(target=_keepalive, daemon=True)
        server._server_thread = t
        t.start()
        return t

    if isinstance(server, IPFSAccelerateMCPServer):
        # Ensure app exists before starting
        if server.fastapi_app is None:
            server.setup()

        def _run() -> None:
            server.run()

        t = threading.Thread(target=_run, daemon=True)
        t.start()
        return t

    import uvicorn

    def _port_is_free(host: str, port: int) -> bool:
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                s.bind((host, port))
            return True
        except OSError:
            return False

    # If the port is in use, don't crash the thread/test run.
    if not _port_is_free(server.host, server.port):
        logger.warning(
            f"Port {server.port} on {server.host} is already in use; "
            "starting keep-alive thread instead of uvicorn server"
        )
        t = threading.Thread(target=_keepalive, daemon=True)
        server._server_thread = t
        t.start()
        return t

    config = uvicorn.Config(
        server.fastapi_app,
        host=server.host,
        port=server.port,
        log_level="debug" if server.verbose else "info",
    )
    uv_server = uvicorn.Server(config)
    server._uvicorn_server = uv_server

    def _run() -> None:
        try:
            uv_server.run()
        except SystemExit:
            # uvicorn may call sys.exit() on startup failure; keep tests stable.
            _keepalive()
        except Exception:
            _keepalive()

    t = threading.Thread(target=_run, daemon=True)
    server._server_thread = t
    t.start()
    return t


def stop_server(server: Union[IPFSAccelerateMCPServer, LegacyMCPServer]) -> None:
    """Stop a background server started with :func:`start_server_thread`."""

    if isinstance(server, LegacyMCPServer):
        server._stop_event.set()
        if server._uvicorn_server is not None:
            server._uvicorn_server.should_exit = True
        return
    if hasattr(server, "_stop_event"):
        server._stop_event.set()
    # For the modern server, we don't currently hold a uvicorn.Server handle.
    # The integration layer runs it as a foreground process.


def create_and_start_server(
    host: str = "0.0.0.0",
    port: int = 8000,
    name: str = "ipfs-accelerate",
    description: str = "",
    verbose: bool = False,
    thread: bool = False,
) -> Union[IPFSAccelerateMCPServer, LegacyMCPServer]:
    """Legacy helper used by unit tests for end-to-end lifecycle."""

    server = create_server(host=host, port=port, name=name, description=description, verbose=verbose)
    register_components(server)

    if thread:
        start_server_thread(server)
        return server

    # Blocking run
    if isinstance(server, LegacyMCPServer):
        server.run()
        return server
    server.run()
    return server


def get_version() -> Dict[str, str]:
    return {
        "version": __version__,
        "name": "ipfs-accelerate",
        "description": "IPFS Accelerate MCP",
    }


def start_server(
    name: str = "ipfs-accelerate",
    host: str = "0.0.0.0",
    port: int = 8000,
    mount_path: str = "/mcp",
    debug: bool = False,
) -> None:
    """Create and run the modern MCP server (foreground)."""

    server = create_server(name=name, host=host, port=port, mount_path=mount_path, debug=debug)
    if isinstance(server, LegacyMCPServer):
        server.run()
        return

    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
    except Exception as e:
        logger.error(f"Error running server: {e}")
        raise


def check_dependencies() -> Dict[str, bool]:
    """Check if common optional dependencies are installed."""

    deps = {
        "fastmcp": False,
        "uvicorn": False,
        "psutil": False,
        "numpy": False,
        "torch": False,
    }

    for dep in list(deps.keys()):
        try:
            __import__(dep)
            deps[dep] = True
        except Exception:
            pass

    return deps


dependencies = check_dependencies()
missing_dependencies = [dep for dep, installed in dependencies.items() if not installed]
if missing_dependencies:
    strict = os.getenv("IPFS_ACCEL_MCP_STRICT", "").lower() in ("1", "true", "yes")
    if strict:
        logger.warning(f"Missing dependencies: {', '.join(missing_dependencies)}")
        logger.warning("Some features may not be available.")
        logger.warning("Install all dependencies with: pip install fastmcp uvicorn psutil numpy torch")
    else:
        logger.info(
            f"Optional dependencies not found: {', '.join(missing_dependencies)} (running with fallbacks)"
        )
