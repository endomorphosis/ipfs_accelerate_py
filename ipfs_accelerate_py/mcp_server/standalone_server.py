#!/usr/bin/env python3
"""Canonical standalone MCP server entrypoints.

This module provides stable standalone startup functions under
``ipfs_accelerate_py.mcp_server`` while delegating to proven runtime paths.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys

from .fastapi_config import UnifiedFastAPIConfig
from .fastapi_service import run_fastapi_server as run_canonical_fastapi_server
from .server import create_server


logger = logging.getLogger(__name__)


def run_server(
    host: str = "localhost",
    port: int = 8080,
    name: str = "ipfs-accelerate",
    description: str = "IPFS Accelerate MCP Server",
    verbose: bool = False,
) -> None:
    """Run standalone MCP server using the canonical server builder."""
    logger.info("Starting standalone MCP server on %s:%s", host, port)

    try:
        mcp = create_server(
            host=host,
            port=port,
            name=name,
            description=description,
            debug=bool(verbose),
            mount_path="/mcp",
        )

        def signal_handler(_sig, _frame):
            logger.info("Received shutdown signal, stopping server...")
            try:
                mcp.stop()
                logger.info("Server stopped")
            except Exception:
                pass
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        logger.info("Server listening on http://%s:%s", host, port)
        if hasattr(mcp, "run") and callable(getattr(mcp, "run")):
            mcp.run(host=host, port=port)
            return
        raise RuntimeError("MCP server instance does not expose a run() method")
    except KeyboardInterrupt:
        logger.info("Keyboard interrupt received, stopping server...")
        sys.exit(0)
    except Exception as exc:
        logger.error("Error running standalone MCP server: %s", exc)
        sys.exit(1)


def run_fastapi_server(
    host: str = "localhost",
    port: int = 8000,
    mount_path: str = "/mcp",
    name: str = "ipfs-accelerate",
    description: str = "IPFS Accelerate MCP Server",
    verbose: bool = False,
) -> None:
    """Run standalone FastAPI server from canonical mcp_server package."""
    run_canonical_fastapi_server(
        UnifiedFastAPIConfig(
            host=host,
            port=port,
            mount_path=mount_path,
            name=name,
            description=description,
            verbose=verbose,
        )
    )


def main() -> None:
    """CLI entrypoint for canonical standalone startup wrappers."""
    parser = argparse.ArgumentParser(description="IPFS Accelerate MCP Standalone Server (canonical facade)")
    parser.add_argument("--host", default="localhost", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind to")
    parser.add_argument("--name", default="ipfs-accelerate", help="Name of the MCP server")
    parser.add_argument("--description", default="IPFS Accelerate MCP Server", help="Description of the MCP server")
    parser.add_argument("--fastapi", action="store_true", help="Use FastAPI integration")
    parser.add_argument("--mount-path", default="/mcp", help="Path to mount the MCP server at (for FastAPI)")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    if args.fastapi:
        run_fastapi_server(
            host=args.host,
            port=args.port,
            mount_path=args.mount_path,
            name=args.name,
            description=args.description,
            verbose=args.verbose,
        )
        return

    run_server(
        host=args.host,
        port=args.port,
        name=args.name,
        description=args.description,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    main()
