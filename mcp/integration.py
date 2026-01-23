"""FastAPI integration helpers for the MCP server.

The test suite expects `initialize_mcp_server(app, accelerate)` to register a
small HTTP surface under `/mcp`.
"""

from __future__ import annotations

import platform
from typing import Any

from fastapi import APIRouter, FastAPI

try:
    # Preferred: test helper implementation
    from common.hardware_detection import detect_hardware  # type: ignore
except Exception:  # pragma: no cover
    def detect_hardware() -> dict:
        return {"platforms": {"cpu": {"available": True}}}


def initialize_mcp_server(app: FastAPI, accelerate: Any, base_path: str = "/mcp") -> None:
    """Register minimal MCP endpoints on a FastAPI app."""

    router = APIRouter(prefix=base_path)

    @router.get("", tags=["mcp"])
    async def get_server_info() -> dict:
        return {
            "name": "IPFS Accelerate MCP",
            "description": "MCP server for IPFS Accelerate",
        }

    @router.get("/health", tags=["mcp"])
    async def health() -> dict:
        return {"status": "ok"}

    @router.get("/resources/system://info", tags=["mcp"])
    async def system_info() -> dict:
        return {
            "platform": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        }

    @router.post("/tools/detect_hardware", tags=["mcp"])
    async def tool_detect_hardware(payload: dict | None = None) -> dict:
        return detect_hardware()

    app.include_router(router)
