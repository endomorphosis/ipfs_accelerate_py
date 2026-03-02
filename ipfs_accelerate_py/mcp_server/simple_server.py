"""Legacy simple-server compatibility facade for canonical MCP runtime."""

from __future__ import annotations

from typing import Any, Dict, Optional

from .standalone_server import main as standalone_main
from .standalone_server import run_fastapi_server, run_server


class SimpleCallResult:
    """Source-compatible simple call result wrapper."""

    def __init__(self, result: Any, error: Optional[str] = None):
        self.result = result
        self.error = error

    def to_dict(self) -> Dict[str, Any]:
        if self.error:
            return {"success": False, "error": self.error}
        return {"success": True, "result": self.result}


class SimpleIPFSDatasetsMCPServer:
    """Thin source-compatible facade over canonical standalone startup."""

    def __init__(self, server_configs: Optional[Any] = None):
        self.configs = server_configs
        self.tools: Dict[str, Any] = {}

    def register_tools(self) -> None:
        # No-op: canonical runtime handles registration internally.
        return None

    def run(self, host: Optional[str] = None, port: Optional[int] = None):
        run_server(host=host or "localhost", port=int(port or 8080))


def start_simple_server(config_path: Optional[str] = None):
    """Source-compatible startup helper that delegates to canonical standalone."""
    _ = config_path
    run_server()


def main() -> None:
    """Compatibility main entrypoint for simple server facade."""
    standalone_main()


__all__ = [
    "SimpleCallResult",
    "SimpleIPFSDatasetsMCPServer",
    "run_server",
    "run_fastapi_server",
    "start_simple_server",
    "main",
]
