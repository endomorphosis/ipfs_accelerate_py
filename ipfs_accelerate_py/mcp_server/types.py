"""Canonical type definitions for ipfs_accelerate_py.mcp_server.

Provides ``IPFSAccelerateContext`` and other shared type definitions used
across the canonical MCP server components.

Drop-in migration::

    # Old (deprecated):
    from ipfs_accelerate_py.mcp.types import IPFSAccelerateContext

    # New (canonical):
    from ipfs_accelerate_py.mcp_server.types import IPFSAccelerateContext
"""

from __future__ import annotations

from typing import Any, Dict, Optional


class IPFSAccelerateContext:
    """Context object for IPFS Accelerate MCP.

    Stores shared state and resources available to all tools throughout the
    lifespan of the server.
    """

    def __init__(self) -> None:
        self.ipfs_client: Any = None
        self.hardware_info: Optional[Dict[str, Any]] = None
        self.accelerated_models: Dict[str, Any] = {}
        self.model_metadata: Dict[str, Any] = {}

    def set_ipfs_client(self, client: Any) -> None:
        """Set the IPFS client."""
        self.ipfs_client = client

    def set_hardware_info(self, hardware_info: Dict[str, Any]) -> None:
        """Set hardware information."""
        self.hardware_info = hardware_info

    def register_model(self, model_id: str, model_info: Any) -> None:
        """Register an accelerated model."""
        self.accelerated_models[model_id] = model_info

    def get_model(self, model_id: str) -> Any:
        """Get an accelerated model by ID."""
        return self.accelerated_models.get(model_id)


__all__ = ["IPFSAccelerateContext"]
