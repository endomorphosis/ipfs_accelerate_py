"""Canonical compatibility facade for source investigation MCP client."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_source_exports() -> Dict[str, Any]:
    """Load source investigation-client symbols for compatibility delegation."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server import investigation_mcp_client as source_module  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-missing fallback
        logger.warning("Source investigation_mcp_client import unavailable: %s", exc)
        return {}

    exports: Dict[str, Any] = {}
    for name in dir(source_module):
        if not name.startswith("_") and hasattr(source_module, name):
            exports[name] = getattr(source_module, name)
    return exports


_EXPORTS = _load_source_exports()

if _EXPORTS:
    globals().update(_EXPORTS)
    __all__ = sorted(_EXPORTS.keys())
else:
    class InvestigationMCPClientError(Exception):
        """Fallback exception when source investigation client is unavailable."""

    class InvestigationMCPClient:  # pragma: no cover - fallback path only
        """Fallback stub preserving import-path compatibility."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self.base_url = ""
            self.endpoint = ""
            self.timeout = 0

    def create_investigation_mcp_client(*_args: Any, **_kwargs: Any) -> InvestigationMCPClient:
        """Return fallback investigation client when source module is unavailable."""
        return InvestigationMCPClient()

    __all__ = [
        "InvestigationMCPClientError",
        "InvestigationMCPClient",
        "create_investigation_mcp_client",
    ]
