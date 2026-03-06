"""Canonical compatibility facade for source enterprise API surface."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_source_exports() -> Dict[str, Any]:
    """Load source enterprise-api symbols for compatibility delegation."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server import enterprise_api as source_module  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-missing fallback
        logger.warning("Source enterprise_api import unavailable: %s", exc)
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
    class EnterpriseAPIUnavailable(RuntimeError):
        """Raised when enterprise API surface is unavailable in this environment."""

    def __getattr__(name: str) -> Any:  # pragma: no cover - fallback path only
        raise EnterpriseAPIUnavailable(f"enterprise_api symbol unavailable: {name}")

    __all__ = ["EnterpriseAPIUnavailable"]
