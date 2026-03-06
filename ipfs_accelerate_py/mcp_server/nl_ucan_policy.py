"""Canonical compatibility facade for source NL-UCAN policy surface."""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_source_exports() -> Dict[str, Any]:
    """Load source nl-ucan-policy symbols for compatibility delegation."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server import nl_ucan_policy as source_module  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-missing fallback
        logger.warning("Source nl_ucan_policy import unavailable: %s", exc)
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
    class NLUcanPolicyUnavailable(RuntimeError):
        """Raised when NL-UCAN policy surface is unavailable in this environment."""

    def __getattr__(name: str) -> Any:  # pragma: no cover - fallback path only
        raise NLUcanPolicyUnavailable(f"nl_ucan_policy symbol unavailable: {name}")

    __all__ = ["NLUcanPolicyUnavailable"]
