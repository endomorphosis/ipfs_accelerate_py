"""Canonical compatibility facade for source compliance checker utilities.

This module keeps legacy import paths stable while delegating behavior to the
source implementation when available.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

logger = logging.getLogger(__name__)


def _load_source_exports() -> Dict[str, Any]:
    """Load source compliance-checker symbols for compatibility delegation."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server import compliance_checker as source_module  # type: ignore
    except Exception as exc:  # pragma: no cover - dependency-missing fallback
        logger.warning("Source compliance_checker import unavailable: %s", exc)
        return {}

    export_names = list(getattr(source_module, "__all__", []))
    if not export_names:
        export_names = [name for name in dir(source_module) if not name.startswith("_")]

    exports: Dict[str, Any] = {}
    for name in export_names:
        if hasattr(source_module, name):
            exports[name] = getattr(source_module, name)
    return exports


_EXPORTS = _load_source_exports()

if _EXPORTS:
    globals().update(_EXPORTS)
    __all__ = sorted(_EXPORTS.keys())
else:
    # Minimal fallback surface retained to keep imports deterministic when the
    # source package is unavailable.
    _COMPLIANCE_RULE_VERSION = "1"

    class ComplianceChecker:  # pragma: no cover - exercised only in fallback environments
        """Fallback stub when source compliance checker is unavailable."""

        def __init__(self, *_args: Any, **_kwargs: Any) -> None:
            self._reason = "source compliance_checker unavailable"

        def check(self, _intent: Any) -> Dict[str, Any]:
            return {
                "summary": "warning",
                "results": [],
                "all_violations": [],
                "reason": self._reason,
            }

    def make_default_checker() -> ComplianceChecker:  # type: ignore[override]
        """Return fallback checker instance when source module is unavailable."""
        return ComplianceChecker()

    def make_default_compliance_checker() -> ComplianceChecker:  # type: ignore[override]
        """Backward-compatible alias for fallback default checker."""
        return make_default_checker()

    __all__ = [
        "_COMPLIANCE_RULE_VERSION",
        "ComplianceChecker",
        "make_default_checker",
        "make_default_compliance_checker",
    ]
