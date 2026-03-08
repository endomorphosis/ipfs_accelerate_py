"""Centralized logging helpers for unified MCP server surfaces."""

from __future__ import annotations

import logging
import re
from pathlib import Path

_MCP_LOGGER_NAME = "ipfs_accelerate.mcp_server"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_LOG_DIR = Path.home() / ".ipfs_accelerate"
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "mcp_server.log"
_OPTIONAL_FALLBACK_WARNING_PATTERNS = (
    re.compile(r"^Source .+ import unavailable, using fallback .+"),
    re.compile(r"^Source .+ imports unavailable, using fallback .+"),
    re.compile(r"^Source .+ not available, using fallback .+"),
)


class _ExpectedOptionalFallbackFilter(logging.Filter):
    """Filter expected optional dependency fallback warnings from noisy startup paths."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return True

        try:
            message = record.getMessage()
        except Exception:
            return True

        return not any(pattern.match(message) for pattern in _OPTIONAL_FALLBACK_WARNING_PATTERNS)


_OPTIONAL_FALLBACK_FILTER = _ExpectedOptionalFallbackFilter()


def _install_optional_fallback_filter() -> None:
    """Attach the optional-fallback warning filter once to root logging sinks."""
    root_logger = logging.getLogger()
    if any(existing is _OPTIONAL_FALLBACK_FILTER for existing in root_logger.filters):
        pass
    else:
        root_logger.addFilter(_OPTIONAL_FALLBACK_FILTER)

    for handler in root_logger.handlers:
        if any(existing is _OPTIONAL_FALLBACK_FILTER for existing in handler.filters):
            continue
        handler.addFilter(_OPTIONAL_FALLBACK_FILTER)


def configure_root_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a file sink for MCP diagnostics."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        _install_optional_fallback_filter()
        return

    handlers: list[logging.Handler]
    try:
        _DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
        handlers = [logging.FileHandler(_DEFAULT_LOG_FILE, mode="a")]
    except Exception:
        handlers = [logging.StreamHandler()]

    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        handlers=handlers,
    )
    _install_optional_fallback_filter()


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module logger using the unified MCP logging namespace."""
    configure_root_logging()
    logger_name = name if isinstance(name, str) and name else _MCP_LOGGER_NAME
    return logging.getLogger(logger_name)


# Backward-compatible module level loggers.
logger = get_logger(_MCP_LOGGER_NAME)
mcp_logger = get_logger("ipfs_accelerate.mcp")
