"""Centralized logging helpers for unified MCP server surfaces."""

from __future__ import annotations

import logging
from pathlib import Path

_MCP_LOGGER_NAME = "ipfs_accelerate.mcp_server"
_LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"
_DEFAULT_LOG_DIR = Path.home() / ".ipfs_accelerate"
_DEFAULT_LOG_FILE = _DEFAULT_LOG_DIR / "mcp_server.log"


def configure_root_logging(level: int = logging.INFO) -> None:
    """Configure root logging once with a file sink for MCP diagnostics."""
    root_logger = logging.getLogger()
    if root_logger.handlers:
        return

    _DEFAULT_LOG_DIR.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=level,
        format=_LOG_FORMAT,
        handlers=[logging.FileHandler(_DEFAULT_LOG_FILE, mode="a")],
    )


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a module logger using the unified MCP logging namespace."""
    configure_root_logging()
    logger_name = name if isinstance(name, str) and name else _MCP_LOGGER_NAME
    return logging.getLogger(logger_name)


# Backward-compatible module level loggers.
logger = get_logger(_MCP_LOGGER_NAME)
mcp_logger = get_logger("ipfs_accelerate.mcp")
