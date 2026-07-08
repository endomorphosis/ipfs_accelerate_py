"""Regression tests for unified MCP logger noise filtering."""

from __future__ import annotations

import importlib
import logging
from io import StringIO

logger_mod = importlib.import_module("ipfs_accelerate_py.mcp_server.logger")


def _make_record(level: int, message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="ipfs_accelerate_py.mcp_server.tools.auth_tools.native_auth_tools",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_optional_fallback_warning_filter_suppresses_expected_source_fallback_messages() -> None:
    record = _make_record(
        logging.WARNING,
        "Source auth_tools import unavailable, using fallback auth functions",
    )

    assert logger_mod._OPTIONAL_FALLBACK_FILTER.filter(record) is False


def test_optional_fallback_warning_filter_keeps_cutover_and_runtime_warnings() -> None:
    runtime_record = _make_record(
        logging.WARNING,
        "Unified cutover dry-run validation failed, continuing on legacy path: boom",
    )

    for phase_label in ("D1 warn-only", "D2 opt-in only"):
        cutover_record = _make_record(
            logging.WARNING,
            f"Legacy MCP facade runtime path is deprecated ({phase_label}); reason=force_legacy_rollback.",
        )
        assert logger_mod._OPTIONAL_FALLBACK_FILTER.filter(cutover_record) is True

    assert logger_mod._OPTIONAL_FALLBACK_FILTER.filter(runtime_record) is True


def test_optional_fallback_filter_does_not_touch_info_messages() -> None:
    info_record = _make_record(
        logging.INFO,
        "Source auth_tools import unavailable, using fallback auth functions",
    )

    assert logger_mod._OPTIONAL_FALLBACK_FILTER.filter(info_record) is True


def test_configure_root_logging_filters_expected_fallback_warning_from_handler() -> None:
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_filters = list(root.filters)
    original_level = root.level
    stream = StringIO()
    handler = logging.StreamHandler(stream)

    try:
        root.handlers = [handler]
        root.filters = []
        root.setLevel(logging.INFO)

        logger_mod.configure_root_logging()

        child_logger = logging.getLogger(
            "ipfs_accelerate_py.mcp_server.tools.auth_tools.native_auth_tools"
        )
        child_logger.warning("Source auth_tools import unavailable, using fallback auth functions")
        child_logger.warning("Legacy MCP facade runtime path is deprecated (D1 warn-only); reason=force_legacy_rollback.")
        child_logger.warning("Legacy MCP facade runtime path is deprecated (D2 opt-in only); reason=force_legacy_rollback.")

        output = stream.getvalue()
        assert "Source auth_tools import unavailable" not in output
        assert "Legacy MCP facade runtime path is deprecated" in output
    finally:
        root.handlers = original_handlers
        root.filters = original_filters
        root.setLevel(original_level)