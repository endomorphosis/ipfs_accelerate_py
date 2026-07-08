"""Regression tests for ipfs_datasets_py optional dependency warning filtering."""

from __future__ import annotations

import importlib
import logging
from io import StringIO


logger_mod = importlib.import_module("ipfs_datasets_py.ipfs_datasets_py.mcp_server.logger")


def _make_record(level: int, message: str) -> logging.LogRecord:
    return logging.LogRecord(
        name="ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_prove_tool",
        level=level,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=(),
        exc_info=None,
    )


def test_ipfs_datasets_filter_suppresses_expected_optional_dependency_warning() -> None:
    record = _make_record(logging.WARNING, "LogicProcessor not available: No module named 'ipfs_datasets_py.core_operations'")

    assert logger_mod._OPTIONAL_WARNING_FILTER.filter(record) is False


def test_ipfs_datasets_filter_keeps_errors_and_cutover_warnings() -> None:
    error_record = _make_record(logging.ERROR, "Neither common/ nor complaint_analysis available")

    assert logger_mod._OPTIONAL_WARNING_FILTER.filter(error_record) is True
    for phase_label in ("D1 warn-only", "D2 opt-in only"):
        cutover_record = _make_record(
            logging.WARNING,
            f"Legacy MCP facade runtime path is deprecated ({phase_label}); reason=force_legacy_rollback.",
        )
        assert logger_mod._OPTIONAL_WARNING_FILTER.filter(cutover_record) is True


def test_ipfs_datasets_filter_installs_on_existing_handler() -> None:
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

        logger_mod._install_optional_warning_filter()

        child_logger = logging.getLogger(
            "ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.logic_tools.tdfol_prove_tool"
        )
        child_logger.warning("LogicProcessor not available: No module named 'ipfs_datasets_py.core_operations'")
        child_logger.warning("Legacy MCP facade runtime path is deprecated (D1 warn-only); reason=force_legacy_rollback.")
        child_logger.warning("Legacy MCP facade runtime path is deprecated (D2 opt-in only); reason=force_legacy_rollback.")

        output = stream.getvalue()
        assert "LogicProcessor not available" not in output
        assert "Legacy MCP facade runtime path is deprecated" in output
    finally:
        root.handlers = original_handlers
        root.filters = original_filters
        root.setLevel(original_level)