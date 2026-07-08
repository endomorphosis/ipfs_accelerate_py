#!/usr/bin/env python3
"""UNI-188 monitoring import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.monitoring_tools import (
    generate_monitoring_report,
    get_performance_metrics,
    health_check,
    monitor_services,
)
from ipfs_accelerate_py.mcp_server.tools.monitoring_tools import native_monitoring_tools


def test_monitoring_package_exports_source_compatible_functions() -> None:
    assert health_check is native_monitoring_tools.health_check
    assert get_performance_metrics is native_monitoring_tools.get_performance_metrics
    assert monitor_services is native_monitoring_tools.monitor_services
    assert generate_monitoring_report is native_monitoring_tools.generate_monitoring_report