#!/usr/bin/env python3
"""UNI-200 software engineering import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.software_engineering_tools import (
    analyze_github_actions,
    analyze_pod_health,
    analyze_service_health,
    coordinate_auto_healing,
    detect_error_patterns,
    monitor_healing_effectiveness,
    parse_kubernetes_logs,
    parse_systemd_logs,
    parse_workflow_logs,
    scrape_repository,
    search_repositories,
    suggest_fixes,
)
from ipfs_accelerate_py.mcp_server.tools.software_engineering_tools import native_software_engineering_tools


def test_software_engineering_package_exports_supported_native_functions() -> None:
    assert scrape_repository is native_software_engineering_tools.scrape_repository
    assert search_repositories is native_software_engineering_tools.search_repositories
    assert analyze_github_actions is native_software_engineering_tools.analyze_github_actions
    assert parse_workflow_logs is native_software_engineering_tools.parse_workflow_logs
    assert parse_systemd_logs is native_software_engineering_tools.parse_systemd_logs
    assert analyze_service_health is native_software_engineering_tools.analyze_service_health
    assert parse_kubernetes_logs is native_software_engineering_tools.parse_kubernetes_logs
    assert analyze_pod_health is native_software_engineering_tools.analyze_pod_health
    assert detect_error_patterns is native_software_engineering_tools.detect_error_patterns
    assert suggest_fixes is native_software_engineering_tools.suggest_fixes
    assert coordinate_auto_healing is native_software_engineering_tools.coordinate_auto_healing
    assert monitor_healing_effectiveness is native_software_engineering_tools.monitor_healing_effectiveness
