#!/usr/bin/env python3
"""UNI-216 admin import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.admin_tools import (
    cleanup_resources,
    configure_system,
    get_system_status,
    manage_endpoints,
    manage_service,
    system_health,
    system_maintenance,
    update_configuration,
)
from ipfs_accelerate_py.mcp_server.tools.admin_tools import native_admin_tools


def test_admin_package_exports_supported_native_functions() -> None:
    assert manage_endpoints is native_admin_tools.manage_endpoints
    assert system_maintenance is native_admin_tools.system_maintenance
    assert configure_system is native_admin_tools.configure_system
    assert system_health is native_admin_tools.system_health
    assert get_system_status is native_admin_tools.get_system_status
    assert manage_service is native_admin_tools.manage_service
    assert update_configuration is native_admin_tools.update_configuration
    assert cleanup_resources is native_admin_tools.cleanup_resources