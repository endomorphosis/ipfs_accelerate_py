#!/usr/bin/env python3
"""UNI-209 audit import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.audit_tools import (
    audit_tools,
    generate_audit_report,
    record_audit_event,
)
from ipfs_accelerate_py.mcp_server.tools.audit_tools import native_audit_tools


def test_audit_package_exports_supported_native_functions() -> None:
    assert record_audit_event is native_audit_tools.record_audit_event
    assert generate_audit_report is native_audit_tools.generate_audit_report
    assert audit_tools is native_audit_tools.audit_tools