#!/usr/bin/env python3
"""UNI-210 session import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.session_tools import (
    cleanup_sessions,
    create_session,
    get_session_state,
    manage_session,
    manage_session_state,
)
from ipfs_accelerate_py.mcp_server.tools.session_tools import native_session_tools


def test_session_package_exports_supported_native_functions() -> None:
    assert create_session is native_session_tools.create_session
    assert manage_session_state is native_session_tools.manage_session_state
    assert cleanup_sessions is native_session_tools.cleanup_sessions
    assert manage_session is native_session_tools.manage_session
    assert get_session_state is native_session_tools.get_session_state