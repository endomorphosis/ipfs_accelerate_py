#!/usr/bin/env python3
"""UNI-207 p2p import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.p2p_tools import (
    p2p_cache_delete,
    p2p_cache_get,
    p2p_cache_has,
    p2p_cache_set,
    p2p_remote_cache_delete,
    p2p_remote_cache_get,
    p2p_remote_cache_has,
    p2p_remote_cache_set,
    p2p_remote_call_tool,
    p2p_remote_status,
    p2p_remote_submit_task,
    p2p_service_status,
    p2p_task_delete,
    p2p_task_get,
    p2p_task_submit,
)
from ipfs_accelerate_py.mcp_server.tools.p2p_tools import native_p2p_tools


def test_p2p_package_exports_supported_native_functions() -> None:
    assert p2p_service_status is native_p2p_tools.p2p_service_status
    assert p2p_cache_get is native_p2p_tools.p2p_cache_get
    assert p2p_cache_has is native_p2p_tools.p2p_cache_has
    assert p2p_cache_set is native_p2p_tools.p2p_cache_set
    assert p2p_cache_delete is native_p2p_tools.p2p_cache_delete
    assert p2p_task_submit is native_p2p_tools.p2p_task_submit
    assert p2p_task_get is native_p2p_tools.p2p_task_get
    assert p2p_task_delete is native_p2p_tools.p2p_task_delete
    assert p2p_remote_status is native_p2p_tools.p2p_remote_status
    assert p2p_remote_call_tool is native_p2p_tools.p2p_remote_call_tool
    assert p2p_remote_cache_get is native_p2p_tools.p2p_remote_cache_get
    assert p2p_remote_cache_set is native_p2p_tools.p2p_remote_cache_set
    assert p2p_remote_cache_has is native_p2p_tools.p2p_remote_cache_has
    assert p2p_remote_cache_delete is native_p2p_tools.p2p_remote_cache_delete
    assert p2p_remote_submit_task is native_p2p_tools.p2p_remote_submit_task
