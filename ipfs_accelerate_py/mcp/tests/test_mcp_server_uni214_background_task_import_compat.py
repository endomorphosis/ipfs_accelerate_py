#!/usr/bin/env python3
"""UNI-214 background task import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.background_task_tools import (
    check_task_status,
    get_task_status,
    manage_background_tasks,
    manage_task_queue,
)
from ipfs_accelerate_py.mcp_server.tools.background_task_tools import native_background_task_tools


def test_background_task_package_exports_supported_native_functions() -> None:
    assert check_task_status is native_background_task_tools.check_task_status
    assert manage_background_tasks is native_background_task_tools.manage_background_tasks
    assert manage_task_queue is native_background_task_tools.manage_task_queue
    assert get_task_status is native_background_task_tools.get_task_status