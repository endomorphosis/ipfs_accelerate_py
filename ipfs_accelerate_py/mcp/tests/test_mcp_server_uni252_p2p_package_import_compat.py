#!/usr/bin/env python3
"""Import compatibility checks for the unified p2p package."""

from ipfs_accelerate_py.mcp_server.tools.p2p import (
    p2p_taskqueue_cache_get,
    p2p_taskqueue_cache_set,
    p2p_taskqueue_call_tool,
    p2p_taskqueue_claim_next,
    p2p_taskqueue_complete_task,
    p2p_taskqueue_get_task,
    p2p_taskqueue_heartbeat,
    p2p_taskqueue_list_tasks,
    p2p_taskqueue_status,
    p2p_taskqueue_submit,
    p2p_taskqueue_submit_docker_github,
    p2p_taskqueue_submit_docker_hub,
    p2p_taskqueue_wait_task,
)
from ipfs_accelerate_py.mcp_server.tools.p2p import native_p2p_tools


def test_p2p_package_exports_native_functions() -> None:
    assert p2p_taskqueue_status is native_p2p_tools.p2p_taskqueue_status
    assert p2p_taskqueue_submit is native_p2p_tools.p2p_taskqueue_submit
    assert p2p_taskqueue_claim_next is native_p2p_tools.p2p_taskqueue_claim_next
    assert p2p_taskqueue_call_tool is native_p2p_tools.p2p_taskqueue_call_tool
    assert p2p_taskqueue_list_tasks is native_p2p_tools.p2p_taskqueue_list_tasks
    assert p2p_taskqueue_get_task is native_p2p_tools.p2p_taskqueue_get_task
    assert p2p_taskqueue_wait_task is native_p2p_tools.p2p_taskqueue_wait_task
    assert p2p_taskqueue_complete_task is native_p2p_tools.p2p_taskqueue_complete_task
    assert p2p_taskqueue_heartbeat is native_p2p_tools.p2p_taskqueue_heartbeat
    assert p2p_taskqueue_cache_get is native_p2p_tools.p2p_taskqueue_cache_get
    assert p2p_taskqueue_cache_set is native_p2p_tools.p2p_taskqueue_cache_set
    assert p2p_taskqueue_submit_docker_hub is native_p2p_tools.p2p_taskqueue_submit_docker_hub
    assert p2p_taskqueue_submit_docker_github is native_p2p_tools.p2p_taskqueue_submit_docker_github