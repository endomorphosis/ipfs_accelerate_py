#!/usr/bin/env python3
"""UNI-236 P2P workflow import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.p2p_workflow_tools import (
    add_p2p_peer,
    calculate_peer_distance,
    get_assigned_workflows,
    get_next_p2p_workflow,
    get_p2p_scheduler_status,
    get_workflow_tags,
    initialize_p2p_scheduler,
    merge_merkle_clock,
    remove_p2p_peer,
    schedule_p2p_workflow,
)
from ipfs_accelerate_py.mcp_server.tools.p2p_workflow_tools import native_p2p_workflow_tools
from ipfs_accelerate_py.mcp_server.tools.workflow_tools import native_workflow_tools_category


def test_p2p_workflow_package_exports_source_compatible_functions() -> None:
    assert initialize_p2p_scheduler is native_p2p_workflow_tools.initialize_p2p_scheduler
    assert schedule_p2p_workflow is native_p2p_workflow_tools.schedule_p2p_workflow
    assert get_next_p2p_workflow is native_p2p_workflow_tools.get_next_p2p_workflow
    assert get_p2p_scheduler_status is native_p2p_workflow_tools.get_p2p_scheduler_status
    assert get_assigned_workflows is native_p2p_workflow_tools.get_assigned_workflows
    assert get_workflow_tags is native_workflow_tools_category.get_workflow_tags
    assert add_p2p_peer is native_workflow_tools_category.add_p2p_peer
    assert remove_p2p_peer is native_workflow_tools_category.remove_p2p_peer
    assert calculate_peer_distance is native_workflow_tools_category.calculate_peer_distance
    assert merge_merkle_clock is native_workflow_tools_category.merge_merkle_clock