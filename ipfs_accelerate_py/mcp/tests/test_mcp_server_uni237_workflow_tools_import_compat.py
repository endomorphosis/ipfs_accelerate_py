#!/usr/bin/env python3
"""UNI-237 workflow-tools package import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.workflow_tools import (
    add_p2p_peer,
    batch_process_datasets,
    calculate_peer_distance,
    enhanced_batch_processing,
    enhanced_data_pipeline,
    enhanced_workflow_management,
    execute_workflow,
    get_assigned_workflows,
    get_next_p2p_workflow,
    get_p2p_scheduler_status,
    get_workflow_status,
    get_workflow_tags,
    initialize_p2p_scheduler,
    merge_merkle_clock,
    remove_p2p_peer,
    schedule_p2p_workflow,
    schedule_workflow,
)
from ipfs_accelerate_py.mcp_server.tools.workflow_tools.native_workflow_tools_category import (
    add_p2p_peer as native_add_p2p_peer,
    batch_process_datasets as native_batch_process_datasets,
    calculate_peer_distance as native_calculate_peer_distance,
    enhanced_batch_processing as native_enhanced_batch_processing,
    enhanced_data_pipeline as native_enhanced_data_pipeline,
    enhanced_workflow_management as native_enhanced_workflow_management,
    execute_workflow as native_execute_workflow,
    get_assigned_workflows as native_get_assigned_workflows,
    get_next_p2p_workflow as native_get_next_p2p_workflow,
    get_p2p_scheduler_status as native_get_p2p_scheduler_status,
    get_workflow_status as native_get_workflow_status,
    get_workflow_tags as native_get_workflow_tags,
    initialize_p2p_scheduler as native_initialize_p2p_scheduler,
    merge_merkle_clock as native_merge_merkle_clock,
    remove_p2p_peer as native_remove_p2p_peer,
    schedule_p2p_workflow as native_schedule_p2p_workflow,
    schedule_workflow as native_schedule_workflow,
)


def test_workflow_tools_package_exports_source_compatible_functions() -> None:
    """Package exports should remain source-compatible aliases to native wrappers."""
    assert execute_workflow is native_execute_workflow
    assert batch_process_datasets is native_batch_process_datasets
    assert schedule_workflow is native_schedule_workflow
    assert get_workflow_status is native_get_workflow_status
    assert enhanced_workflow_management is native_enhanced_workflow_management
    assert enhanced_batch_processing is native_enhanced_batch_processing
    assert enhanced_data_pipeline is native_enhanced_data_pipeline
    assert initialize_p2p_scheduler is native_initialize_p2p_scheduler
    assert schedule_p2p_workflow is native_schedule_p2p_workflow
    assert get_next_p2p_workflow is native_get_next_p2p_workflow
    assert add_p2p_peer is native_add_p2p_peer
    assert remove_p2p_peer is native_remove_p2p_peer
    assert get_p2p_scheduler_status is native_get_p2p_scheduler_status
    assert calculate_peer_distance is native_calculate_peer_distance
    assert get_workflow_tags is native_get_workflow_tags
    assert merge_merkle_clock is native_merge_merkle_clock
    assert get_assigned_workflows is native_get_assigned_workflows