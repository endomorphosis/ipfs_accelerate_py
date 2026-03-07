#!/usr/bin/env python3
"""UNI-195 workflow import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.workflow import (
    create_workflow,
    delete_workflow,
    get_workflow,
    get_workflow_templates,
    list_workflows,
    pause_workflow,
    start_workflow,
    stop_workflow,
    update_workflow,
)
from ipfs_accelerate_py.mcp_server.tools.workflow import native_workflow_tools


def test_workflow_package_exports_supported_native_functions() -> None:
    assert get_workflow_templates is native_workflow_tools.get_workflow_templates
    assert list_workflows is native_workflow_tools.list_workflows
    assert get_workflow is native_workflow_tools.get_workflow
    assert create_workflow is native_workflow_tools.create_workflow
    assert update_workflow is native_workflow_tools.update_workflow
    assert delete_workflow is native_workflow_tools.delete_workflow
    assert start_workflow is native_workflow_tools.start_workflow
    assert pause_workflow is native_workflow_tools.pause_workflow
    assert stop_workflow is native_workflow_tools.stop_workflow
