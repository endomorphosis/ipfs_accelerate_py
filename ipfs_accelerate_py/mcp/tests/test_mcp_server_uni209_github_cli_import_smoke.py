from __future__ import annotations


def test_github_cli_workflow_manager_alias_import_smoke() -> None:
    from ipfs_accelerate_py.github_cli import WorkflowManager, WorkflowQueue

    assert WorkflowManager is WorkflowQueue


def test_github_tools_module_import_smoke() -> None:
    from ipfs_accelerate_py.mcp.tools.github_tools import register_github_tools

    assert callable(register_github_tools)