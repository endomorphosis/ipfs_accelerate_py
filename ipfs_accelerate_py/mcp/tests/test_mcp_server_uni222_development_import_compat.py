#!/usr/bin/env python3
"""UNI-222 development import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.development_tools import (
    codebase_search,
    documentation_generator,
    lint_python_codebase,
    run_comprehensive_tests,
    test_generator,
    vscode_cli_execute,
    vscode_cli_install,
    vscode_cli_install_extension,
    vscode_cli_list_extensions,
    vscode_cli_status,
    vscode_cli_tunnel_install_service,
    vscode_cli_tunnel_login,
    vscode_cli_uninstall_extension,
)
from ipfs_accelerate_py.mcp_server.tools.development_tools import native_development_tools


def test_development_package_exports_supported_native_functions() -> None:
    assert test_generator is native_development_tools.test_generator
    assert documentation_generator is native_development_tools.documentation_generator
    assert lint_python_codebase is native_development_tools.lint_python_codebase
    assert run_comprehensive_tests is native_development_tools.run_comprehensive_tests
    assert codebase_search is native_development_tools.codebase_search
    assert vscode_cli_status is native_development_tools.vscode_cli_status
    assert vscode_cli_install is native_development_tools.vscode_cli_install
    assert vscode_cli_execute is native_development_tools.vscode_cli_execute
    assert vscode_cli_list_extensions is native_development_tools.vscode_cli_list_extensions
    assert vscode_cli_install_extension is native_development_tools.vscode_cli_install_extension
    assert vscode_cli_uninstall_extension is native_development_tools.vscode_cli_uninstall_extension
    assert vscode_cli_tunnel_login is native_development_tools.vscode_cli_tunnel_login
    assert vscode_cli_tunnel_install_service is native_development_tools.vscode_cli_tunnel_install_service