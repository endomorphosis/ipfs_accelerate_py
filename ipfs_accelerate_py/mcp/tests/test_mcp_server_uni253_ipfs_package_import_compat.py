#!/usr/bin/env python3
"""Import compatibility checks for the unified ipfs package."""

from ipfs_accelerate_py.mcp_server.tools.ipfs import (
    ipfs_files_add_file,
    ipfs_files_cat,
    ipfs_files_get_file,
    ipfs_files_list_files,
    ipfs_files_pin_file,
    ipfs_files_unpin_file,
    ipfs_files_validate_cid,
)
from ipfs_accelerate_py.mcp_server.tools.ipfs import native_ipfs_tools


def test_ipfs_package_exports_native_functions() -> None:
    assert ipfs_files_validate_cid is native_ipfs_tools.ipfs_files_validate_cid
    assert ipfs_files_list_files is native_ipfs_tools.ipfs_files_list_files
    assert ipfs_files_add_file is native_ipfs_tools.ipfs_files_add_file
    assert ipfs_files_pin_file is native_ipfs_tools.ipfs_files_pin_file
    assert ipfs_files_unpin_file is native_ipfs_tools.ipfs_files_unpin_file
    assert ipfs_files_get_file is native_ipfs_tools.ipfs_files_get_file
    assert ipfs_files_cat is native_ipfs_tools.ipfs_files_cat