#!/usr/bin/env python3
"""Import compatibility checks for the unified lizardperson_argparse_programs package."""

from ipfs_accelerate_py.mcp_server.tools.lizardperson_argparse_programs import (
    municipal_bluebook_validator_info,
    municipal_bluebook_validator_invoke,
)
from ipfs_accelerate_py.mcp_server.tools.lizardperson_argparse_programs.native_lizardperson_argparse_programs import (
    municipal_bluebook_validator_info as native_municipal_bluebook_validator_info,
    municipal_bluebook_validator_invoke as native_municipal_bluebook_validator_invoke,
)


def test_lizardperson_argparse_programs_package_exports_native_functions() -> None:
    assert municipal_bluebook_validator_info is native_municipal_bluebook_validator_info
    assert municipal_bluebook_validator_invoke is native_municipal_bluebook_validator_invoke