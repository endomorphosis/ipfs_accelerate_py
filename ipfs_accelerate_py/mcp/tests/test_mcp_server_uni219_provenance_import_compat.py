#!/usr/bin/env python3
"""UNI-219 provenance import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.provenance_tools import (
    generate_provenance_report,
    record_provenance,
    record_provenance_batch,
    verify_provenance_records,
)
from ipfs_accelerate_py.mcp_server.tools.provenance_tools import native_provenance_tools


def test_provenance_package_exports_supported_native_functions() -> None:
    assert record_provenance is native_provenance_tools.record_provenance
    assert record_provenance_batch is native_provenance_tools.record_provenance_batch
    assert verify_provenance_records is native_provenance_tools.verify_provenance_records
    assert generate_provenance_report is native_provenance_tools.generate_provenance_report