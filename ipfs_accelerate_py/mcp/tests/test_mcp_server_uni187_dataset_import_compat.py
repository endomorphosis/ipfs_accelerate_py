#!/usr/bin/env python3
"""UNI-187 dataset import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.dataset_tools import (
    convert_dataset_format,
    legal_text_to_deontic,
    load_dataset,
    process_dataset,
    save_dataset,
    text_to_fol,
)
from ipfs_accelerate_py.mcp_server.tools.dataset_tools import native_dataset_tools


def test_dataset_package_exports_source_compatible_functions() -> None:
    assert load_dataset is native_dataset_tools.load_dataset
    assert save_dataset is native_dataset_tools.save_dataset
    assert process_dataset is native_dataset_tools.process_dataset
    assert convert_dataset_format is native_dataset_tools.convert_dataset_format
    assert text_to_fol is native_dataset_tools.text_to_fol
    assert legal_text_to_deontic is native_dataset_tools.legal_text_to_deontic