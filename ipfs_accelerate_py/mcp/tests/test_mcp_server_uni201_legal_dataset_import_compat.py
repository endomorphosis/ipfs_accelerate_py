#!/usr/bin/env python3
"""UNI-201 legal dataset import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.legal_dataset_tools import (
    expand_legal_query,
    get_legal_relationships,
    get_legal_synonyms,
    list_state_jurisdictions,
    scrape_state_laws,
)
from ipfs_accelerate_py.mcp_server.tools.legal_dataset_tools import native_legal_dataset_tools


def test_legal_dataset_package_exports_supported_native_functions() -> None:
    assert list_state_jurisdictions is native_legal_dataset_tools.list_state_jurisdictions
    assert scrape_state_laws is native_legal_dataset_tools.scrape_state_laws
    assert expand_legal_query is native_legal_dataset_tools.expand_legal_query
    assert get_legal_synonyms is native_legal_dataset_tools.get_legal_synonyms
    assert get_legal_relationships is native_legal_dataset_tools.get_legal_relationships
