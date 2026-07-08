#!/usr/bin/env python3
"""UNI-202 geospatial import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.geospatial_tools import (
    analyze_geospatial_corpus,
    extract_geographic_entities,
    map_spatiotemporal_events,
    query_geographic_context,
)
from ipfs_accelerate_py.mcp_server.tools.geospatial_tools import native_geospatial_tools


def test_geospatial_package_exports_supported_native_functions() -> None:
    assert extract_geographic_entities is native_geospatial_tools.extract_geographic_entities
    assert map_spatiotemporal_events is native_geospatial_tools.map_spatiotemporal_events
    assert query_geographic_context is native_geospatial_tools.query_geographic_context
    assert analyze_geospatial_corpus is native_geospatial_tools.analyze_geospatial_corpus
