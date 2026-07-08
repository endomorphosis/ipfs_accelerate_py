#!/usr/bin/env python3
"""UNI-194 investigation import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.investigation_tools import (
    analyze_deontological_conflicts,
    analyze_entities,
    analyze_entity_timeline,
    detect_patterns,
    explore_entity,
    extract_geographic_entities,
    ingest_document_collection,
    ingest_news_article,
    ingest_news_feed,
    ingest_website,
    map_relationships,
    map_spatiotemporal_events,
    query_deontic_conflicts,
    query_deontic_statements,
    query_geographic_context,
    track_provenance,
)
from ipfs_accelerate_py.mcp_server.tools.investigation_tools import native_investigation_tools


def test_investigation_package_exports_source_compatible_functions() -> None:
    assert analyze_entities is native_investigation_tools.analyze_entities
    assert explore_entity is native_investigation_tools.explore_entity
    assert map_relationships is native_investigation_tools.map_relationships
    assert analyze_entity_timeline is native_investigation_tools.analyze_entity_timeline
    assert detect_patterns is native_investigation_tools.detect_patterns
    assert track_provenance is native_investigation_tools.track_provenance
    assert ingest_news_article is native_investigation_tools.ingest_news_article
    assert ingest_news_feed is native_investigation_tools.ingest_news_feed
    assert ingest_website is native_investigation_tools.ingest_website
    assert ingest_document_collection is native_investigation_tools.ingest_document_collection
    assert analyze_deontological_conflicts is native_investigation_tools.analyze_deontological_conflicts
    assert query_deontic_statements is native_investigation_tools.query_deontic_statements
    assert query_deontic_conflicts is native_investigation_tools.query_deontic_conflicts
    assert extract_geographic_entities is native_investigation_tools.extract_geographic_entities
    assert map_spatiotemporal_events is native_investigation_tools.map_spatiotemporal_events
    assert query_geographic_context is native_investigation_tools.query_geographic_context
