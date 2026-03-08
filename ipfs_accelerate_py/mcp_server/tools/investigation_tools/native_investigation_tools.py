"""Native investigation-tools category implementations for unified mcp_server."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_SEVERITY_LEVELS = {"low", "medium", "high"}
_VALID_DEONTIC_MODALITIES = {"obligation", "permission", "prohibition"}
_VALID_CONFLICT_TYPES = {"direct", "conditional", "jurisdictional", "temporal"}
_VALID_TIME_GRANULARITIES = {"hour", "day", "week", "month"}
_VALID_TEMPORAL_RESOLUTIONS = {"hour", "day", "week", "month"}
_VALID_PATTERN_TYPES = {"behavioral", "relational", "temporal", "anomaly"}


def _load_investigation_api() -> Dict[str, Any]:
    """Resolve source investigation APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.investigation_tools import (  # type: ignore
            analyze_deontological_conflicts as _analyze_deontological_conflicts,
            analyze_entities as _analyze_entities,
            analyze_entity_timeline as _analyze_entity_timeline,
            detect_patterns as _detect_patterns,
            explore_entity as _explore_entity,
            extract_geographic_entities as _extract_geographic_entities,
            ingest_document_collection as _ingest_document_collection,
            ingest_news_article as _ingest_news_article,
            ingest_news_feed as _ingest_news_feed,
            ingest_website as _ingest_website,
            map_relationships as _map_relationships,
            map_spatiotemporal_events as _map_spatiotemporal_events,
            query_deontic_conflicts as _query_deontic_conflicts,
            query_deontic_statements as _query_deontic_statements,
            query_geographic_context as _query_geographic_context,
            track_provenance as _track_provenance,
        )

        return {
            "analyze_entities": _analyze_entities,
            "explore_entity": _explore_entity,
            "map_relationships": _map_relationships,
            "analyze_entity_timeline": _analyze_entity_timeline,
            "detect_patterns": _detect_patterns,
            "track_provenance": _track_provenance,
            "ingest_news_article": _ingest_news_article,
            "ingest_news_feed": _ingest_news_feed,
            "ingest_website": _ingest_website,
            "ingest_document_collection": _ingest_document_collection,
            "analyze_deontological_conflicts": _analyze_deontological_conflicts,
            "query_deontic_statements": _query_deontic_statements,
            "query_deontic_conflicts": _query_deontic_conflicts,
            "extract_geographic_entities": _extract_geographic_entities,
            "map_spatiotemporal_events": _map_spatiotemporal_events,
            "query_geographic_context": _query_geographic_context,
        }
    except Exception:
        logger.warning("Source investigation_tools import unavailable, using fallback investigation functions")

        async def _analyze_entities_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "analysis_type": kwargs.get("analysis_type", "comprehensive"),
                "entities": [],
                "relationships": [],
                "clusters": [],
                "statistics": {},
                "fallback": True,
            }

        async def _explore_entity_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "entity_id": kwargs.get("entity_id"),
                "entity_details": {},
                "relationships": [],
                "timeline": [],
                "sources": [],
                "fallback": True,
            }

        async def _map_relationships_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "entities": [],
                "relationships": [],
                "graph_metrics": {},
                "max_depth": kwargs.get("max_depth", 3),
                "fallback": True,
            }

        async def _analyze_entity_timeline_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "entity_id": kwargs.get("entity_id"),
                "timeline_events": [],
                "time_distribution": {},
                "event_clusters": [],
                "related_entities_timeline": {},
                "fallback": True,
            }

        async def _detect_patterns_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "patterns": [],
                "pattern_statistics": {},
                "parameters": {
                    "pattern_types": kwargs.get("pattern_types")
                    or ["behavioral", "relational", "temporal", "anomaly"],
                    "time_window": kwargs.get("time_window", "30d"),
                },
                "fallback": True,
            }

        async def _track_provenance_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "entity_id": kwargs.get("entity_id"),
                "provenance_chain": [],
                "source_documents": [],
                "citation_network": [],
                "trust_metrics": {},
                "fallback": True,
            }

        async def _ingest_news_article_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "url": kwargs.get("url"),
                "source_type": kwargs.get("source_type", "news"),
                "analysis_type": kwargs.get("analysis_type", "comprehensive"),
                "fallback": True,
            }

        async def _ingest_news_feed_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "feed_url": kwargs.get("feed_url"),
                "articles": [],
                "max_articles": kwargs.get("max_articles", 50),
                "fallback": True,
            }

        async def _ingest_website_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "base_url": kwargs.get("base_url"),
                "crawled_pages": [],
                "max_pages": kwargs.get("max_pages", 100),
                "fallback": True,
            }

        async def _ingest_document_collection_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "document_paths": kwargs.get("document_paths"),
                "collection_name": kwargs.get("collection_name", "document_collection"),
                "processed_count": 0,
                "fallback": True,
            }

        async def _analyze_deontological_conflicts_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "deontic_statements": [],
                "conflicts": [],
                "statistics": {},
                "severity_threshold": kwargs.get("severity_threshold", "medium"),
                "fallback": True,
            }

        async def _query_deontic_statements_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "query": {
                    "entity": kwargs.get("entity"),
                    "modality": kwargs.get("modality"),
                    "action_pattern": kwargs.get("action_pattern"),
                },
                "total_found": 0,
                "statements": [],
                "fallback": True,
            }

        async def _query_deontic_conflicts_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "query": {
                    "conflict_type": kwargs.get("conflict_type"),
                    "severity": kwargs.get("severity"),
                    "entity": kwargs.get("entity"),
                },
                "total_found": 0,
                "conflicts": [],
                "fallback": True,
            }

        async def _extract_geographic_entities_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "entities": [],
                "confidence_threshold": kwargs.get("confidence_threshold", 0.8),
                "fallback": True,
            }

        async def _map_spatiotemporal_events_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "events": [],
                "clusters": [],
                "temporal_resolution": kwargs.get("temporal_resolution", "day"),
                "fallback": True,
            }

        async def _query_geographic_context_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "query": kwargs.get("query"),
                "results": [],
                "radius_km": kwargs.get("radius_km", 100.0),
                "fallback": True,
            }

        return {
            "analyze_entities": _analyze_entities_fallback,
            "explore_entity": _explore_entity_fallback,
            "map_relationships": _map_relationships_fallback,
            "analyze_entity_timeline": _analyze_entity_timeline_fallback,
            "detect_patterns": _detect_patterns_fallback,
            "track_provenance": _track_provenance_fallback,
            "ingest_news_article": _ingest_news_article_fallback,
            "ingest_news_feed": _ingest_news_feed_fallback,
            "ingest_website": _ingest_website_fallback,
            "ingest_document_collection": _ingest_document_collection_fallback,
            "analyze_deontological_conflicts": _analyze_deontological_conflicts_fallback,
            "query_deontic_statements": _query_deontic_statements_fallback,
            "query_deontic_conflicts": _query_deontic_conflicts_fallback,
            "extract_geographic_entities": _extract_geographic_entities_fallback,
            "map_spatiotemporal_events": _map_spatiotemporal_events_fallback,
            "query_geographic_context": _query_geographic_context_fallback,
        }


_API = _load_investigation_api()


def _error(message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "error", "success": False, "error": message}
    payload.update(extra)
    return payload


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _normalize_success(payload: Any, **defaults: Any) -> Dict[str, Any]:
    envelope = _normalize_payload(payload)
    envelope.setdefault("status", "success")
    if envelope.get("status") == "success":
        envelope.setdefault("success", True)
    for key, value in defaults.items():
        envelope.setdefault(key, value)
    return envelope


def _validate_string_list(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        return _error(f"{field} must be null or a list of non-empty strings", **{field: value})
    return None


def _clean_optional_string(value: Optional[str], field: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if value is None:
        return None, None
    if not isinstance(value, str) or not value.strip():
        return None, _error(f"{field} must be null or a non-empty string", **{field: value})
    return value.strip(), None


def _json_argument(value: Any, field: str, expected: str) -> tuple[Optional[str], Optional[Dict[str, Any]]]:
    if value is None:
        return None, None
    if isinstance(value, str):
        if not value.strip():
            return None, _error(f"{field} must be null or a non-empty JSON string", **{field: value})
        return value, None
    if isinstance(value, (dict, list)):
        return json.dumps(value), None
    return None, _error(f"{field} must be null, a {expected}, or a JSON string", **{field: value})


async def _invoke(tool_name: str, **kwargs: Any) -> Dict[str, Any]:
    result = _API[tool_name](**kwargs)
    if hasattr(result, "__await__"):
        result = await result
    return _normalize_payload(result)


async def analyze_entities(
    corpus_data: str,
    analysis_type: str = "comprehensive",
    entity_types: Optional[List[str]] = None,
    confidence_threshold: float = 0.85,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze investigation entities with deterministic validation and envelope shape."""
    normalized_corpus = str(corpus_data or "").strip()
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    normalized_analysis_type = str(analysis_type or "").strip() or "comprehensive"
    list_error = _validate_string_list(entity_types, "entity_types")
    if list_error:
        return list_error
    try:
        normalized_threshold = float(confidence_threshold)
    except (TypeError, ValueError):
        return _error(
            "confidence_threshold must be a number between 0 and 1",
            confidence_threshold=confidence_threshold,
        )
    if normalized_threshold < 0 or normalized_threshold > 1:
        return _error(
            "confidence_threshold must be a number between 0 and 1",
            confidence_threshold=confidence_threshold,
        )
    clean_user_context, error = _clean_optional_string(user_context, "user_context")
    if error:
        return error
    try:
        payload = await _invoke(
            "analyze_entities",
            corpus_data=normalized_corpus,
            analysis_type=normalized_analysis_type,
            entity_types=[item.strip() for item in entity_types] if entity_types is not None else None,
            confidence_threshold=normalized_threshold,
            user_context=clean_user_context,
        )
        return _normalize_success(
            payload,
            analysis_type=normalized_analysis_type,
            entities=[],
            relationships=[],
            clusters=[],
            statistics={},
        )
    except Exception as exc:
        return _error(str(exc), analysis_type=normalized_analysis_type)


async def explore_entity(
    entity_id: str,
    corpus_data: str,
    include_relationships: bool = True,
    include_timeline: bool = True,
    include_sources: bool = True,
) -> Dict[str, Any]:
    """Explore detailed information about an investigation entity."""
    normalized_entity_id = str(entity_id or "").strip()
    normalized_corpus = str(corpus_data or "").strip()
    if not normalized_entity_id:
        return _error("entity_id must be a non-empty string", entity_id=entity_id)
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    if not isinstance(include_relationships, bool):
        return _error("include_relationships must be a boolean", include_relationships=include_relationships)
    if not isinstance(include_timeline, bool):
        return _error("include_timeline must be a boolean", include_timeline=include_timeline)
    if not isinstance(include_sources, bool):
        return _error("include_sources must be a boolean", include_sources=include_sources)
    try:
        payload = await _invoke(
            "explore_entity",
            entity_id=normalized_entity_id,
            corpus_data=normalized_corpus,
            include_relationships=include_relationships,
            include_timeline=include_timeline,
            include_sources=include_sources,
        )
        return _normalize_success(payload, entity_id=normalized_entity_id)
    except Exception as exc:
        return _error(str(exc), entity_id=normalized_entity_id)


async def map_relationships(
    corpus_data: str,
    relationship_types: Optional[List[str]] = None,
    min_strength: float = 0.5,
    max_depth: int = 3,
    focus_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """Map investigation relationships with deterministic validation and envelope shape."""
    normalized_corpus = str(corpus_data or "").strip()
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    list_error = _validate_string_list(relationship_types, "relationship_types")
    if list_error:
        return list_error
    try:
        normalized_strength = float(min_strength)
    except (TypeError, ValueError):
        return _error("min_strength must be a number between 0 and 1", min_strength=min_strength)
    if normalized_strength < 0 or normalized_strength > 1:
        return _error("min_strength must be a number between 0 and 1", min_strength=min_strength)
    if not isinstance(max_depth, int) or max_depth < 1:
        return _error("max_depth must be an integer >= 1", max_depth=max_depth)
    clean_focus_entity, error = _clean_optional_string(focus_entity, "focus_entity")
    if error:
        return error
    try:
        payload = await _invoke(
            "map_relationships",
            corpus_data=normalized_corpus,
            relationship_types=[item.strip() for item in relationship_types] if relationship_types is not None else None,
            min_strength=normalized_strength,
            max_depth=max_depth,
            focus_entity=clean_focus_entity,
        )
        return _normalize_success(
            payload,
            max_depth=max_depth,
            entities=[],
            relationships=[],
            graph_metrics={},
        )
    except Exception as exc:
        return _error(str(exc), max_depth=max_depth)


async def analyze_entity_timeline(
    corpus_data: str,
    entity_id: str,
    time_granularity: str = "day",
    include_related: bool = True,
    event_types: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze timeline events associated with a target entity."""
    normalized_corpus = str(corpus_data or "").strip()
    normalized_entity_id = str(entity_id or "").strip()
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    if not normalized_entity_id:
        return _error("entity_id must be a non-empty string", entity_id=entity_id)
    normalized_granularity = str(time_granularity or "").strip().lower()
    if normalized_granularity not in _VALID_TIME_GRANULARITIES:
        return _error(
            "time_granularity must be one of: day, hour, month, week",
            time_granularity=time_granularity,
        )
    if not isinstance(include_related, bool):
        return _error("include_related must be a boolean", include_related=include_related)
    list_error = _validate_string_list(event_types, "event_types")
    if list_error:
        return list_error
    try:
        payload = await _invoke(
            "analyze_entity_timeline",
            corpus_data=normalized_corpus,
            entity_id=normalized_entity_id,
            time_granularity=normalized_granularity,
            include_related=include_related,
            event_types=[item.strip() for item in event_types] if event_types is not None else None,
        )
        return _normalize_success(payload, entity_id=normalized_entity_id, time_granularity=normalized_granularity)
    except Exception as exc:
        return _error(str(exc), entity_id=normalized_entity_id)


async def detect_patterns(
    corpus_data: str,
    pattern_types: Optional[List[str]] = None,
    time_window: str = "30d",
    confidence_threshold: float = 0.7,
) -> Dict[str, Any]:
    """Detect investigation patterns in corpus data."""
    normalized_corpus = str(corpus_data or "").strip()
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    list_error = _validate_string_list(pattern_types, "pattern_types")
    if list_error:
        return list_error
    clean_pattern_types = [item.strip().lower() for item in pattern_types] if pattern_types is not None else None
    if clean_pattern_types is not None and any(item not in _VALID_PATTERN_TYPES for item in clean_pattern_types):
        return _error(
            "pattern_types must be drawn from: anomaly, behavioral, relational, temporal",
            pattern_types=pattern_types,
        )
    normalized_time_window = str(time_window or "").strip()
    if not normalized_time_window:
        return _error("time_window must be a non-empty string", time_window=time_window)
    try:
        normalized_threshold = float(confidence_threshold)
    except (TypeError, ValueError):
        return _error(
            "confidence_threshold must be a number between 0 and 1",
            confidence_threshold=confidence_threshold,
        )
    if normalized_threshold < 0 or normalized_threshold > 1:
        return _error(
            "confidence_threshold must be a number between 0 and 1",
            confidence_threshold=confidence_threshold,
        )
    try:
        payload = await _invoke(
            "detect_patterns",
            corpus_data=normalized_corpus,
            pattern_types=clean_pattern_types,
            time_window=normalized_time_window,
            confidence_threshold=normalized_threshold,
        )
        return _normalize_success(payload, time_window=normalized_time_window)
    except Exception as exc:
        return _error(str(exc), time_window=normalized_time_window)


async def track_provenance(
    corpus_data: str,
    entity_id: str,
    trace_depth: int = 5,
    include_citations: bool = True,
    include_transformations: bool = True,
) -> Dict[str, Any]:
    """Track provenance for a target entity."""
    normalized_corpus = str(corpus_data or "").strip()
    normalized_entity_id = str(entity_id or "").strip()
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    if not normalized_entity_id:
        return _error("entity_id must be a non-empty string", entity_id=entity_id)
    if not isinstance(trace_depth, int) or trace_depth < 1:
        return _error("trace_depth must be an integer >= 1", trace_depth=trace_depth)
    if not isinstance(include_citations, bool):
        return _error("include_citations must be a boolean", include_citations=include_citations)
    if not isinstance(include_transformations, bool):
        return _error("include_transformations must be a boolean", include_transformations=include_transformations)
    try:
        payload = await _invoke(
            "track_provenance",
            corpus_data=normalized_corpus,
            entity_id=normalized_entity_id,
            trace_depth=trace_depth,
            include_citations=include_citations,
            include_transformations=include_transformations,
        )
        return _normalize_success(payload, entity_id=normalized_entity_id, trace_depth=trace_depth)
    except Exception as exc:
        return _error(str(exc), entity_id=normalized_entity_id)


async def ingest_news_article(
    url: str,
    source_type: str = "news",
    analysis_type: str = "comprehensive",
    metadata: Optional[Any] = None,
) -> Dict[str, Any]:
    """Ingest a single article into the investigation pipeline."""
    normalized_url = str(url or "").strip()
    normalized_source_type = str(source_type or "").strip()
    normalized_analysis_type = str(analysis_type or "").strip()
    if not normalized_url:
        return _error("url must be a non-empty string", url=url)
    if not normalized_source_type:
        return _error("source_type must be a non-empty string", source_type=source_type)
    if not normalized_analysis_type:
        return _error("analysis_type must be a non-empty string", analysis_type=analysis_type)
    metadata_json, error = _json_argument(metadata, "metadata", "object")
    if error:
        return error
    try:
        payload = await _invoke(
            "ingest_news_article",
            url=normalized_url,
            source_type=normalized_source_type,
            analysis_type=normalized_analysis_type,
            metadata=metadata_json,
        )
        return _normalize_success(payload, url=normalized_url)
    except Exception as exc:
        return _error(str(exc), url=normalized_url)


async def ingest_news_feed(
    feed_url: str,
    max_articles: int = 50,
    filters: Optional[Any] = None,
    processing_mode: str = "parallel",
) -> Dict[str, Any]:
    """Ingest a feed source into the investigation pipeline."""
    normalized_feed_url = str(feed_url or "").strip()
    normalized_processing_mode = str(processing_mode or "").strip()
    if not normalized_feed_url:
        return _error("feed_url must be a non-empty string", feed_url=feed_url)
    if not isinstance(max_articles, int) or max_articles < 1:
        return _error("max_articles must be an integer >= 1", max_articles=max_articles)
    if not normalized_processing_mode:
        return _error("processing_mode must be a non-empty string", processing_mode=processing_mode)
    filters_json, error = _json_argument(filters, "filters", "object")
    if error:
        return error
    try:
        payload = await _invoke(
            "ingest_news_feed",
            feed_url=normalized_feed_url,
            max_articles=max_articles,
            filters=filters_json,
            processing_mode=normalized_processing_mode,
        )
        return _normalize_success(payload, feed_url=normalized_feed_url, max_articles=max_articles)
    except Exception as exc:
        return _error(str(exc), feed_url=normalized_feed_url)


async def ingest_website(
    base_url: str,
    max_pages: int = 100,
    max_depth: int = 3,
    url_patterns: Optional[Any] = None,
    content_types: Optional[Any] = None,
) -> Dict[str, Any]:
    """Crawl and ingest website content for investigation workflows."""
    normalized_base_url = str(base_url or "").strip()
    if not normalized_base_url:
        return _error("base_url must be a non-empty string", base_url=base_url)
    if not isinstance(max_pages, int) or max_pages < 1:
        return _error("max_pages must be an integer >= 1", max_pages=max_pages)
    if not isinstance(max_depth, int) or max_depth < 1:
        return _error("max_depth must be an integer >= 1", max_depth=max_depth)
    url_patterns_json, error = _json_argument(url_patterns, "url_patterns", "object")
    if error:
        return error
    content_types_json, error = _json_argument(content_types, "content_types", "array")
    if error:
        return error
    try:
        payload = await _invoke(
            "ingest_website",
            base_url=normalized_base_url,
            max_pages=max_pages,
            max_depth=max_depth,
            url_patterns=url_patterns_json,
            content_types=content_types_json,
        )
        return _normalize_success(payload, base_url=normalized_base_url, max_pages=max_pages)
    except Exception as exc:
        return _error(str(exc), base_url=normalized_base_url)


async def ingest_document_collection(
    document_paths: Any,
    collection_name: str = "document_collection",
    processing_options: Optional[Any] = None,
    metadata_extraction: bool = True,
) -> Dict[str, Any]:
    """Ingest a document collection into the investigation pipeline."""
    document_paths_json, error = _json_argument(document_paths, "document_paths", "array")
    if error:
        return error
    normalized_collection_name = str(collection_name or "").strip()
    if not normalized_collection_name:
        return _error("collection_name must be a non-empty string", collection_name=collection_name)
    processing_options_json, error = _json_argument(processing_options, "processing_options", "object")
    if error:
        return error
    if not isinstance(metadata_extraction, bool):
        return _error("metadata_extraction must be a boolean", metadata_extraction=metadata_extraction)
    try:
        payload = await _invoke(
            "ingest_document_collection",
            document_paths=document_paths_json,
            collection_name=normalized_collection_name,
            processing_options=processing_options_json,
            metadata_extraction=metadata_extraction,
        )
        return _normalize_success(payload, collection_name=normalized_collection_name)
    except Exception as exc:
        return _error(str(exc), collection_name=normalized_collection_name)


async def analyze_deontological_conflicts(
    corpus_data: str,
    conflict_types: Optional[List[str]] = None,
    severity_threshold: str = "medium",
    entity_filter: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Analyze deontological conflicts in corpus data."""
    normalized_corpus = str(corpus_data or "").strip()
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    list_error = _validate_string_list(conflict_types, "conflict_types")
    if list_error:
        return list_error
    normalized_conflict_types = [item.strip().lower() for item in conflict_types] if conflict_types is not None else None
    if normalized_conflict_types is not None and any(item not in _VALID_CONFLICT_TYPES for item in normalized_conflict_types):
        return _error(
            "conflict_types must be drawn from: conditional, direct, jurisdictional, temporal",
            conflict_types=conflict_types,
        )
    normalized_severity_threshold = str(severity_threshold or "").strip().lower()
    if normalized_severity_threshold not in _VALID_SEVERITY_LEVELS:
        return _error(
            "severity_threshold must be one of: high, low, medium",
            severity_threshold=severity_threshold,
        )
    list_error = _validate_string_list(entity_filter, "entity_filter")
    if list_error:
        return list_error
    try:
        payload = await _invoke(
            "analyze_deontological_conflicts",
            corpus_data=normalized_corpus,
            conflict_types=normalized_conflict_types,
            severity_threshold=normalized_severity_threshold,
            entity_filter=[item.strip() for item in entity_filter] if entity_filter is not None else None,
        )
        return _normalize_success(payload, severity_threshold=normalized_severity_threshold)
    except Exception as exc:
        return _error(str(exc), severity_threshold=severity_threshold)


async def query_deontic_statements(
    corpus_data: str,
    entity: Optional[str] = None,
    modality: Optional[str] = None,
    action_pattern: Optional[str] = None,
) -> Dict[str, Any]:
    """Query deontic statements by entity, modality, or action pattern."""
    normalized_corpus = str(corpus_data or "").strip()
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    clean_entity, error = _clean_optional_string(entity, "entity")
    if error:
        return error
    clean_action_pattern, error = _clean_optional_string(action_pattern, "action_pattern")
    if error:
        return error
    clean_modality, error = _clean_optional_string(modality, "modality")
    if error:
        return error
    normalized_modality = clean_modality.lower() if clean_modality is not None else None
    if normalized_modality is not None and normalized_modality not in _VALID_DEONTIC_MODALITIES:
        return _error(
            "modality must be null or one of: obligation, permission, prohibition",
            modality=modality,
        )
    try:
        payload = await _invoke(
            "query_deontic_statements",
            corpus_data=normalized_corpus,
            entity=clean_entity,
            modality=normalized_modality,
            action_pattern=clean_action_pattern,
        )
        return _normalize_success(payload)
    except Exception as exc:
        return _error(str(exc), modality=normalized_modality)


async def query_deontic_conflicts(
    corpus_data: str,
    conflict_type: Optional[str] = None,
    severity: Optional[str] = None,
    entity: Optional[str] = None,
) -> Dict[str, Any]:
    """Query deontic conflicts by conflict type, severity, or entity."""
    normalized_corpus = str(corpus_data or "").strip()
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    clean_entity, error = _clean_optional_string(entity, "entity")
    if error:
        return error
    clean_conflict_type, error = _clean_optional_string(conflict_type, "conflict_type")
    if error:
        return error
    clean_severity, error = _clean_optional_string(severity, "severity")
    if error:
        return error
    normalized_conflict_type = clean_conflict_type.lower() if clean_conflict_type is not None else None
    normalized_severity = clean_severity.lower() if clean_severity is not None else None
    if normalized_conflict_type is not None and normalized_conflict_type not in _VALID_CONFLICT_TYPES:
        return _error(
            "conflict_type must be null or one of: conditional, direct, jurisdictional, temporal",
            conflict_type=conflict_type,
        )
    if normalized_severity is not None and normalized_severity not in _VALID_SEVERITY_LEVELS:
        return _error(
            "severity must be null or one of: high, low, medium",
            severity=severity,
        )
    try:
        payload = await _invoke(
            "query_deontic_conflicts",
            corpus_data=normalized_corpus,
            conflict_type=normalized_conflict_type,
            severity=normalized_severity,
            entity=clean_entity,
        )
        return _normalize_success(payload)
    except Exception as exc:
        return _error(str(exc), severity=normalized_severity)


async def extract_geographic_entities(
    corpus_data: str,
    confidence_threshold: float = 0.8,
    entity_types: Optional[List[str]] = None,
    include_coordinates: bool = True,
    geographic_scope: Optional[str] = None,
) -> Dict[str, Any]:
    """Extract geographic entities from investigation corpus data."""
    normalized_corpus = str(corpus_data or "").strip()
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    try:
        normalized_threshold = float(confidence_threshold)
    except (TypeError, ValueError):
        return _error(
            "confidence_threshold must be a number between 0 and 1",
            confidence_threshold=confidence_threshold,
        )
    if normalized_threshold < 0 or normalized_threshold > 1:
        return _error(
            "confidence_threshold must be a number between 0 and 1",
            confidence_threshold=confidence_threshold,
        )
    list_error = _validate_string_list(entity_types, "entity_types")
    if list_error:
        return list_error
    if not isinstance(include_coordinates, bool):
        return _error("include_coordinates must be a boolean", include_coordinates=include_coordinates)
    clean_scope, error = _clean_optional_string(geographic_scope, "geographic_scope")
    if error:
        return error
    try:
        payload = await _invoke(
            "extract_geographic_entities",
            corpus_data=normalized_corpus,
            confidence_threshold=normalized_threshold,
            entity_types=[item.strip() for item in entity_types] if entity_types is not None else None,
            include_coordinates=include_coordinates,
            geographic_scope=clean_scope,
        )
        return _normalize_success(payload, confidence_threshold=normalized_threshold)
    except Exception as exc:
        return _error(str(exc), confidence_threshold=confidence_threshold)


async def map_spatiotemporal_events(
    corpus_data: str,
    time_range: Optional[Dict[str, str]] = None,
    geographic_bounds: Optional[Dict[str, float]] = None,
    event_types: Optional[List[str]] = None,
    clustering_distance: float = 50.0,
    temporal_resolution: str = "day",
) -> Dict[str, Any]:
    """Map spatial and temporal events in investigation corpus data."""
    normalized_corpus = str(corpus_data or "").strip()
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    if time_range is not None and not isinstance(time_range, dict):
        return _error("time_range must be null or an object", time_range=time_range)
    if geographic_bounds is not None and not isinstance(geographic_bounds, dict):
        return _error("geographic_bounds must be null or an object", geographic_bounds=geographic_bounds)
    list_error = _validate_string_list(event_types, "event_types")
    if list_error:
        return list_error
    try:
        normalized_distance = float(clustering_distance)
    except (TypeError, ValueError):
        return _error("clustering_distance must be a number > 0", clustering_distance=clustering_distance)
    if normalized_distance <= 0:
        return _error("clustering_distance must be a number > 0", clustering_distance=clustering_distance)
    normalized_temporal_resolution = str(temporal_resolution or "").strip().lower()
    if normalized_temporal_resolution not in _VALID_TEMPORAL_RESOLUTIONS:
        return _error(
            "temporal_resolution must be one of: day, hour, month, week",
            temporal_resolution=temporal_resolution,
        )
    try:
        payload = await _invoke(
            "map_spatiotemporal_events",
            corpus_data=normalized_corpus,
            time_range=time_range,
            geographic_bounds=geographic_bounds,
            event_types=[item.strip() for item in event_types] if event_types is not None else None,
            clustering_distance=normalized_distance,
            temporal_resolution=normalized_temporal_resolution,
        )
        return _normalize_success(payload, temporal_resolution=normalized_temporal_resolution)
    except Exception as exc:
        return _error(str(exc), temporal_resolution=temporal_resolution)


async def query_geographic_context(
    query: str,
    corpus_data: str,
    radius_km: float = 100.0,
    center_location: Optional[str] = None,
    include_related_entities: bool = True,
    temporal_context: bool = True,
) -> Dict[str, Any]:
    """Query geographic context around investigation corpus data."""
    normalized_query = str(query or "").strip()
    normalized_corpus = str(corpus_data or "").strip()
    if not normalized_query:
        return _error("query must be a non-empty string", query=query)
    if not normalized_corpus:
        return _error("corpus_data must be a non-empty string", corpus_data=corpus_data)
    try:
        normalized_radius = float(radius_km)
    except (TypeError, ValueError):
        return _error("radius_km must be a number > 0", radius_km=radius_km)
    if normalized_radius <= 0:
        return _error("radius_km must be a number > 0", radius_km=radius_km)
    clean_center_location, error = _clean_optional_string(center_location, "center_location")
    if error:
        return error
    if not isinstance(include_related_entities, bool):
        return _error(
            "include_related_entities must be a boolean",
            include_related_entities=include_related_entities,
        )
    if not isinstance(temporal_context, bool):
        return _error("temporal_context must be a boolean", temporal_context=temporal_context)
    try:
        payload = await _invoke(
            "query_geographic_context",
            query=normalized_query,
            corpus_data=normalized_corpus,
            radius_km=normalized_radius,
            center_location=clean_center_location,
            include_related_entities=include_related_entities,
            temporal_context=temporal_context,
        )
        return _normalize_success(payload, query=normalized_query, radius_km=normalized_radius, results=[])
    except Exception as exc:
        return _error(str(exc), query=normalized_query)


def register_native_investigation_tools(manager: Any) -> None:
    """Register native investigation tools in unified hierarchical manager."""
    tool_specs = [
        {
            "name": "analyze_entities",
            "func": analyze_entities,
            "description": "Analyze entities in a corpus for investigation workflows.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "corpus_data": {"type": "string", "minLength": 1},
                    "analysis_type": {"type": "string", "minLength": 1, "default": "comprehensive"},
                    "entity_types": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.85},
                    "user_context": {"type": ["string", "null"]},
                },
                "required": ["corpus_data"],
            },
        },
        {
            "name": "explore_entity",
            "func": explore_entity,
            "description": "Explore detailed information about a specific investigation entity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "entity_id": {"type": "string", "minLength": 1},
                    "corpus_data": {"type": "string", "minLength": 1},
                    "include_relationships": {"type": "boolean", "default": True},
                    "include_timeline": {"type": "boolean", "default": True},
                    "include_sources": {"type": "boolean", "default": True},
                },
                "required": ["entity_id", "corpus_data"],
            },
        },
        {
            "name": "map_relationships",
            "func": map_relationships,
            "description": "Map relationships between entities in an investigation corpus.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "corpus_data": {"type": "string", "minLength": 1},
                    "relationship_types": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                    "min_strength": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                    "max_depth": {"type": "integer", "minimum": 1, "default": 3},
                    "focus_entity": {"type": ["string", "null"]},
                },
                "required": ["corpus_data"],
            },
        },
        {
            "name": "analyze_entity_timeline",
            "func": analyze_entity_timeline,
            "description": "Analyze timeline events for a specific entity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "corpus_data": {"type": "string", "minLength": 1},
                    "entity_id": {"type": "string", "minLength": 1},
                    "time_granularity": {"type": "string", "enum": ["hour", "day", "week", "month"], "default": "day"},
                    "include_related": {"type": "boolean", "default": True},
                    "event_types": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                },
                "required": ["corpus_data", "entity_id"],
            },
        },
        {
            "name": "detect_patterns",
            "func": detect_patterns,
            "description": "Detect behavioral, relational, temporal, and anomaly patterns.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "corpus_data": {"type": "string", "minLength": 1},
                    "pattern_types": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                    "time_window": {"type": "string", "minLength": 1, "default": "30d"},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.7},
                },
                "required": ["corpus_data"],
            },
        },
        {
            "name": "track_provenance",
            "func": track_provenance,
            "description": "Track information provenance for an investigation entity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "corpus_data": {"type": "string", "minLength": 1},
                    "entity_id": {"type": "string", "minLength": 1},
                    "trace_depth": {"type": "integer", "minimum": 1, "default": 5},
                    "include_citations": {"type": "boolean", "default": True},
                    "include_transformations": {"type": "boolean", "default": True},
                },
                "required": ["corpus_data", "entity_id"],
            },
        },
        {
            "name": "ingest_news_article",
            "func": ingest_news_article,
            "description": "Ingest a single article for investigation analysis.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "minLength": 1},
                    "source_type": {"type": "string", "minLength": 1, "default": "news"},
                    "analysis_type": {"type": "string", "minLength": 1, "default": "comprehensive"},
                    "metadata": {"type": ["object", "string", "null"]},
                },
                "required": ["url"],
            },
        },
        {
            "name": "ingest_news_feed",
            "func": ingest_news_feed,
            "description": "Ingest multiple articles from a feed or RSS source.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "feed_url": {"type": "string", "minLength": 1},
                    "max_articles": {"type": "integer", "minimum": 1, "default": 50},
                    "filters": {"type": ["object", "string", "null"]},
                    "processing_mode": {"type": "string", "minLength": 1, "default": "parallel"},
                },
                "required": ["feed_url"],
            },
        },
        {
            "name": "ingest_website",
            "func": ingest_website,
            "description": "Crawl and ingest website content for investigation analysis.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "base_url": {"type": "string", "minLength": 1},
                    "max_pages": {"type": "integer", "minimum": 1, "default": 100},
                    "max_depth": {"type": "integer", "minimum": 1, "default": 3},
                    "url_patterns": {"type": ["object", "string", "null"]},
                    "content_types": {"type": ["array", "string", "null"]},
                },
                "required": ["base_url"],
            },
        },
        {
            "name": "ingest_document_collection",
            "func": ingest_document_collection,
            "description": "Ingest a document collection for investigation analysis.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "document_paths": {"type": ["array", "string"]},
                    "collection_name": {"type": "string", "minLength": 1, "default": "document_collection"},
                    "processing_options": {"type": ["object", "string", "null"]},
                    "metadata_extraction": {"type": "boolean", "default": True},
                },
                "required": ["document_paths"],
            },
        },
        {
            "name": "analyze_deontological_conflicts",
            "func": analyze_deontological_conflicts,
            "description": "Analyze deontological conflicts in a corpus.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "corpus_data": {"type": "string", "minLength": 1},
                    "conflict_types": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                    "severity_threshold": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
                    "entity_filter": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                },
                "required": ["corpus_data"],
            },
        },
        {
            "name": "query_deontic_statements",
            "func": query_deontic_statements,
            "description": "Query deontic statements by entity, modality, or action pattern.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "corpus_data": {"type": "string", "minLength": 1},
                    "entity": {"type": ["string", "null"]},
                    "modality": {"type": ["string", "null"], "enum": ["obligation", "permission", "prohibition", None]},
                    "action_pattern": {"type": ["string", "null"]},
                },
                "required": ["corpus_data"],
            },
        },
        {
            "name": "query_deontic_conflicts",
            "func": query_deontic_conflicts,
            "description": "Query deontic conflicts by type, severity, or entity.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "corpus_data": {"type": "string", "minLength": 1},
                    "conflict_type": {"type": ["string", "null"], "enum": ["direct", "conditional", "jurisdictional", "temporal", None]},
                    "severity": {"type": ["string", "null"], "enum": ["low", "medium", "high", None]},
                    "entity": {"type": ["string", "null"]},
                },
                "required": ["corpus_data"],
            },
        },
        {
            "name": "extract_geographic_entities",
            "func": extract_geographic_entities,
            "description": "Extract geographic entities and optional coordinate metadata.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "corpus_data": {"type": "string", "minLength": 1},
                    "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.8},
                    "entity_types": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                    "include_coordinates": {"type": "boolean", "default": True},
                    "geographic_scope": {"type": ["string", "null"]},
                },
                "required": ["corpus_data"],
            },
        },
        {
            "name": "map_spatiotemporal_events",
            "func": map_spatiotemporal_events,
            "description": "Map events with spatial and temporal dimensions.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "corpus_data": {"type": "string", "minLength": 1},
                    "time_range": {"type": ["object", "null"]},
                    "geographic_bounds": {"type": ["object", "null"]},
                    "event_types": {"type": ["array", "null"], "items": {"type": "string", "minLength": 1}},
                    "clustering_distance": {"type": "number", "exclusiveMinimum": 0, "default": 50.0},
                    "temporal_resolution": {"type": "string", "enum": ["hour", "day", "week", "month"], "default": "day"},
                },
                "required": ["corpus_data"],
            },
        },
        {
            "name": "query_geographic_context",
            "func": query_geographic_context,
            "description": "Query geographic context around corpus data.",
            "input_schema": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "minLength": 1},
                    "corpus_data": {"type": "string", "minLength": 1},
                    "radius_km": {"type": "number", "exclusiveMinimum": 0, "default": 100.0},
                    "center_location": {"type": ["string", "null"]},
                    "include_related_entities": {"type": "boolean", "default": True},
                    "temporal_context": {"type": "boolean", "default": True},
                },
                "required": ["query", "corpus_data"],
            },
        },
    ]

    for spec in tool_specs:
        manager.register_tool(
            category="investigation_tools",
            name=spec["name"],
            func=spec["func"],
            description=spec["description"],
            input_schema=spec["input_schema"],
            runtime="fastapi",
            tags=["native", "mcpp", "investigation"],
        )


__all__ = [
    "analyze_entities",
    "explore_entity",
    "map_relationships",
    "analyze_entity_timeline",
    "detect_patterns",
    "track_provenance",
    "ingest_news_article",
    "ingest_news_feed",
    "ingest_website",
    "ingest_document_collection",
    "analyze_deontological_conflicts",
    "query_deontic_statements",
    "query_deontic_conflicts",
    "extract_geographic_entities",
    "map_spatiotemporal_events",
    "query_geographic_context",
    "register_native_investigation_tools",
]
