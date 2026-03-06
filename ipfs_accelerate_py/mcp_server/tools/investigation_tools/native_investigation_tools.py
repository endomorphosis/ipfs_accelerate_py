"""Native investigation tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_investigation_api() -> Dict[str, Any]:
    """Resolve source investigation APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.investigation_tools.entity_analysis_tools import (  # type: ignore
            analyze_entities as _analyze_entities,
        )
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.investigation_tools.relationship_timeline_tools import (  # type: ignore
            map_relationships as _map_relationships,
        )

        return {
            "analyze_entities": _analyze_entities,
            "map_relationships": _map_relationships,
        }
    except Exception:
        logger.warning("Source investigation_tools import unavailable, using fallback investigation functions")

        async def _analyze_entities_fallback(
            corpus_data: str,
            analysis_type: str = "comprehensive",
            entity_types: Optional[List[str]] = None,
            confidence_threshold: float = 0.85,
            user_context: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = corpus_data, user_context
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "entity_types": list(entity_types or ["PERSON", "ORG", "GPE", "EVENT"]),
                "confidence_threshold": float(confidence_threshold),
                "entities": [],
                "relationships": [],
                "clusters": [],
                "statistics": {
                    "total_entities": 0,
                    "total_relationships": 0,
                },
            }

        async def _map_relationships_fallback(
            corpus_data: str,
            relationship_types: Optional[List[str]] = None,
            min_strength: float = 0.5,
            max_depth: int = 3,
            focus_entity: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = corpus_data
            return {
                "status": "success",
                "relationship_types": list(
                    relationship_types or ["co_occurrence", "citation", "semantic", "temporal"]
                ),
                "min_strength": float(min_strength),
                "max_depth": int(max_depth),
                "focus_entity": focus_entity,
                "entities": [],
                "relationships": [],
                "clusters": [],
                "graph_metrics": {},
            }

        return {
            "analyze_entities": _analyze_entities_fallback,
            "map_relationships": _map_relationships_fallback,
        }


_API = _load_investigation_api()


def _error(message: str, **extra: Any) -> Dict[str, Any]:
    payload: Dict[str, Any] = {"status": "error", "error": message}
    payload.update(extra)
    return payload


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
    try:
        normalized_threshold = float(confidence_threshold)
    except (TypeError, ValueError):
        return _error(
            "confidence_threshold must be a number between 0 and 1",
            confidence_threshold=confidence_threshold,
            analysis_type=normalized_analysis_type,
        )
    if normalized_threshold < 0 or normalized_threshold > 1:
        return _error(
            "confidence_threshold must be a number between 0 and 1",
            confidence_threshold=confidence_threshold,
            analysis_type=normalized_analysis_type,
        )

    if entity_types is not None:
        if not isinstance(entity_types, list) or not all(isinstance(item, str) and item.strip() for item in entity_types):
            return _error(
                "entity_types must be an array of non-empty strings when provided",
                entity_types=entity_types,
                analysis_type=normalized_analysis_type,
            )

    result = _API["analyze_entities"](
        corpus_data=normalized_corpus,
        analysis_type=normalized_analysis_type,
        entity_types=entity_types,
        confidence_threshold=normalized_threshold,
        user_context=user_context,
    )
    if hasattr(result, "__await__"):
        payload = dict(await result or {})
    else:
        payload = dict(result or {})
    if payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("analysis_type", normalized_analysis_type)
    payload.setdefault("confidence_threshold", normalized_threshold)
    payload.setdefault("entity_types", list(entity_types or []))
    return payload


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

    try:
        normalized_strength = float(min_strength)
    except (TypeError, ValueError):
        return _error("min_strength must be a number between 0 and 1", min_strength=min_strength)
    if normalized_strength < 0 or normalized_strength > 1:
        return _error("min_strength must be between 0 and 1", min_strength=min_strength)

    try:
        normalized_depth = int(max_depth)
    except (TypeError, ValueError):
        return _error("max_depth must be an integer >= 1", max_depth=max_depth)
    if normalized_depth < 1:
        return _error("max_depth must be an integer >= 1", max_depth=max_depth)

    if relationship_types is not None:
        if not isinstance(relationship_types, list) or not all(
            isinstance(item, str) and item.strip() for item in relationship_types
        ):
            return _error(
                "relationship_types must be an array of non-empty strings when provided",
                relationship_types=relationship_types,
                max_depth=normalized_depth,
            )

    result = _API["map_relationships"](
        corpus_data=normalized_corpus,
        relationship_types=relationship_types,
        min_strength=normalized_strength,
        max_depth=normalized_depth,
        focus_entity=focus_entity,
    )
    if hasattr(result, "__await__"):
        payload = dict(await result or {})
    else:
        payload = dict(result or {})
    if payload.get("error"):
        payload.setdefault("status", "error")
    else:
        payload.setdefault("status", "success")
    payload.setdefault("min_strength", normalized_strength)
    payload.setdefault("max_depth", normalized_depth)
    payload.setdefault("relationship_types", list(relationship_types or []))
    return payload


def register_native_investigation_tools(manager: Any) -> None:
    """Register native investigation tools in unified hierarchical manager."""
    manager.register_tool(
        category="investigation_tools",
        name="analyze_entities",
        func=analyze_entities,
        description="Analyze entities in a corpus for investigation workflows.",
        input_schema={
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
        runtime="fastapi",
        tags=["native", "mcpp", "investigation"],
    )

    manager.register_tool(
        category="investigation_tools",
        name="map_relationships",
        func=map_relationships,
        description="Map relationships between entities in an investigation corpus.",
        input_schema={
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
        runtime="fastapi",
        tags=["native", "mcpp", "investigation"],
    )


__all__ = ["analyze_entities", "map_relationships", "register_native_investigation_tools"]
