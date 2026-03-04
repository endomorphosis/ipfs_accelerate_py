"""Native investigation-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_investigation_tools_api() -> Dict[str, Any]:
    """Resolve source investigation-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.investigation_tools import (  # type: ignore
            analyze_entities as _analyze_entities,
            map_relationships as _map_relationships,
        )

        return {
            "analyze_entities": _analyze_entities,
            "map_relationships": _map_relationships,
        }
    except Exception:
        logger.warning(
            "Source investigation_tools import unavailable, using fallback investigation functions"
        )

        async def _analyze_entities_fallback(
            corpus_data: str,
            analysis_type: str = "comprehensive",
            entity_types: Optional[List[str]] = None,
            confidence_threshold: float = 0.85,
            user_context: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (corpus_data, entity_types, confidence_threshold, user_context)
            return {
                "status": "success",
                "analysis_type": analysis_type,
                "entities": [],
                "relationships": [],
                "fallback": True,
            }

        async def _map_relationships_fallback(
            corpus_data: str,
            relationship_types: Optional[List[str]] = None,
            min_strength: float = 0.5,
            max_depth: int = 3,
            focus_entity: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = (corpus_data, relationship_types, min_strength, max_depth, focus_entity)
            return {
                "status": "success",
                "entities": [],
                "relationships": [],
                "fallback": True,
            }

        return {
            "analyze_entities": _analyze_entities_fallback,
            "map_relationships": _map_relationships_fallback,
        }


_API = _load_investigation_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        return payload
    if payload is None:
        return {}
    return {"result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def analyze_entities(
    corpus_data: str,
    analysis_type: str = "comprehensive",
    entity_types: Optional[List[str]] = None,
    confidence_threshold: float = 0.85,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze entities in corpus data for investigation workflows."""
    if not isinstance(corpus_data, str) or not corpus_data.strip():
        return _error_result("corpus_data must be a non-empty string", corpus_data=corpus_data)
    if not isinstance(analysis_type, str) or not analysis_type.strip():
        return _error_result("analysis_type must be a non-empty string", analysis_type=analysis_type)
    if entity_types is not None and (
        not isinstance(entity_types, list)
        or not all(isinstance(item, str) and item.strip() for item in entity_types)
    ):
        return _error_result(
            "entity_types must be null or a list of non-empty strings",
            entity_types=entity_types,
        )
    if not isinstance(confidence_threshold, (int, float)) or confidence_threshold < 0 or confidence_threshold > 1:
        return _error_result(
            "confidence_threshold must be a number between 0 and 1",
            confidence_threshold=confidence_threshold,
        )
    if user_context is not None and (not isinstance(user_context, str) or not user_context.strip()):
        return _error_result("user_context must be null or a non-empty string", user_context=user_context)

    clean_entity_types = [item.strip() for item in entity_types] if entity_types is not None else None
    clean_user_context = user_context.strip() if isinstance(user_context, str) else None
    clean_analysis_type = analysis_type.strip()
    clean_corpus_data = corpus_data.strip()

    try:
        result = _API["analyze_entities"](
            corpus_data=clean_corpus_data,
            analysis_type=clean_analysis_type,
            entity_types=clean_entity_types,
            confidence_threshold=float(confidence_threshold),
            user_context=clean_user_context,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("analysis_type", clean_analysis_type)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), analysis_type=clean_analysis_type)


async def map_relationships(
    corpus_data: str,
    relationship_types: Optional[List[str]] = None,
    min_strength: float = 0.5,
    max_depth: int = 3,
    focus_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """Map relationships between entities in investigation corpus data."""
    if not isinstance(corpus_data, str) or not corpus_data.strip():
        return _error_result("corpus_data must be a non-empty string", corpus_data=corpus_data)
    if relationship_types is not None and (
        not isinstance(relationship_types, list)
        or not all(isinstance(item, str) and item.strip() for item in relationship_types)
    ):
        return _error_result(
            "relationship_types must be null or a list of non-empty strings",
            relationship_types=relationship_types,
        )
    if not isinstance(min_strength, (int, float)) or min_strength < 0 or min_strength > 1:
        return _error_result("min_strength must be a number between 0 and 1", min_strength=min_strength)
    if not isinstance(max_depth, int) or max_depth < 1:
        return _error_result("max_depth must be an integer >= 1", max_depth=max_depth)
    if focus_entity is not None and (not isinstance(focus_entity, str) or not focus_entity.strip()):
        return _error_result("focus_entity must be null or a non-empty string", focus_entity=focus_entity)

    clean_relationship_types = (
        [item.strip() for item in relationship_types] if relationship_types is not None else None
    )
    clean_focus_entity = focus_entity.strip() if isinstance(focus_entity, str) else None
    clean_corpus_data = corpus_data.strip()

    try:
        result = _API["map_relationships"](
            corpus_data=clean_corpus_data,
            relationship_types=clean_relationship_types,
            min_strength=float(min_strength),
            max_depth=max_depth,
            focus_entity=clean_focus_entity,
        )
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("max_depth", max_depth)
        return envelope
    except Exception as exc:
        return _error_result(str(exc), max_depth=max_depth)


def register_native_investigation_tools(manager: Any) -> None:
    """Register native investigation-tools category tools in unified manager."""
    manager.register_tool(
        category="investigation_tools",
        name="analyze_entities",
        func=analyze_entities,
        description="Analyze entities in a corpus for investigative workflows.",
        input_schema={
            "type": "object",
            "properties": {
                "corpus_data": {"type": "string"},
                "analysis_type": {"type": "string", "minLength": 1, "default": "comprehensive"},
                "entity_types": {"type": ["array", "null"], "items": {"type": "string"}},
                "confidence_threshold": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.85},
                "user_context": {"type": ["string", "null"]},
            },
            "required": ["corpus_data"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "investigation-tools"],
    )

    manager.register_tool(
        category="investigation_tools",
        name="map_relationships",
        func=map_relationships,
        description="Map entity relationships in a corpus for investigative workflows.",
        input_schema={
            "type": "object",
            "properties": {
                "corpus_data": {"type": "string"},
                "relationship_types": {"type": ["array", "null"], "items": {"type": "string"}},
                "min_strength": {"type": "number", "minimum": 0, "maximum": 1, "default": 0.5},
                "max_depth": {"type": "integer", "minimum": 1, "default": 3},
                "focus_entity": {"type": ["string", "null"]},
            },
            "required": ["corpus_data"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "investigation-tools"],
    )
