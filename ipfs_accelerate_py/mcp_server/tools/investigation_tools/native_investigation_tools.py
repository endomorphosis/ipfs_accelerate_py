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


async def analyze_entities(
    corpus_data: str,
    analysis_type: str = "comprehensive",
    entity_types: Optional[List[str]] = None,
    confidence_threshold: float = 0.85,
    user_context: Optional[str] = None,
) -> Dict[str, Any]:
    """Analyze entities in corpus data for investigation workflows."""
    result = _API["analyze_entities"](
        corpus_data=corpus_data,
        analysis_type=analysis_type,
        entity_types=entity_types,
        confidence_threshold=confidence_threshold,
        user_context=user_context,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


async def map_relationships(
    corpus_data: str,
    relationship_types: Optional[List[str]] = None,
    min_strength: float = 0.5,
    max_depth: int = 3,
    focus_entity: Optional[str] = None,
) -> Dict[str, Any]:
    """Map relationships between entities in investigation corpus data."""
    result = _API["map_relationships"](
        corpus_data=corpus_data,
        relationship_types=relationship_types,
        min_strength=min_strength,
        max_depth=max_depth,
        focus_entity=focus_entity,
    )
    if hasattr(result, "__await__"):
        return await result
    return result


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
                "analysis_type": {"type": "string"},
                "entity_types": {"type": ["array", "null"], "items": {"type": "string"}},
                "confidence_threshold": {"type": "number"},
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
                "min_strength": {"type": "number"},
                "max_depth": {"type": "integer"},
                "focus_entity": {"type": ["string", "null"]},
            },
            "required": ["corpus_data"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "investigation-tools"],
    )
