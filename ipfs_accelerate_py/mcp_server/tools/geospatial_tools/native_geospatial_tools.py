"""Native geospatial tool implementations for unified mcp_server."""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def _load_geospatial_api() -> Dict[str, Any]:
    """Resolve source geospatial APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.geospatial_tools.geospatial_tools import (  # type: ignore
            extract_geographic_entities as _extract_geographic_entities,
            map_spatiotemporal_events as _map_spatiotemporal_events,
            query_geographic_context as _query_geographic_context,
        )

        def _extract(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return json.loads(_extract_geographic_entities(*args, **kwargs))

        def _map(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return json.loads(_map_spatiotemporal_events(*args, **kwargs))

        def _query(*args: Any, **kwargs: Any) -> Dict[str, Any]:
            return json.loads(_query_geographic_context(*args, **kwargs))

        return {
            "extract_geographic_entities": _extract,
            "map_spatiotemporal_events": _map,
            "query_geographic_context": _query,
        }
    except Exception:
        logger.warning("Source geospatial_tools import unavailable, using fallback geospatial functions")

        def _extract_fallback(
            corpus_data: str,
            confidence_threshold: float = 0.7,
            include_coordinates: bool = True,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "corpus_length": len(corpus_data or ""),
                "confidence_threshold": confidence_threshold,
                "include_coordinates": include_coordinates,
                "entities": [],
                "entity_count": 0,
            }

        def _map_fallback(
            corpus_data: str,
            time_range: Optional[Dict[str, Any]] = None,
            clustering_distance: float = 50.0,
            temporal_resolution: str = "day",
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "corpus_length": len(corpus_data or ""),
                "time_range": time_range or {},
                "clustering_distance": clustering_distance,
                "temporal_resolution": temporal_resolution,
                "clusters": [],
                "cluster_count": 0,
            }

        def _query_fallback(
            query: str,
            corpus_data: str,
            radius_km: float = 100.0,
            center_location: Optional[str] = None,
            include_related_entities: bool = True,
            temporal_context: bool = True,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "query": query,
                "corpus_length": len(corpus_data or ""),
                "radius_km": radius_km,
                "center_location": center_location,
                "include_related_entities": include_related_entities,
                "temporal_context": temporal_context,
                "matches": [],
                "match_count": 0,
            }

        return {
            "extract_geographic_entities": _extract_fallback,
            "map_spatiotemporal_events": _map_fallback,
            "query_geographic_context": _query_fallback,
        }


_API = _load_geospatial_api()


async def extract_geographic_entities(
    corpus_data: str,
    confidence_threshold: float = 0.7,
    include_coordinates: bool = True,
) -> Dict[str, Any]:
    """Extract geographic entities from corpus data."""
    normalized_corpus = str(corpus_data or "")
    if not normalized_corpus.strip():
        return {
            "status": "error",
            "message": "corpus_data is required",
            "corpus_data": corpus_data,
        }
    try:
        normalized_threshold = float(confidence_threshold)
    except (TypeError, ValueError):
        return {
            "status": "error",
            "message": "confidence_threshold must be numeric",
            "confidence_threshold": confidence_threshold,
        }
    if normalized_threshold < 0.0 or normalized_threshold > 1.0:
        return {
            "status": "error",
            "message": "confidence_threshold must be between 0 and 1",
            "confidence_threshold": confidence_threshold,
        }
    if not isinstance(include_coordinates, bool):
        return {
            "status": "error",
            "message": "include_coordinates must be a boolean",
            "include_coordinates": include_coordinates,
        }

    result = _API["extract_geographic_entities"](
        corpus_data=corpus_data,
        confidence_threshold=normalized_threshold,
        include_coordinates=include_coordinates,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    return payload


async def map_spatiotemporal_events(
    corpus_data: str,
    time_range: Optional[Dict[str, Any]] = None,
    clustering_distance: float = 50.0,
    temporal_resolution: str = "day",
) -> Dict[str, Any]:
    """Map events with spatial-temporal clustering analysis."""
    normalized_corpus = str(corpus_data or "")
    if not normalized_corpus.strip():
        return {
            "status": "error",
            "message": "corpus_data is required",
            "corpus_data": corpus_data,
        }
    if time_range is not None and not isinstance(time_range, dict):
        return {
            "status": "error",
            "message": "time_range must be an object when provided",
            "time_range": time_range,
        }
    try:
        normalized_distance = float(clustering_distance)
    except (TypeError, ValueError):
        return {
            "status": "error",
            "message": "clustering_distance must be numeric",
            "clustering_distance": clustering_distance,
        }
    if normalized_distance <= 0:
        return {
            "status": "error",
            "message": "clustering_distance must be positive",
            "clustering_distance": clustering_distance,
        }

    normalized_resolution = str(temporal_resolution or "").strip().lower()
    allowed_resolutions = {"hour", "day", "week", "month", "year"}
    if normalized_resolution not in allowed_resolutions:
        return {
            "status": "error",
            "message": "temporal_resolution must be one of: hour, day, week, month, year",
            "temporal_resolution": temporal_resolution,
        }

    result = _API["map_spatiotemporal_events"](
        corpus_data=corpus_data,
        time_range=time_range,
        clustering_distance=normalized_distance,
        temporal_resolution=normalized_resolution,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    return payload


async def query_geographic_context(
    query: str,
    corpus_data: str,
    radius_km: float = 100.0,
    center_location: Optional[str] = None,
    include_related_entities: bool = True,
    temporal_context: bool = True,
) -> Dict[str, Any]:
    """Perform natural language geographic queries."""
    normalized_query = str(query or "")
    normalized_corpus = str(corpus_data or "")
    if not normalized_query.strip():
        return {
            "status": "error",
            "message": "query is required",
            "query": query,
        }
    if not normalized_corpus.strip():
        return {
            "status": "error",
            "message": "corpus_data is required",
            "corpus_data": corpus_data,
        }
    try:
        normalized_radius = float(radius_km)
    except (TypeError, ValueError):
        return {
            "status": "error",
            "message": "radius_km must be numeric",
            "radius_km": radius_km,
        }
    if normalized_radius <= 0:
        return {
            "status": "error",
            "message": "radius_km must be positive",
            "radius_km": radius_km,
        }
    if center_location is not None and not str(center_location).strip():
        return {
            "status": "error",
            "message": "center_location must be a non-empty string when provided",
            "center_location": center_location,
        }
    if not isinstance(include_related_entities, bool):
        return {
            "status": "error",
            "message": "include_related_entities must be a boolean",
            "include_related_entities": include_related_entities,
        }
    if not isinstance(temporal_context, bool):
        return {
            "status": "error",
            "message": "temporal_context must be a boolean",
            "temporal_context": temporal_context,
        }

    result = _API["query_geographic_context"](
        query=query,
        corpus_data=corpus_data,
        radius_km=normalized_radius,
        center_location=center_location,
        include_related_entities=include_related_entities,
        temporal_context=temporal_context,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    return payload


def register_native_geospatial_tools(manager: Any) -> None:
    """Register native geospatial tools in unified hierarchical manager."""
    manager.register_tool(
        category="geospatial_tools",
        name="extract_geographic_entities",
        func=extract_geographic_entities,
        description="Extract geographic entities from corpus data.",
        input_schema={
            "type": "object",
            "properties": {
                "corpus_data": {"type": "string"},
                "confidence_threshold": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.7},
                "include_coordinates": {"type": "boolean", "default": True},
            },
            "required": ["corpus_data"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "geospatial"],
    )

    manager.register_tool(
        category="geospatial_tools",
        name="map_spatiotemporal_events",
        func=map_spatiotemporal_events,
        description="Map events using spatial-temporal clustering.",
        input_schema={
            "type": "object",
            "properties": {
                "corpus_data": {"type": "string"},
                "time_range": {"type": ["object", "null"]},
                "clustering_distance": {"type": "number", "minimum": 0.000001, "default": 50.0},
                "temporal_resolution": {
                    "type": "string",
                    "enum": ["hour", "day", "week", "month", "year"],
                    "default": "day",
                },
            },
            "required": ["corpus_data"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "geospatial"],
    )

    manager.register_tool(
        category="geospatial_tools",
        name="query_geographic_context",
        func=query_geographic_context,
        description="Run natural language queries over geographic context.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "corpus_data": {"type": "string"},
                "radius_km": {"type": "number", "minimum": 0.000001, "default": 100.0},
                "center_location": {"type": ["string", "null"]},
                "include_related_entities": {"type": "boolean", "default": True},
                "temporal_context": {"type": "boolean", "default": True},
            },
            "required": ["query", "corpus_data"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "geospatial"],
    )
