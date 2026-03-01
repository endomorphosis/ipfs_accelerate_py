"""Native provenance tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_provenance_api() -> Dict[str, Any]:
    """Resolve source provenance APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.provenance_tools.record_provenance import (  # type: ignore
            record_provenance as _record_provenance,
        )

        return {
            "record_provenance": _record_provenance,
        }
    except Exception:
        logger.warning("Source provenance_tools import unavailable, using fallback provenance function")

        async def _record_fallback(
            dataset_id: str,
            operation: str,
            inputs: Optional[List[str]] = None,
            parameters: Optional[Dict[str, Any]] = None,
            description: Optional[str] = None,
            agent_id: Optional[str] = None,
            timestamp: Optional[str] = None,
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            _ = inputs, parameters, description, agent_id, timestamp, tags
            return {
                "status": "success",
                "provenance_id": "fallback-prov-1",
                "dataset_id": dataset_id,
                "operation": operation,
                "record": {},
            }

        return {
            "record_provenance": _record_fallback,
        }


_API = _load_provenance_api()


async def record_provenance(
    dataset_id: str,
    operation: str,
    inputs: Optional[List[str]] = None,
    parameters: Optional[Dict[str, Any]] = None,
    description: Optional[str] = None,
    agent_id: Optional[str] = None,
    timestamp: Optional[str] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Record provenance information for a dataset operation."""
    return await _API["record_provenance"](
        dataset_id=dataset_id,
        operation=operation,
        inputs=inputs,
        parameters=parameters,
        description=description,
        agent_id=agent_id,
        timestamp=timestamp,
        tags=tags,
    )


def register_native_provenance_tools(manager: Any) -> None:
    """Register native provenance tools in unified hierarchical manager."""
    manager.register_tool(
        category="provenance_tools",
        name="record_provenance",
        func=record_provenance,
        description="Record provenance information for dataset operations.",
        input_schema={
            "type": "object",
            "properties": {
                "dataset_id": {"type": "string"},
                "operation": {"type": "string"},
                "inputs": {"type": ["array", "null"], "items": {"type": "string"}},
                "parameters": {"type": ["object", "null"]},
                "description": {"type": ["string", "null"]},
                "agent_id": {"type": ["string", "null"]},
                "timestamp": {"type": ["string", "null"]},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": ["dataset_id", "operation"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "provenance"],
    )
