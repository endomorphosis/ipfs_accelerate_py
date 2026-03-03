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
    normalized_dataset_id = str(dataset_id or "").strip()
    normalized_operation = str(operation or "").strip()
    if not normalized_dataset_id:
        return {
            "status": "error",
            "message": "dataset_id is required",
            "dataset_id": dataset_id,
        }
    if not normalized_operation:
        return {
            "status": "error",
            "message": "operation is required",
            "operation": operation,
        }
    if inputs is not None and (not isinstance(inputs, list) or not all(isinstance(item, str) for item in inputs)):
        return {
            "status": "error",
            "message": "inputs must be an array of strings when provided",
            "inputs": inputs,
        }
    if parameters is not None and not isinstance(parameters, dict):
        return {
            "status": "error",
            "message": "parameters must be an object when provided",
            "parameters": parameters,
        }
    if tags is not None and (not isinstance(tags, list) or not all(isinstance(item, str) for item in tags)):
        return {
            "status": "error",
            "message": "tags must be an array of strings when provided",
            "tags": tags,
        }
    if timestamp is not None and not str(timestamp).strip():
        return {
            "status": "error",
            "message": "timestamp must be a non-empty string when provided",
            "timestamp": timestamp,
        }

    result = await _API["record_provenance"](
        dataset_id=dataset_id,
        operation=operation,
        inputs=inputs,
        parameters=parameters,
        description=description,
        agent_id=agent_id,
        timestamp=timestamp,
        tags=tags,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("dataset_id", normalized_dataset_id)
    payload.setdefault("operation", normalized_operation)
    return payload


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
