"""Native provenance tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from datetime import datetime
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

        async def _record_batch_fallback(
            records: List[Dict[str, Any]],
        ) -> Dict[str, Any]:
            _ = records
            return {
                "status": "success",
                "results": [],
                "processed": 0,
            }

        return {
            "record_provenance": _record_fallback,
            "record_provenance_batch": _record_batch_fallback,
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
    if isinstance(inputs, list) and any(not str(item).strip() for item in inputs):
        return {
            "status": "error",
            "message": "inputs cannot contain empty strings",
            "inputs": inputs,
        }
    if parameters is not None and not isinstance(parameters, dict):
        return {
            "status": "error",
            "message": "parameters must be an object when provided",
            "parameters": parameters,
        }
    if description is not None and not str(description).strip():
        return {
            "status": "error",
            "message": "description must be a non-empty string when provided",
            "description": description,
        }
    if agent_id is not None and not str(agent_id).strip():
        return {
            "status": "error",
            "message": "agent_id must be a non-empty string when provided",
            "agent_id": agent_id,
        }
    if tags is not None and (not isinstance(tags, list) or not all(isinstance(item, str) for item in tags)):
        return {
            "status": "error",
            "message": "tags must be an array of strings when provided",
            "tags": tags,
        }
    if isinstance(tags, list) and any(not str(item).strip() for item in tags):
        return {
            "status": "error",
            "message": "tags cannot contain empty strings",
            "tags": tags,
        }
    if timestamp is not None and not str(timestamp).strip():
        return {
            "status": "error",
            "message": "timestamp must be a non-empty string when provided",
            "timestamp": timestamp,
        }
    normalized_timestamp: Optional[str] = None
    if timestamp is not None:
        normalized_timestamp = str(timestamp).strip()
        try:
            datetime.fromisoformat(normalized_timestamp.replace("Z", "+00:00"))
        except ValueError:
            return {
                "status": "error",
                "message": "timestamp must be a valid ISO-8601 string when provided",
                "timestamp": timestamp,
            }

    result = await _API["record_provenance"](
        dataset_id=normalized_dataset_id,
        operation=normalized_operation,
        inputs=inputs,
        parameters=parameters,
        description=description,
        agent_id=agent_id,
        timestamp=normalized_timestamp,
        tags=tags,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("dataset_id", normalized_dataset_id)
    payload.setdefault("operation", normalized_operation)
    return payload


async def record_provenance_batch(
    records: List[Dict[str, Any]],
    fail_fast: bool = False,
) -> Dict[str, Any]:
    """Record provenance for multiple operations with deterministic aggregate output."""
    if not isinstance(records, list) or not records:
        return {
            "status": "error",
            "message": "records must be a non-empty array",
            "results": [],
            "processed": 0,
            "requested": 0,
            "success_count": 0,
            "error_count": 0,
        }

    results: List[Dict[str, Any]] = []
    success_count = 0
    error_count = 0

    for index, record in enumerate(records):
        if not isinstance(record, dict):
            item_result = {
                "status": "error",
                "message": "record entry must be an object",
                "index": index,
            }
        else:
            item_result = await record_provenance(
                dataset_id=str(record.get("dataset_id", "")),
                operation=str(record.get("operation", "")),
                inputs=record.get("inputs"),
                parameters=record.get("parameters"),
                description=record.get("description"),
                agent_id=record.get("agent_id"),
                timestamp=record.get("timestamp"),
                tags=record.get("tags"),
            )
            item_result = dict(item_result)
            item_result.setdefault("index", index)

        if item_result.get("status") == "error":
            error_count += 1
        else:
            success_count += 1

        results.append(item_result)

        if fail_fast and item_result.get("status") == "error":
            break

    return {
        "status": "success",
        "results": results,
        "processed": len(results),
        "requested": len(records),
        "success_count": success_count,
        "error_count": error_count,
        "fail_fast": bool(fail_fast),
    }


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
                "timestamp": {"type": ["string", "null"], "format": "date-time"},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": ["dataset_id", "operation"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "provenance"],
    )

    manager.register_tool(
        category="provenance_tools",
        name="record_provenance_batch",
        func=record_provenance_batch,
        description="Record provenance information for multiple dataset operations.",
        input_schema={
            "type": "object",
            "properties": {
                "records": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "properties": {
                            "dataset_id": {"type": "string", "minLength": 1},
                            "operation": {"type": "string", "minLength": 1},
                            "inputs": {"type": ["array", "null"], "items": {"type": "string"}},
                            "parameters": {"type": ["object", "null"]},
                            "description": {"type": ["string", "null"]},
                            "agent_id": {"type": ["string", "null"]},
                            "timestamp": {"type": ["string", "null"], "format": "date-time"},
                            "tags": {"type": ["array", "null"], "items": {"type": "string"}},
                        },
                        "required": ["dataset_id", "operation"],
                    },
                },
                "fail_fast": {"type": "boolean", "default": False},
            },
            "required": ["records"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "provenance"],
    )
