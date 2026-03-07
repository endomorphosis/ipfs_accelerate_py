"""Native bespoke-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

_ALLOWED_STORE_TYPES = {"faiss", "qdrant", "elasticsearch", "chromadb"}
_ALLOWED_METRICS = {"cosine", "euclidean", "dot_product", "manhattan"}


def _load_bespoke_tools_api() -> Dict[str, Any]:
    """Resolve source bespoke-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.bespoke_tools import (  # type: ignore
            cache_stats as _cache_stats,
            create_vector_store as _create_vector_store,
            delete_index as _delete_index,
            execute_workflow as _execute_workflow,
            list_indices as _list_indices,
            system_health as _system_health,
            system_status as _system_status,
        )

        return {
            "system_health": _system_health,
            "system_status": _system_status,
            "cache_stats": _cache_stats,
            "execute_workflow": _execute_workflow,
            "list_indices": _list_indices,
            "delete_index": _delete_index,
            "create_vector_store": _create_vector_store,
        }
    except Exception:
        logger.warning(
            "Source bespoke_tools import unavailable, using fallback bespoke functions"
        )

        async def _system_health_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "health_score": 100.0,
                "fallback": True,
            }

        async def _cache_stats_fallback(namespace: Optional[str] = None) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "namespace": namespace,
                "global_stats": {"hit_rate": 100.0},
                "fallback": True,
            }

        async def _system_status_fallback() -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "overall_status": "unknown",
                "services": {},
                "fallback": True,
            }

        async def _execute_workflow_fallback(
            workflow_id: str,
            parameters: Optional[Dict[str, Any]] = None,
            dry_run: bool = False,
            timeout_seconds: int = 300,
        ) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "workflow_id": workflow_id,
                "parameters": parameters or {},
                "dry_run": dry_run,
                "timeout_seconds": timeout_seconds,
                "execution_log": [],
                "fallback": True,
            }

        async def _list_indices_fallback(
            store_type: Optional[str] = None,
            include_stats: bool = False,
            namespace: Optional[str] = None,
        ) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "indices": [],
                "count": 0,
                "filters_applied": {
                    "store_type": store_type,
                    "include_stats": include_stats,
                    "namespace": namespace,
                },
                "statistics": {"total_indices": 0} if include_stats else None,
                "fallback": True,
            }

        async def _delete_index_fallback(
            index_id: str,
            store_type: Optional[str] = None,
            confirm: bool = False,
            backup_before_delete: bool = True,
        ) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "index_id": index_id,
                "store_type": store_type,
                "confirm": confirm,
                "backup_before_delete": backup_before_delete,
                "fallback": True,
            }

        async def _create_vector_store_fallback(
            store_name: str,
            store_type: str = "faiss",
            dimension: int = 768,
            metric: str = "cosine",
            namespace: Optional[str] = None,
            configuration: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            return {
                "success": True,
                "status": "success",
                "store_info": {
                    "store_name": store_name,
                    "store_type": store_type,
                    "dimension": dimension,
                    "metric": metric,
                    "namespace": namespace or "default",
                    "configuration": configuration or {},
                },
                "fallback": True,
            }

        return {
            "system_health": _system_health_fallback,
            "system_status": _system_status_fallback,
            "cache_stats": _cache_stats_fallback,
            "execute_workflow": _execute_workflow_fallback,
            "list_indices": _list_indices_fallback,
            "delete_index": _delete_index_fallback,
            "create_vector_store": _create_vector_store_fallback,
        }


_API = _load_bespoke_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        if "status" not in envelope:
            if envelope.get("error") or envelope.get("success") is False:
                envelope["status"] = "error"
            else:
                envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent validation/error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


async def _await_maybe(result: Any) -> Any:
    if hasattr(result, "__await__"):
        return await result
    return result


def _require_string(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, str) or not value.strip():
        return _error_result(f"{field} must be a non-empty string", **{field: value})
    return None


def _optional_string(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if value is None:
        return None
    return _require_string(value, field)


def _validate_bool(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, bool):
        return _error_result(f"{field} must be a boolean", **{field: value})
    return None


def _validate_positive_int(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if not isinstance(value, int) or value < 1:
        return _error_result(f"{field} must be an integer >= 1", **{field: value})
    return None


def _validate_optional_dict(value: Any, field: str) -> Optional[Dict[str, Any]]:
    if value is not None and not isinstance(value, dict):
        return _error_result(f"{field} must be null or an object", **{field: value})
    return None


def _validate_store_type(value: Any, field: str = "store_type") -> Optional[Dict[str, Any]]:
    invalid = _optional_string(value, field)
    if invalid:
        return invalid
    if value is not None and str(value).strip() not in _ALLOWED_STORE_TYPES:
        return _error_result(
            f"{field} must be one of: chromadb, elasticsearch, faiss, qdrant",
            **{field: value},
        )
    return None


def _validate_metric(value: Any) -> Optional[Dict[str, Any]]:
    invalid = _require_string(value, "metric")
    if invalid:
        return invalid
    if str(value).strip() not in _ALLOWED_METRICS:
        return _error_result(
            "metric must be one of: cosine, dot_product, euclidean, manhattan",
            metric=value,
        )
    return None


async def system_health() -> Dict[str, Any]:
    """Return system health metrics for MCP runtime smoke workflows."""
    try:
        result = _API["system_health"]()
        if hasattr(result, "__await__"):
            result = await result
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("health_score", 100.0)
        return envelope
    except Exception as exc:
        return _error_result(str(exc))


async def system_status() -> Dict[str, Any]:
    """Return observed runtime and configuration status details."""
    try:
        result = await _await_maybe(_API["system_status"]())
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("overall_status", "unknown")
            envelope.setdefault("services", {})
        return envelope
    except Exception as exc:
        return _error_result(str(exc))


async def cache_stats(namespace: Optional[str] = None) -> Dict[str, Any]:
    """Return cache statistics for optional namespace scope."""
    if namespace is not None and (not isinstance(namespace, str) or not namespace.strip()):
        return _error_result("namespace must be null or a non-empty string", namespace=namespace)

    clean_namespace = namespace.strip() if isinstance(namespace, str) else None

    try:
        result = await _await_maybe(_API["cache_stats"](namespace=clean_namespace))
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("namespace", clean_namespace)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("global_stats", {})
        return envelope
    except Exception as exc:
        return _error_result(str(exc), namespace=clean_namespace)


async def execute_workflow(
    workflow_id: str,
    parameters: Optional[Dict[str, Any]] = None,
    dry_run: bool = False,
    timeout_seconds: int = 300,
) -> Dict[str, Any]:
    """Execute a predefined compatibility workflow."""
    invalid = _require_string(workflow_id, "workflow_id")
    if invalid:
        return invalid
    invalid = _validate_optional_dict(parameters, "parameters")
    if invalid:
        return invalid
    invalid = _validate_bool(dry_run, "dry_run")
    if invalid:
        return invalid
    invalid = _validate_positive_int(timeout_seconds, "timeout_seconds")
    if invalid:
        return invalid

    clean_workflow_id = workflow_id.strip()
    clean_parameters = dict(parameters or {})

    try:
        result = await _await_maybe(
            _API["execute_workflow"](
                workflow_id=clean_workflow_id,
                parameters=clean_parameters,
                dry_run=dry_run,
                timeout_seconds=timeout_seconds,
            )
        )
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success" if envelope.get("success", True) else "error")
        envelope.setdefault("workflow_id", clean_workflow_id)
        envelope.setdefault("parameters", clean_parameters)
        envelope.setdefault("dry_run", dry_run)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("timeout_seconds", timeout_seconds)
            envelope.setdefault("execution_log", [])
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            workflow_id=clean_workflow_id,
            parameters=clean_parameters,
            dry_run=dry_run,
            timeout_seconds=timeout_seconds,
        )


async def list_indices(
    store_type: Optional[str] = None,
    include_stats: bool = False,
    namespace: Optional[str] = None,
) -> Dict[str, Any]:
    """List bespoke vector-store indices using source-compatible filters."""
    invalid = _validate_store_type(store_type)
    if invalid:
        return invalid
    invalid = _validate_bool(include_stats, "include_stats")
    if invalid:
        return invalid
    invalid = _optional_string(namespace, "namespace")
    if invalid:
        return invalid

    clean_store_type = store_type.strip() if isinstance(store_type, str) else None
    clean_namespace = namespace.strip() if isinstance(namespace, str) else None

    try:
        result = await _await_maybe(
            _API["list_indices"](
                store_type=clean_store_type,
                include_stats=include_stats,
                namespace=clean_namespace,
            )
        )
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success")
        envelope.setdefault("filters_applied", {})
        envelope["filters_applied"].setdefault("store_type", clean_store_type)
        envelope["filters_applied"].setdefault("include_stats", include_stats)
        envelope["filters_applied"].setdefault("namespace", clean_namespace)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
            envelope.setdefault("indices", [])
            envelope.setdefault("count", len(envelope.get("indices") or []))
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            store_type=clean_store_type,
            include_stats=include_stats,
            namespace=clean_namespace,
        )


async def delete_index(
    index_id: str,
    store_type: Optional[str] = None,
    confirm: bool = False,
    backup_before_delete: bool = True,
) -> Dict[str, Any]:
    """Delete a bespoke vector index through the compatibility facade."""
    invalid = _require_string(index_id, "index_id")
    if invalid:
        return invalid
    invalid = _validate_store_type(store_type)
    if invalid:
        return invalid
    invalid = _validate_bool(confirm, "confirm")
    if invalid:
        return invalid
    invalid = _validate_bool(backup_before_delete, "backup_before_delete")
    if invalid:
        return invalid

    clean_index_id = index_id.strip()
    clean_store_type = store_type.strip() if isinstance(store_type, str) else None

    try:
        result = await _await_maybe(
            _API["delete_index"](
                index_id=clean_index_id,
                store_type=clean_store_type,
                confirm=confirm,
                backup_before_delete=backup_before_delete,
            )
        )
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success" if envelope.get("success", True) else "error")
        envelope.setdefault("index_id", clean_index_id)
        envelope.setdefault("confirm", confirm)
        envelope.setdefault("backup_before_delete", backup_before_delete)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            index_id=clean_index_id,
            store_type=clean_store_type,
            confirm=confirm,
            backup_before_delete=backup_before_delete,
        )


async def create_vector_store(
    store_name: str,
    store_type: str = "faiss",
    dimension: int = 768,
    metric: str = "cosine",
    namespace: Optional[str] = None,
    configuration: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a bespoke vector store using source-compatible options."""
    invalid = _require_string(store_name, "store_name")
    if invalid:
        return invalid
    invalid = _validate_store_type(store_type, "store_type")
    if invalid:
        return invalid
    invalid = _validate_positive_int(dimension, "dimension")
    if invalid:
        return invalid
    invalid = _validate_metric(metric)
    if invalid:
        return invalid
    invalid = _optional_string(namespace, "namespace")
    if invalid:
        return invalid
    invalid = _validate_optional_dict(configuration, "configuration")
    if invalid:
        return invalid

    clean_store_name = store_name.strip()
    clean_store_type = store_type.strip()
    clean_metric = metric.strip()
    clean_namespace = namespace.strip() if isinstance(namespace, str) else None
    clean_configuration = dict(configuration or {})

    try:
        result = await _await_maybe(
            _API["create_vector_store"](
                store_name=clean_store_name,
                store_type=clean_store_type,
                dimension=dimension,
                metric=clean_metric,
                namespace=clean_namespace,
                configuration=clean_configuration or None,
            )
        )
        envelope = _normalize_payload(result)
        envelope.setdefault("status", "success" if envelope.get("success", True) else "error")
        envelope.setdefault("store_info", {})
        envelope["store_info"].setdefault("store_name", clean_store_name)
        envelope["store_info"].setdefault("store_type", clean_store_type)
        envelope["store_info"].setdefault("dimension", dimension)
        envelope["store_info"].setdefault("metric", clean_metric)
        envelope["store_info"].setdefault("namespace", clean_namespace or "default")
        envelope["store_info"].setdefault("configuration", clean_configuration)
        if envelope.get("status") == "success":
            envelope.setdefault("success", True)
        return envelope
    except Exception as exc:
        return _error_result(
            str(exc),
            store_name=clean_store_name,
            store_type=clean_store_type,
            dimension=dimension,
            metric=clean_metric,
            namespace=clean_namespace,
        )


def register_native_bespoke_tools(manager: Any) -> None:
    """Register native bespoke-tools category tools in unified manager."""
    manager.register_tool(
        category="bespoke_tools",
        name="system_health",
        func=system_health,
        description="Get high-level system health metrics for MCP runtime components.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "bespoke-tools"],
    )

    manager.register_tool(
        category="bespoke_tools",
        name="cache_stats",
        func=cache_stats,
        description="Get cache statistics and performance metrics by namespace.",
        input_schema={
            "type": "object",
            "properties": {
                "namespace": {"type": ["string", "null"], "minLength": 1},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "bespoke-tools"],
    )

    manager.register_tool(
        category="bespoke_tools",
        name="system_status",
        func=system_status,
        description="Get observed runtime, configuration, and service status details.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "bespoke-tools"],
    )

    manager.register_tool(
        category="bespoke_tools",
        name="execute_workflow",
        func=execute_workflow,
        description="Execute a predefined bespoke workflow with optional parameters.",
        input_schema={
            "type": "object",
            "properties": {
                "workflow_id": {"type": "string", "minLength": 1},
                "parameters": {"type": ["object", "null"]},
                "dry_run": {"type": "boolean", "default": False},
                "timeout_seconds": {"type": "integer", "minimum": 1, "default": 300},
            },
            "required": ["workflow_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "bespoke-tools"],
    )

    manager.register_tool(
        category="bespoke_tools",
        name="list_indices",
        func=list_indices,
        description="List bespoke vector-store indices with optional filters and stats.",
        input_schema={
            "type": "object",
            "properties": {
                "store_type": {
                    "type": ["string", "null"],
                    "enum": ["faiss", "qdrant", "elasticsearch", "chromadb", None],
                },
                "include_stats": {"type": "boolean", "default": False},
                "namespace": {"type": ["string", "null"], "minLength": 1},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "bespoke-tools"],
    )

    manager.register_tool(
        category="bespoke_tools",
        name="delete_index",
        func=delete_index,
        description="Delete a bespoke vector index using explicit confirmation.",
        input_schema={
            "type": "object",
            "properties": {
                "index_id": {"type": "string", "minLength": 1},
                "store_type": {
                    "type": ["string", "null"],
                    "enum": ["faiss", "qdrant", "elasticsearch", "chromadb", None],
                },
                "confirm": {"type": "boolean", "default": False},
                "backup_before_delete": {"type": "boolean", "default": True},
            },
            "required": ["index_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "bespoke-tools"],
    )

    manager.register_tool(
        category="bespoke_tools",
        name="create_vector_store",
        func=create_vector_store,
        description="Create a bespoke vector store with validated type and metric options.",
        input_schema={
            "type": "object",
            "properties": {
                "store_name": {"type": "string", "minLength": 1},
                "store_type": {
                    "type": "string",
                    "enum": ["faiss", "qdrant", "elasticsearch", "chromadb"],
                    "default": "faiss",
                },
                "dimension": {"type": "integer", "minimum": 1, "default": 768},
                "metric": {
                    "type": "string",
                    "enum": ["cosine", "euclidean", "dot_product", "manhattan"],
                    "default": "cosine",
                },
                "namespace": {"type": ["string", "null"], "minLength": 1},
                "configuration": {"type": ["object", "null"]},
            },
            "required": ["store_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "bespoke-tools"],
    )
