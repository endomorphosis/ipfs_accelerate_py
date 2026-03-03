"""Native index-management tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


def _load_index_management_api() -> Dict[str, Any]:
    """Resolve source index-management APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.index_management_tools.index_management_tools import (  # type: ignore
            load_index as _load_index,
            manage_shards as _manage_shards,
            monitor_index_status as _monitor_index_status,
            manage_index_configuration as _manage_index_configuration,
        )

        return {
            "load_index": _load_index,
            "manage_shards": _manage_shards,
            "monitor_index_status": _monitor_index_status,
            "manage_index_configuration": _manage_index_configuration,
        }
    except Exception:
        logger.warning("Source index_management_tools import unavailable, using fallback index functions")

        async def _load_index_fallback(
            action: str,
            dataset: Optional[str] = None,
            knn_index: Optional[str] = None,
            dataset_split: str = "train",
            knn_index_split: str = "train",
            columns: str = "text",
            index_config: Optional[Dict[str, Any]] = None,
        ) -> Dict[str, Any]:
            _ = knn_index, dataset_split, knn_index_split, columns, index_config
            return {
                "status": "success",
                "action": action,
                "dataset": dataset,
                "index_id": "fallback-index-1",
            }

        async def _manage_shards_fallback(
            action: str,
            dataset: Optional[str] = None,
            num_shards: int = 4,
            shard_size: str = "auto",
            sharding_strategy: str = "clustering",
            models: Optional[List[str]] = None,
            shard_ids: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            _ = shard_size, sharding_strategy, models, shard_ids
            return {
                "status": "success",
                "action": action,
                "dataset": dataset,
                "num_shards": int(num_shards),
            }

        async def _monitor_status_fallback(
            index_id: Optional[str] = None,
            metrics: Optional[List[str]] = None,
            time_range: str = "24h",
            include_details: bool = False,
        ) -> Dict[str, Any]:
            _ = metrics, include_details
            return {
                "status": "success",
                "index_id": index_id,
                "time_range": time_range,
                "indices": [],
            }

        async def _manage_config_fallback(
            action: str,
            index_id: Optional[str] = None,
            config_updates: Optional[Dict[str, Any]] = None,
            optimization_level: int = 1,
        ) -> Dict[str, Any]:
            return {
                "status": "success",
                "action": action,
                "index_id": index_id,
                "optimization_level": int(optimization_level),
                "config_updates": config_updates or {},
            }

        return {
            "load_index": _load_index_fallback,
            "manage_shards": _manage_shards_fallback,
            "monitor_index_status": _monitor_status_fallback,
            "manage_index_configuration": _manage_config_fallback,
        }


_API = _load_index_management_api()


async def load_index(
    action: str,
    dataset: Optional[str] = None,
    knn_index: Optional[str] = None,
    dataset_split: str = "train",
    knn_index_split: str = "train",
    columns: str = "text",
    index_config: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Load and manage vector indices."""
    normalized_action = str(action or "").strip().lower()
    allowed_actions = {"load", "create", "reload", "unload", "status", "optimize"}
    if normalized_action not in allowed_actions:
        return {
            "status": "error",
            "message": f"action must be one of: {', '.join(sorted(allowed_actions))}",
            "action": action,
        }
    if index_config is not None and not isinstance(index_config, dict):
        return {
            "status": "error",
            "message": "index_config must be an object when provided",
            "index_config": index_config,
        }
    if dataset is not None and not str(dataset).strip():
        return {
            "status": "error",
            "message": "dataset must be a non-empty string when provided",
            "dataset": dataset,
        }

    result = await _API["load_index"](
        action=action,
        dataset=dataset,
        knn_index=knn_index,
        dataset_split=dataset_split,
        knn_index_split=knn_index_split,
        columns=columns,
        index_config=index_config,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("action", normalized_action)
    return payload


async def manage_shards(
    action: str,
    dataset: Optional[str] = None,
    num_shards: int = 4,
    shard_size: str = "auto",
    sharding_strategy: str = "clustering",
    models: Optional[List[str]] = None,
    shard_ids: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Manage index sharding operations."""
    normalized_action = str(action or "").strip().lower()
    allowed_actions = {"create_shards", "list_shards", "rebalance", "merge_shards", "status", "distribute"}
    if normalized_action not in allowed_actions:
        return {
            "status": "error",
            "message": f"action must be one of: {', '.join(sorted(allowed_actions))}",
            "action": action,
        }
    normalized_num_shards = int(num_shards)
    if normalized_num_shards <= 0:
        return {
            "status": "error",
            "message": "num_shards must be a positive integer",
            "num_shards": num_shards,
        }
    if models is not None and (not isinstance(models, list) or not all(isinstance(item, str) for item in models)):
        return {
            "status": "error",
            "message": "models must be an array of strings when provided",
            "models": models,
        }
    if shard_ids is not None and (not isinstance(shard_ids, list) or not all(isinstance(item, str) for item in shard_ids)):
        return {
            "status": "error",
            "message": "shard_ids must be an array of strings when provided",
            "shard_ids": shard_ids,
        }

    result = await _API["manage_shards"](
        action=action,
        dataset=dataset,
        num_shards=normalized_num_shards,
        shard_size=shard_size,
        sharding_strategy=sharding_strategy,
        models=models,
        shard_ids=shard_ids,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("action", normalized_action)
    return payload


async def monitor_index_status(
    index_id: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    time_range: str = "24h",
    include_details: bool = False,
) -> Dict[str, Any]:
    """Monitor index health and performance state."""
    if metrics is not None and (not isinstance(metrics, list) or not all(isinstance(item, str) for item in metrics)):
        return {
            "status": "error",
            "message": "metrics must be an array of strings when provided",
            "metrics": metrics,
        }
    normalized_time_range = str(time_range or "").strip().lower()
    allowed_time_ranges = {"1h", "6h", "24h", "7d", "30d"}
    if normalized_time_range not in allowed_time_ranges:
        return {
            "status": "error",
            "message": "time_range must be one of: 1h, 6h, 24h, 7d, 30d",
            "time_range": time_range,
        }
    if not isinstance(include_details, bool):
        return {
            "status": "error",
            "message": "include_details must be a boolean",
            "include_details": include_details,
        }

    result = await _API["monitor_index_status"](
        index_id=index_id,
        metrics=metrics,
        time_range=normalized_time_range,
        include_details=include_details,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("time_range", normalized_time_range)
    return payload


async def manage_index_configuration(
    action: str,
    index_id: Optional[str] = None,
    config_updates: Optional[Dict[str, Any]] = None,
    optimization_level: int = 1,
) -> Dict[str, Any]:
    """Manage index configuration and optimization settings."""
    normalized_action = str(action or "").strip().lower()
    allowed_actions = {"get_config", "update_config", "optimize_config", "reset_config"}
    if normalized_action not in allowed_actions:
        return {
            "status": "error",
            "message": f"action must be one of: {', '.join(sorted(allowed_actions))}",
            "action": action,
        }
    if config_updates is not None and not isinstance(config_updates, dict):
        return {
            "status": "error",
            "message": "config_updates must be an object when provided",
            "config_updates": config_updates,
        }
    normalized_optimization_level = int(optimization_level)
    if normalized_optimization_level < 1 or normalized_optimization_level > 3:
        return {
            "status": "error",
            "message": "optimization_level must be between 1 and 3",
            "optimization_level": optimization_level,
        }

    result = await _API["manage_index_configuration"](
        action=action,
        index_id=index_id,
        config_updates=config_updates,
        optimization_level=normalized_optimization_level,
    )
    payload = dict(result or {})
    payload.setdefault("status", "success")
    payload.setdefault("action", normalized_action)
    payload.setdefault("optimization_level", normalized_optimization_level)
    return payload


def register_native_index_management_tools(manager: Any) -> None:
    """Register native index-management tools in unified hierarchical manager."""
    manager.register_tool(
        category="index_management_tools",
        name="load_index",
        func=load_index,
        description="Load and manage vector indices.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "dataset": {"type": ["string", "null"]},
                "knn_index": {"type": ["string", "null"]},
                "dataset_split": {"type": "string"},
                "knn_index_split": {"type": "string"},
                "columns": {"type": "string"},
                "index_config": {"type": ["object", "null"]},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "indexing"],
    )

    manager.register_tool(
        category="index_management_tools",
        name="manage_shards",
        func=manage_shards,
        description="Manage index sharding operations.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "dataset": {"type": ["string", "null"]},
                "num_shards": {"type": "integer"},
                "shard_size": {"type": "string"},
                "sharding_strategy": {"type": "string"},
                "models": {"type": ["array", "null"], "items": {"type": "string"}},
                "shard_ids": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "indexing"],
    )

    manager.register_tool(
        category="index_management_tools",
        name="monitor_index_status",
        func=monitor_index_status,
        description="Monitor index health and performance state.",
        input_schema={
            "type": "object",
            "properties": {
                "index_id": {"type": ["string", "null"]},
                "metrics": {"type": ["array", "null"], "items": {"type": "string"}},
                "time_range": {"type": "string"},
                "include_details": {"type": "boolean"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "indexing"],
    )

    manager.register_tool(
        category="index_management_tools",
        name="manage_index_configuration",
        func=manage_index_configuration,
        description="Manage index configuration and optimization settings.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string"},
                "index_id": {"type": ["string", "null"]},
                "config_updates": {"type": ["object", "null"]},
                "optimization_level": {"type": "integer"},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "indexing"],
    )
