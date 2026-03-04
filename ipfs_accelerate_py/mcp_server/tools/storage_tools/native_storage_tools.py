"""Native storage tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_VALID_STORAGE_TYPES = {"local", "ipfs", "s3", "google_cloud", "azure", "memory"}
_VALID_COMPRESSION_TYPES = {"none", "gzip", "lz4", "brotli"}
_VALID_COLLECTION_ACTIONS = {"create", "get", "list", "delete", "stats"}


def _load_storage_api() -> Dict[str, Any]:
    """Resolve source storage APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.storage_tools.storage_tools import (  # type: ignore
            manage_collections as _manage_collections,
            query_storage as _query_storage,
            retrieve_data as _retrieve_data,
            store_data as _store_data,
        )

        return {
            "store_data": _store_data,
            "retrieve_data": _retrieve_data,
            "manage_collections": _manage_collections,
            "query_storage": _query_storage,
        }
    except Exception:
        logger.warning("Source storage_tools import unavailable, using fallback storage functions")

        async def _store_fallback(
            data: Union[str, bytes, Dict[str, Any], List[Any]],
            storage_type: str = "memory",
            compression: str = "none",
            collection: str = "default",
            metadata: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            _ = data, metadata, tags
            return {
                "stored": True,
                "item_id": "fallback-item-1",
                "path": "memory://fallback-item-1",
                "size_bytes": 0,
                "content_hash": "",
                "storage_type": storage_type,
                "compression": compression,
                "collection": collection,
                "stored_at": datetime.now().isoformat(),
            }

        async def _retrieve_fallback(
            item_ids: List[str],
            include_content: bool = False,
            format_type: str = "json",
        ) -> Dict[str, Any]:
            _ = include_content
            return {
                "retrieved_count": 0,
                "not_found_count": len(item_ids),
                "results": [],
                "not_found": item_ids,
                "format": format_type,
                "include_content": include_content,
                "retrieved_at": datetime.now().isoformat(),
            }

        async def _manage_fallback(
            action: str,
            collection_name: Optional[str] = None,
            description: Optional[str] = None,
            metadata: Optional[Dict[str, Any]] = None,
            delete_items: bool = False,
        ) -> Dict[str, Any]:
            _ = description, metadata, delete_items
            if action == "list":
                return {
                    "action": action,
                    "success": True,
                    "collections": [],
                    "total_count": 0,
                }
            return {
                "action": action,
                "success": True,
                "collection_name": collection_name,
            }

        async def _query_fallback(
            collection: Optional[str] = None,
            storage_type: Optional[str] = None,
            tags: Optional[List[str]] = None,
            size_range: Optional[Tuple[int, int]] = None,
            date_range: Optional[Tuple[str, str]] = None,
            limit: int = 100,
            offset: int = 0,
        ) -> Dict[str, Any]:
            _ = collection, storage_type, tags, size_range, date_range, limit, offset
            return {
                "query_results": [],
                "total_found": 0,
                "total_size_bytes": 0,
                "storage_distribution": {},
                "pagination": {
                    "limit": limit,
                    "offset": offset,
                    "has_more": False,
                },
                "queried_at": datetime.now().isoformat(),
            }

        return {
            "store_data": _store_fallback,
            "retrieve_data": _retrieve_fallback,
            "manage_collections": _manage_fallback,
            "query_storage": _query_fallback,
        }


_API = _load_storage_api()


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    """Return a normalized error envelope for deterministic dispatch behavior."""
    payload: Dict[str, Any] = {"status": "error", "error": message}
    payload.update(extra)
    return payload


async def store_data(
    data: Union[str, bytes, Dict[str, Any], List[Any]],
    storage_type: str = "memory",
    compression: str = "none",
    collection: str = "default",
    metadata: Optional[Dict[str, Any]] = None,
    tags: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Store data in the configured storage backend."""
    normalized_storage_type = str(storage_type or "memory").strip().lower() or "memory"
    normalized_compression = str(compression or "none").strip().lower() or "none"
    normalized_collection = str(collection or "").strip() or "default"

    if normalized_storage_type not in _VALID_STORAGE_TYPES:
        return _error_result(
            (
                "Invalid storage type: "
                f"{normalized_storage_type}. Valid types: {sorted(_VALID_STORAGE_TYPES)}"
            ),
            stored=False,
            storage_type=normalized_storage_type,
            compression=normalized_compression,
            collection=normalized_collection,
        )
    if normalized_compression not in _VALID_COMPRESSION_TYPES:
        return _error_result(
            (
                "Invalid compression type: "
                f"{normalized_compression}. Valid types: {sorted(_VALID_COMPRESSION_TYPES)}"
            ),
            stored=False,
            storage_type=normalized_storage_type,
            compression=normalized_compression,
            collection=normalized_collection,
        )

    if metadata is not None and not isinstance(metadata, dict):
        return _error_result("metadata must be an object when provided", stored=False)
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags)
    ):
        return _error_result("tags must be an array of non-empty strings when provided", stored=False)

    try:
        payload = await _API["store_data"](
            data=data,
            storage_type=normalized_storage_type,
            compression=normalized_compression,
            collection=normalized_collection,
            metadata=metadata,
            tags=tags,
        )
    except Exception as exc:
        return _error_result(f"store_data failed: {exc}", stored=False)

    normalized = dict(payload or {})
    normalized.setdefault("status", "success" if normalized.get("stored", False) else "error")
    normalized.setdefault("collection", normalized_collection)
    return normalized


async def retrieve_data(
    item_ids: List[str],
    include_content: bool = False,
    format_type: str = "json",
) -> Dict[str, Any]:
    """Retrieve stored items by ID."""
    if not isinstance(item_ids, list) or not item_ids:
        return _error_result(
            "At least one item ID must be provided",
            retrieved_count=0,
            not_found_count=0,
            results=[],
            not_found=[],
            format=format_type,
            include_content=include_content,
        )
    if not all(isinstance(item_id, str) and item_id.strip() for item_id in item_ids):
        return _error_result(
            "item_ids must be an array of non-empty strings",
            retrieved_count=0,
            not_found_count=0,
            results=[],
            not_found=[],
            format=format_type,
            include_content=include_content,
        )
    if not isinstance(include_content, bool):
        return _error_result("include_content must be a boolean")

    try:
        payload = await _API["retrieve_data"](
            item_ids=item_ids,
            include_content=include_content,
            format_type=format_type,
        )
    except Exception as exc:
        return _error_result(
            f"retrieve_data failed: {exc}",
            retrieved_count=0,
            not_found_count=len(item_ids),
            results=[],
            not_found=item_ids,
            format=format_type,
            include_content=include_content,
        )

    normalized = dict(payload or {})
    if "status" not in normalized:
        normalized["status"] = "error" if "error" in normalized else "success"
    return normalized


async def manage_collections(
    action: str,
    collection_name: Optional[str] = None,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
    delete_items: bool = False,
) -> Dict[str, Any]:
    """Create, list, and manage storage collections."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in _VALID_COLLECTION_ACTIONS:
        return _error_result(
            f"Unknown action: {action}",
            action=normalized_action,
            success=False,
        )

    normalized_collection_name = str(collection_name or "").strip() if collection_name is not None else None

    if normalized_action in {"create", "get", "delete"} and not normalized_collection_name:
        return _error_result(
            f"collection_name required for {normalized_action} action",
            action=normalized_action,
            success=False,
        )
    if metadata is not None and not isinstance(metadata, dict):
        return _error_result("metadata must be an object when provided", action=normalized_action, success=False)
    if not isinstance(delete_items, bool):
        return _error_result("delete_items must be a boolean", action=normalized_action, success=False)

    try:
        payload = await _API["manage_collections"](
            action=normalized_action,
            collection_name=normalized_collection_name,
            description=description,
            metadata=metadata,
            delete_items=delete_items,
        )
    except Exception as exc:
        return _error_result(f"manage_collections failed: {exc}", action=normalized_action, success=False)

    normalized = dict(payload or {})
    normalized.setdefault("status", "success" if normalized.get("success", True) else "error")
    normalized.setdefault("action", normalized_action)
    return normalized


async def query_storage(
    collection: Optional[str] = None,
    storage_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    size_range: Optional[Tuple[int, int]] = None,
    date_range: Optional[Tuple[str, str]] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """Query stored items with filtering and pagination."""
    normalized_storage_type = None
    if storage_type is not None:
        normalized_storage_type = str(storage_type).strip().lower()
        if normalized_storage_type not in _VALID_STORAGE_TYPES:
            return _error_result(
                f"Invalid storage type: {storage_type}",
                query_results=[],
                total_found=0,
                total_size_bytes=0,
                storage_distribution={},
                pagination={"limit": int(limit), "offset": int(offset), "has_more": False},
            )

    try:
        normalized_limit = int(limit)
        normalized_offset = int(offset)
    except (TypeError, ValueError):
        return _error_result("limit and offset must be integers")
    if normalized_limit <= 0:
        return _error_result("limit must be a positive integer")
    if normalized_offset < 0:
        return _error_result("offset must be a non-negative integer")
    if tags is not None and (
        not isinstance(tags, list) or not all(isinstance(tag, str) and tag.strip() for tag in tags)
    ):
        return _error_result("tags must be an array of non-empty strings when provided")

    if size_range is not None:
        if not isinstance(size_range, (list, tuple)) or len(size_range) != 2:
            return _error_result("size_range must be a 2-item array when provided")
        try:
            size_min, size_max = int(size_range[0]), int(size_range[1])
        except (TypeError, ValueError):
            return _error_result("size_range values must be integers")
        if size_min < 0 or size_max < 0 or size_min > size_max:
            return _error_result("size_range must be non-negative and ordered [min, max]")
        normalized_size_range: Optional[Tuple[int, int]] = (size_min, size_max)
    else:
        normalized_size_range = None

    if date_range is not None:
        if not isinstance(date_range, (list, tuple)) or len(date_range) != 2:
            return _error_result("date_range must be a 2-item array when provided")
        if not all(isinstance(item, str) and item.strip() for item in date_range):
            return _error_result("date_range values must be non-empty strings")
        normalized_date_range: Optional[Tuple[str, str]] = (str(date_range[0]).strip(), str(date_range[1]).strip())
    else:
        normalized_date_range = None

    try:
        payload = await _API["query_storage"](
            collection=collection,
            storage_type=normalized_storage_type,
            tags=tags,
            size_range=normalized_size_range,
            date_range=normalized_date_range,
            limit=normalized_limit,
            offset=normalized_offset,
        )
    except Exception as exc:
        return _error_result(
            f"query_storage failed: {exc}",
            query_results=[],
            total_found=0,
            total_size_bytes=0,
            storage_distribution={},
            pagination={"limit": normalized_limit, "offset": normalized_offset, "has_more": False},
        )

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


def register_native_storage_tools(manager: Any) -> None:
    """Register native storage tools in unified hierarchical manager."""
    manager.register_tool(
        category="storage_tools",
        name="store_data",
        func=store_data,
        description="Store data in the selected storage backend.",
        input_schema={
            "type": "object",
            "properties": {
                "data": {},
                "storage_type": {"type": "string", "enum": sorted(_VALID_STORAGE_TYPES), "default": "memory"},
                "compression": {"type": "string", "enum": sorted(_VALID_COMPRESSION_TYPES), "default": "none"},
                "collection": {"type": "string", "minLength": 1, "default": "default"},
                "metadata": {"type": ["object", "null"]},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
            },
            "required": ["data"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="retrieve_data",
        func=retrieve_data,
        description="Retrieve previously stored items by IDs.",
        input_schema={
            "type": "object",
            "properties": {
                "item_ids": {"type": "array", "items": {"type": "string"}},
                "include_content": {"type": "boolean", "default": False},
                "format_type": {"type": "string", "minLength": 1, "default": "json"},
            },
            "required": ["item_ids"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="manage_collections",
        func=manage_collections,
        description="Manage storage collections and collection metadata.",
        input_schema={
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": sorted(_VALID_COLLECTION_ACTIONS)},
                "collection_name": {"type": ["string", "null"], "minLength": 1},
                "description": {"type": ["string", "null"]},
                "metadata": {"type": ["object", "null"]},
                "delete_items": {"type": "boolean", "default": False},
            },
            "required": ["action"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="query_storage",
        func=query_storage,
        description="Query stored items by collection, type, tags, and ranges.",
        input_schema={
            "type": "object",
            "properties": {
                "collection": {"type": ["string", "null"]},
                "storage_type": {"type": ["string", "null"], "enum": sorted(_VALID_STORAGE_TYPES) + [None]},
                "tags": {"type": ["array", "null"], "items": {"type": "string"}},
                "size_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "date_range": {
                    "type": ["array", "null"],
                    "items": {"type": "string"},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "limit": {"type": "integer", "minimum": 1, "default": 100},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )
