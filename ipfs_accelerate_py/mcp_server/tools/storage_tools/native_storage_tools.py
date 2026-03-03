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

    if normalized_storage_type not in _VALID_STORAGE_TYPES:
        return {
            "stored": False,
            "error": (
                "Invalid storage type: "
                f"{normalized_storage_type}. Valid types: {sorted(_VALID_STORAGE_TYPES)}"
            ),
            "storage_type": normalized_storage_type,
            "compression": normalized_compression,
            "collection": collection,
        }
    if normalized_compression not in _VALID_COMPRESSION_TYPES:
        return {
            "stored": False,
            "error": (
                "Invalid compression type: "
                f"{normalized_compression}. Valid types: {sorted(_VALID_COMPRESSION_TYPES)}"
            ),
            "storage_type": normalized_storage_type,
            "compression": normalized_compression,
            "collection": collection,
        }

    return await _API["store_data"](
        data=data,
        storage_type=normalized_storage_type,
        compression=normalized_compression,
        collection=collection,
        metadata=metadata,
        tags=tags,
    )


async def retrieve_data(
    item_ids: List[str],
    include_content: bool = False,
    format_type: str = "json",
) -> Dict[str, Any]:
    """Retrieve stored items by ID."""
    if not item_ids:
        return {
            "retrieved_count": 0,
            "not_found_count": 0,
            "results": [],
            "not_found": [],
            "format": format_type,
            "include_content": include_content,
            "error": "At least one item ID must be provided",
        }
    return await _API["retrieve_data"](
        item_ids=item_ids,
        include_content=include_content,
        format_type=format_type,
    )


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
        return {
            "action": normalized_action,
            "success": False,
            "error": f"Unknown action: {action}",
        }

    if normalized_action in {"create", "get", "delete"} and not collection_name:
        return {
            "action": normalized_action,
            "success": False,
            "error": f"collection_name required for {normalized_action} action",
        }

    return await _API["manage_collections"](
        action=normalized_action,
        collection_name=collection_name,
        description=description,
        metadata=metadata,
        delete_items=delete_items,
    )


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
            return {
                "query_results": [],
                "total_found": 0,
                "total_size_bytes": 0,
                "storage_distribution": {},
                "pagination": {
                    "limit": int(limit),
                    "offset": int(offset),
                    "has_more": False,
                },
                "error": f"Invalid storage type: {storage_type}",
            }

    return await _API["query_storage"](
        collection=collection,
        storage_type=normalized_storage_type,
        tags=tags,
        size_range=size_range,
        date_range=date_range,
        limit=limit,
        offset=offset,
    )


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
                "collection": {"type": "string"},
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
                "include_content": {"type": "boolean"},
                "format_type": {"type": "string"},
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
                "collection_name": {"type": ["string", "null"]},
                "description": {"type": ["string", "null"]},
                "metadata": {"type": ["object", "null"]},
                "delete_items": {"type": "boolean"},
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
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )
