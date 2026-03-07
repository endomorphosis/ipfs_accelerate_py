"""Native storage tool implementations for unified mcp_server."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union

logger = logging.getLogger(__name__)

_VALID_STORAGE_TYPES = {"local", "ipfs", "s3", "google_cloud", "azure", "memory"}
_VALID_COMPRESSION_TYPES = {"none", "gzip", "lz4", "brotli"}
_VALID_COLLECTION_ACTIONS = {
    "create",
    "get",
    "list",
    "delete",
    "stats",
    "backend_status",
    "lifecycle_report",
}
_VALID_REPORT_FORMATS = {"summary", "detailed", "analytics"}


def _extract_stats_totals(stats_payload: Dict[str, Any]) -> Dict[str, int]:
    """Normalize total counters from either direct or nested basic-stats payloads."""
    if not isinstance(stats_payload, dict):
        return {"total_items": 0, "total_size_bytes": 0}

    basic_stats = stats_payload.get("basic_stats")
    stats_root = basic_stats if isinstance(basic_stats, dict) else stats_payload

    return {
        "total_items": int(stats_root.get("total_items", 0) or 0),
        "total_size_bytes": int(stats_root.get("total_size_bytes", 0) or 0),
    }


def _extract_storage_distribution(stats_payload: Dict[str, Any]) -> Dict[str, int]:
    """Normalize storage distribution map from direct or nested storage stats."""
    if not isinstance(stats_payload, dict):
        return {}

    basic_stats = stats_payload.get("basic_stats")
    stats_root = basic_stats if isinstance(basic_stats, dict) else stats_payload
    storage_types = stats_root.get("storage_types")
    if not isinstance(storage_types, dict):
        return {}

    normalized: Dict[str, int] = {}
    for key, value in storage_types.items():
        try:
            normalized[str(key)] = int(value)
        except (TypeError, ValueError):
            normalized[str(key)] = 0
    return normalized


_VALID_BACKEND_AVAILABILITY_FILTERS = {"all", "available", "unavailable"}


def _load_storage_api() -> Dict[str, Any]:
    """Resolve source storage APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.storage_tools.storage_tools import (  # type: ignore
            _storage_manager as _source_storage_manager,
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
            "storage_manager": _source_storage_manager,
        }
    except Exception:
        logger.warning("Source storage_tools import unavailable, using fallback storage functions")

        fallback_manager = None
        fallback_storage_enum = None
        fallback_compression_enum = None
        try:
            from ipfs_datasets_py.ipfs_datasets_py.storage.storage_engine import (  # type: ignore
                CompressionType as _FallbackCompressionType,
                MockStorageManager as _FallbackMockStorageManager,
                StorageType as _FallbackStorageType,
            )

            fallback_manager = _FallbackMockStorageManager()
            fallback_storage_enum = _FallbackStorageType
            fallback_compression_enum = _FallbackCompressionType
        except Exception:
            fallback_manager = None
            fallback_storage_enum = None
            fallback_compression_enum = None

        async def _store_fallback(
            data: Union[str, bytes, Dict[str, Any], List[Any]],
            storage_type: str = "memory",
            compression: str = "none",
            collection: str = "default",
            metadata: Optional[Dict[str, Any]] = None,
            tags: Optional[List[str]] = None,
        ) -> Dict[str, Any]:
            if (
                fallback_manager is not None
                and fallback_storage_enum is not None
                and fallback_compression_enum is not None
            ):
                stored_item = fallback_manager.store_item(
                    content=data,
                    storage_type=fallback_storage_enum(storage_type),
                    compression=fallback_compression_enum(compression),
                    metadata=metadata,
                    tags=tags,
                    collection_name=collection,
                )
                return {
                    "stored": True,
                    "item_id": stored_item.id,
                    "path": stored_item.path,
                    "size_bytes": stored_item.size_bytes,
                    "content_hash": stored_item.content_hash,
                    "storage_type": stored_item.storage_type.value,
                    "compression": stored_item.compression.value,
                    "collection": collection,
                    "stored_at": stored_item.created_at.isoformat(),
                }

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
            if fallback_manager is not None:
                results = []
                not_found = []
                for item_id in item_ids:
                    retrieved = fallback_manager.retrieve_item(item_id, include_content=include_content)
                    if retrieved is None:
                        not_found.append(item_id)
                    else:
                        results.append(retrieved)
                return {
                    "retrieved_count": len(results),
                    "not_found_count": len(not_found),
                    "results": results,
                    "not_found": not_found,
                    "format": format_type,
                    "include_content": include_content,
                    "retrieved_at": datetime.now().isoformat(),
                }

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
            if fallback_manager is not None:
                if action == "create":
                    created = fallback_manager.create_collection(
                        name=str(collection_name),
                        description=description or "",
                        metadata=metadata,
                    )
                    return {"action": action, "success": True, "collection": created}
                if action == "get":
                    collection = fallback_manager.get_collection(str(collection_name))
                    if collection is None:
                        return {"action": action, "success": False, "error": f"Collection '{collection_name}' not found"}
                    return {"action": action, "success": True, "collection": collection}
                if action == "list":
                    collections = fallback_manager.list_collections()
                    return {
                        "action": action,
                        "success": True,
                        "collections": collections,
                        "total_count": len(collections),
                    }
                if action == "delete":
                    deleted = fallback_manager.delete_collection(str(collection_name), delete_items)
                    return {
                        "action": action,
                        "success": deleted,
                        "collection_name": collection_name,
                        "items_deleted": delete_items,
                    }
                if action == "stats":
                    if collection_name:
                        collection = fallback_manager.get_collection(str(collection_name))
                        if collection is None:
                            return {"action": action, "success": False, "error": f"Collection '{collection_name}' not found"}
                        return {"action": action, "success": True, "collection_stats": collection}
                    return {
                        "action": action,
                        "success": True,
                        "global_stats": fallback_manager.get_storage_stats(),
                    }

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
            if fallback_manager is not None:
                items = fallback_manager.list_items(
                    collection_name=collection,
                    storage_type=(fallback_storage_enum(storage_type) if storage_type and fallback_storage_enum is not None else None),
                    tags=tags,
                    limit=max(limit * 2, limit),
                    offset=offset,
                )

                filtered_items = items
                if size_range is not None:
                    size_min, size_max = size_range
                    filtered_items = [item for item in filtered_items if size_min <= int(item.get("size_bytes", 0)) <= size_max]
                if date_range is not None:
                    start_raw, end_raw = date_range
                    start_dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
                    end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
                    filtered_items = [
                        item for item in filtered_items
                        if start_dt <= datetime.fromisoformat(str(item.get("created_at"))) <= end_dt
                    ]

                filtered_items = filtered_items[:limit]
                storage_distribution: Dict[str, int] = {}
                for item in filtered_items:
                    item_storage_type = str(item.get("storage_type", ""))
                    storage_distribution[item_storage_type] = storage_distribution.get(item_storage_type, 0) + 1

                return {
                    "query_results": filtered_items,
                    "total_found": len(filtered_items),
                    "total_size_bytes": sum(int(item.get("size_bytes", 0)) for item in filtered_items),
                    "storage_distribution": storage_distribution,
                    "pagination": {
                        "limit": limit,
                        "offset": offset,
                        "has_more": len(items) > len(filtered_items),
                    },
                    "queried_at": datetime.now().isoformat(),
                }

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
            "storage_manager": fallback_manager,
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
    if collection is None:
        normalized_collection = "default"
    elif not isinstance(collection, str) or not collection.strip():
        return _error_result("collection must be a non-empty string when provided", stored=False)
    else:
        normalized_collection = collection.strip()

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
    if not isinstance(format_type, str) or not format_type.strip():
        return _error_result(
            "format_type must be a non-empty string",
            retrieved_count=0,
            not_found_count=0,
            results=[],
            not_found=[],
            format=format_type,
            include_content=include_content,
        )

    try:
        payload = await _API["retrieve_data"](
            item_ids=item_ids,
            include_content=include_content,
            format_type=format_type.strip(),
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
    include_breakdown: bool = False,
    include_capabilities: bool = False,
    backend_types: Optional[List[str]] = None,
    unavailable_backends: Optional[List[str]] = None,
    unavailable_reasons: Optional[Dict[str, str]] = None,
    availability_filter: str = "all",
    report_format: str = "detailed",
) -> Dict[str, Any]:
    """Create, list, and manage storage collections."""
    normalized_action = str(action or "").strip().lower()
    if normalized_action not in _VALID_COLLECTION_ACTIONS:
        return _error_result(
            f"Unknown action: {action}",
            action=normalized_action,
            success=False,
        )

    if collection_name is None:
        normalized_collection_name = None
    elif not isinstance(collection_name, str) or not collection_name.strip():
        return _error_result(
            "collection_name must be a non-empty string when provided",
            action=normalized_action,
            success=False,
        )
    else:
        normalized_collection_name = collection_name.strip()

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
    if not isinstance(include_breakdown, bool):
        return _error_result("include_breakdown must be a boolean", action=normalized_action, success=False)
    if not isinstance(include_capabilities, bool):
        return _error_result("include_capabilities must be a boolean", action=normalized_action, success=False)
    if backend_types is not None and (
        not isinstance(backend_types, list)
        or not all(isinstance(item, str) and item.strip() for item in backend_types)
    ):
        return _error_result("backend_types must be an array of non-empty strings", action=normalized_action, success=False)
    if unavailable_backends is not None and (
        not isinstance(unavailable_backends, list)
        or not all(isinstance(item, str) and item.strip() for item in unavailable_backends)
    ):
        return _error_result(
            "unavailable_backends must be an array of non-empty strings",
            action=normalized_action,
            success=False,
        )
    if unavailable_reasons is not None and (
        not isinstance(unavailable_reasons, dict)
        or not all(
            isinstance(key, str)
            and key.strip()
            and isinstance(value, str)
            and value.strip()
            for key, value in unavailable_reasons.items()
        )
    ):
        return _error_result(
            "unavailable_reasons must be an object with non-empty string keys/values",
            action=normalized_action,
            success=False,
        )
    normalized_availability_filter = str(availability_filter or "all").strip().lower() or "all"
    if normalized_availability_filter not in _VALID_BACKEND_AVAILABILITY_FILTERS:
        return _error_result(
            (
                "availability_filter must be one of: "
                f"{sorted(_VALID_BACKEND_AVAILABILITY_FILTERS)}"
            ),
            action=normalized_action,
            success=False,
        )

    normalized_report_format = str(report_format or "detailed").strip().lower()
    if normalized_action == "stats" and normalized_report_format not in _VALID_REPORT_FORMATS:
        return _error_result(
            (
                "report_format must be one of: "
                f"{sorted(_VALID_REPORT_FORMATS)}"
            ),
            action=normalized_action,
            success=False,
        )

    if normalized_action == "backend_status":
        requested_backends = (
            [str(item).strip().lower() for item in backend_types]
            if backend_types is not None
            else sorted(_VALID_STORAGE_TYPES)
        )
        invalid_requested = sorted({name for name in requested_backends if name not in _VALID_STORAGE_TYPES})
        if invalid_requested:
            return _error_result(
                "backend_types contains unknown storage backends",
                action=normalized_action,
                success=False,
                invalid_backends=invalid_requested,
            )

        unavailable_set = (
            {str(item).strip().lower() for item in unavailable_backends}
            if unavailable_backends is not None
            else set()
        )
        invalid_unavailable = sorted({name for name in unavailable_set if name not in _VALID_STORAGE_TYPES})
        if invalid_unavailable:
            return _error_result(
                "unavailable_backends contains unknown storage backends",
                action=normalized_action,
                success=False,
                invalid_backends=invalid_unavailable,
            )

        normalized_unavailable_reasons = (
            {str(key).strip().lower(): str(value).strip() for key, value in unavailable_reasons.items()}
            if unavailable_reasons is not None
            else {}
        )
        invalid_reason_backends = sorted(
            {
                backend
                for backend in normalized_unavailable_reasons
                if backend not in _VALID_STORAGE_TYPES
            }
        )
        if invalid_reason_backends:
            return _error_result(
                "unavailable_reasons contains unknown storage backends",
                action=normalized_action,
                success=False,
                invalid_backends=invalid_reason_backends,
            )

        backend_entries: List[Dict[str, Any]] = []
        for storage_name in requested_backends:
            entry: Dict[str, Any] = {
                "storage_type": storage_name,
                "available": storage_name not in unavailable_set,
                "supports_collections": True,
            }
            if not entry["available"] and storage_name in normalized_unavailable_reasons:
                entry["unavailable_reason"] = normalized_unavailable_reasons[storage_name]
            if include_capabilities:
                entry["capabilities"] = {
                    "supports_compression": storage_name != "memory",
                    "supports_tags": True,
                    "supports_metadata": True,
                    "supports_query_filters": True,
                }
            backend_entries.append(entry)

        if normalized_availability_filter == "available":
            backend_entries = [entry for entry in backend_entries if entry.get("available")]
        elif normalized_availability_filter == "unavailable":
            backend_entries = [entry for entry in backend_entries if not entry.get("available")]

        result: Dict[str, Any] = {
            "status": "success",
            "action": normalized_action,
            "success": True,
            "backend_report": {
                "generated_at": datetime.now().isoformat(),
                "availability_filter": normalized_availability_filter,
                "backend_count": len(backend_entries),
                "backends": backend_entries,
            },
        }

        if include_breakdown:
            available_count = len([entry for entry in backend_entries if entry.get("available")])
            result["backend_report"]["breakdown"] = {
                "available_count": available_count,
                "unavailable_count": len(backend_entries) - available_count,
            }

        return result

    if normalized_action == "lifecycle_report":
        list_payload = await manage_collections(action="list")
        if list_payload.get("status") == "error":
            return list_payload

        collections = list_payload.get("collections")
        if not isinstance(collections, list):
            collections = []

        stats_payload = await manage_collections(
            action="stats",
            collection_name=normalized_collection_name,
            include_breakdown=include_breakdown,
            report_format=report_format,
        )
        if stats_payload.get("status") == "error":
            return stats_payload

        collection_names = [
            str((entry or {}).get("name", "")).strip()
            for entry in collections
            if isinstance(entry, dict)
        ]
        collection_names = [name for name in collection_names if name]

        storage_report = stats_payload.get("storage_report")
        if not isinstance(storage_report, dict):
            storage_report = {}

        summary = storage_report.get("summary")
        if not isinstance(summary, dict):
            summary = {}
        totals = {
            "total_items": int(summary.get("total_items", 0) or 0),
            "total_size_bytes": int(summary.get("total_size_bytes", 0) or 0),
        }

        lifecycle_report: Dict[str, Any] = {
            "generated_at": datetime.now().isoformat(),
            "scope": "collection" if normalized_collection_name else "global",
            "collection_name": normalized_collection_name,
            "collections_total": int(list_payload.get("total_count", len(collections)) or 0),
            "collection_names": collection_names,
            "totals": totals,
            "stats_report_format": str(storage_report.get("report_format") or report_format),
        }

        if include_breakdown:
            lifecycle_report["collections"] = collections
            breakdown = storage_report.get("breakdown")
            if isinstance(breakdown, dict):
                lifecycle_report["breakdown"] = breakdown

        if normalized_report_format == "analytics":
            analytics = storage_report.get("analytics")
            if isinstance(analytics, dict):
                lifecycle_report["analytics"] = analytics

        return {
            "status": "success",
            "action": normalized_action,
            "success": True,
            "lifecycle_report": lifecycle_report,
        }

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

    if normalized_action == "stats" and normalized.get("status") == "success":
        report_payload: Dict[str, Any] = {
            "report_format": normalized_report_format,
            "generated_at": datetime.now().isoformat(),
        }

        if normalized_report_format == "summary":
            global_stats = normalized.get("global_stats") or {}
            collection_stats = normalized.get("collection_stats") or {}
            if isinstance(global_stats, dict) and global_stats:
                report_payload["summary"] = _extract_stats_totals(global_stats)
            elif isinstance(collection_stats, dict):
                report_payload["summary"] = {
                    "total_items": int(collection_stats.get("items_count", 0) or 0),
                    "total_size_bytes": int(collection_stats.get("total_size_bytes", 0) or 0),
                }
            else:
                report_payload["summary"] = {"total_items": 0, "total_size_bytes": 0}
        elif normalized_report_format == "analytics":
            global_stats = normalized.get("global_stats") or {}
            collection_stats = normalized.get("collection_stats") or {}

            if isinstance(global_stats, dict) and global_stats:
                totals = _extract_stats_totals(global_stats)
                storage_distribution = _extract_storage_distribution(global_stats)
                average_item_size_bytes = float(global_stats.get("average_item_size_bytes", 0.0) or 0.0)
                compression_usage_ratios = global_stats.get("compression_usage_ratios")

                report_payload["analytics"] = {
                    "scope": "global",
                    "totals": totals,
                    "average_item_size_bytes": average_item_size_bytes,
                    "largest_collection": str(global_stats.get("largest_collection", "none")),
                    "storage_distribution": storage_distribution,
                    "compression_usage_ratios": (
                        compression_usage_ratios if isinstance(compression_usage_ratios, dict) else {}
                    ),
                }
            elif isinstance(collection_stats, dict):
                report_payload["analytics"] = {
                    "scope": "collection",
                    "collection_name": str(collection_stats.get("name", normalized_collection_name or "")),
                    "totals": {
                        "total_items": int(collection_stats.get("items_count", 0) or 0),
                        "total_size_bytes": int(collection_stats.get("total_size_bytes", 0) or 0),
                    },
                    "storage_distribution": (
                        collection_stats.get("storage_breakdown")
                        if isinstance(collection_stats.get("storage_breakdown"), dict)
                        else {}
                    ),
                }
            else:
                report_payload["analytics"] = {
                    "scope": "unknown",
                    "totals": {"total_items": 0, "total_size_bytes": 0},
                    "storage_distribution": {},
                }
        else:
            report_payload["details"] = {
                "collection_name": normalized_collection_name,
                "has_global_stats": isinstance(normalized.get("global_stats"), dict),
                "has_collection_stats": isinstance(normalized.get("collection_stats"), dict),
            }

        if include_breakdown:
            global_stats = normalized.get("global_stats")
            report_payload["breakdown"] = {
                "storage_distribution": (
                    _extract_storage_distribution(global_stats)
                    if isinstance(global_stats, dict)
                    else {}
                ),
            }

        normalized["storage_report"] = report_payload

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

        start_raw = str(date_range[0]).strip()
        end_raw = str(date_range[1]).strip()
        try:
            start_dt = datetime.fromisoformat(start_raw.replace("Z", "+00:00"))
            end_dt = datetime.fromisoformat(end_raw.replace("Z", "+00:00"))
        except ValueError:
            return _error_result("date_range values must be valid ISO-8601 datetime strings")
        if start_dt > end_dt:
            return _error_result("date_range must be ordered as [start, end]")

        normalized_date_range: Optional[Tuple[str, str]] = (start_raw, end_raw)
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


async def list_storage(
    collection: Optional[str] = None,
    storage_type: Optional[str] = None,
    tags: Optional[List[str]] = None,
    limit: int = 100,
    offset: int = 0,
) -> Dict[str, Any]:
    """List stored items using the query_storage filtering surface."""
    result = await query_storage(
        collection=collection,
        storage_type=storage_type,
        tags=tags,
        limit=limit,
        offset=offset,
    )
    if result.get("status") == "error":
        return result

    objects = result.get("query_results")
    if not isinstance(objects, list):
        objects = []

    normalized_objects: List[Dict[str, Any]] = []
    for item in objects:
        if not isinstance(item, dict):
            continue
        normalized_objects.append(
            {
                "item_id": item.get("id"),
                "path": item.get("path"),
                "size_bytes": item.get("size_bytes", 0),
                "storage_type": item.get("storage_type"),
                "last_modified": item.get("created_at"),
                "tags": item.get("tags", []),
                "metadata": item.get("metadata", {}),
            }
        )

    return {
        "status": "success",
        "objects": normalized_objects,
        "total_count": int(result.get("total_found", len(normalized_objects)) or 0),
        "storage_distribution": result.get("storage_distribution", {}),
        "pagination": result.get(
            "pagination",
            {"limit": limit, "offset": offset, "has_more": False},
        ),
    }


async def get_storage_stats(
    collection_name: Optional[str] = None,
    report_format: str = "summary",
    include_breakdown: bool = False,
) -> Dict[str, Any]:
    """Return normalized storage statistics via manage_collections(stats)."""
    if collection_name is not None and (not isinstance(collection_name, str) or not collection_name.strip()):
        return _error_result("collection_name must be a non-empty string when provided")

    result = await manage_collections(
        action="stats",
        collection_name=collection_name,
        report_format=report_format,
        include_breakdown=include_breakdown,
    )
    if result.get("status") == "error":
        return result

    storage_report = result.get("storage_report")
    if not isinstance(storage_report, dict):
        storage_report = {}
    summary = storage_report.get("summary")
    if not isinstance(summary, dict):
        summary = {}
    analytics = storage_report.get("analytics")
    if not isinstance(analytics, dict):
        analytics = {}
    details = storage_report.get("details")
    if not isinstance(details, dict):
        details = {}
    breakdown = storage_report.get("breakdown")
    if not isinstance(breakdown, dict):
        breakdown = {}

    totals = {
        "total_objects": int(summary.get("total_items", analytics.get("totals", {}).get("total_items", 0)) or 0),
        "total_bytes": int(summary.get("total_size_bytes", analytics.get("totals", {}).get("total_size_bytes", 0)) or 0),
    }

    return {
        "status": "success",
        "collection_name": collection_name,
        "report_format": storage_report.get("report_format", report_format),
        "backends": breakdown.get("storage_distribution", analytics.get("storage_distribution", {})),
        **totals,
        "details": details,
        "analytics": analytics,
        "breakdown": breakdown if include_breakdown else {},
        "generated_at": storage_report.get("generated_at"),
    }


async def get_storage_collection_stats(
    collection_name: str,
    report_format: str = "summary",
    include_breakdown: bool = False,
) -> Dict[str, Any]:
    """Return normalized collection-scoped storage statistics."""
    normalized_collection_name = str(collection_name or "").strip()
    if not normalized_collection_name:
        return _error_result("collection_name is required")
    if not isinstance(include_breakdown, bool):
        return _error_result("include_breakdown must be a boolean")

    normalized_report_format = str(report_format or "summary").strip().lower()
    if normalized_report_format not in _VALID_REPORT_FORMATS:
        return _error_result(
            (
                "report_format must be one of: "
                f"{sorted(_VALID_REPORT_FORMATS)}"
            )
        )

    result = await get_storage_stats(
        collection_name=normalized_collection_name,
        report_format=normalized_report_format,
        include_breakdown=include_breakdown,
    )
    if result.get("status") == "error":
        return result

    return {
        "status": "success",
        "scope": "collection",
        "collection_name": normalized_collection_name,
        "report_format": result.get("report_format", normalized_report_format),
        "total_objects": int(result.get("total_objects", 0) or 0),
        "total_bytes": int(result.get("total_bytes", 0) or 0),
        "backends": result.get("backends", {}),
        "details": result.get("details", {}),
        "analytics": result.get("analytics", {}),
        "breakdown": result.get("breakdown", {}),
        "generated_at": result.get("generated_at"),
    }


async def get_storage_lifecycle_report(
    collection_name: Optional[str] = None,
    report_format: str = "detailed",
    include_breakdown: bool = False,
) -> Dict[str, Any]:
    """Return normalized storage lifecycle telemetry via manage_collections(lifecycle_report)."""
    if collection_name is not None and (not isinstance(collection_name, str) or not collection_name.strip()):
        return _error_result("collection_name must be a non-empty string when provided")
    if not isinstance(include_breakdown, bool):
        return _error_result("include_breakdown must be a boolean")

    normalized_report_format = str(report_format or "detailed").strip().lower()
    if normalized_report_format not in _VALID_REPORT_FORMATS:
        return _error_result(
            (
                "report_format must be one of: "
                f"{sorted(_VALID_REPORT_FORMATS)}"
            )
        )

    result = await manage_collections(
        action="lifecycle_report",
        collection_name=collection_name,
        report_format=normalized_report_format,
        include_breakdown=include_breakdown,
    )
    if result.get("status") == "error":
        return result

    lifecycle_report = result.get("lifecycle_report")
    if not isinstance(lifecycle_report, dict):
        lifecycle_report = {}

    totals = lifecycle_report.get("totals")
    if not isinstance(totals, dict):
        totals = {"total_items": 0, "total_size_bytes": 0}

    collection_names = lifecycle_report.get("collection_names")
    if not isinstance(collection_names, list):
        collection_names = []

    return {
        "status": "success",
        "collection_name": collection_name,
        "report_format": lifecycle_report.get("stats_report_format", normalized_report_format),
        "scope": lifecycle_report.get("scope", "global"),
        "collections_total": int(lifecycle_report.get("collections_total", len(collection_names)) or 0),
        "collection_names": collection_names,
        "totals": {
            "total_items": int(totals.get("total_items", 0) or 0),
            "total_size_bytes": int(totals.get("total_size_bytes", 0) or 0),
        },
        "lifecycle_report": lifecycle_report,
        "generated_at": lifecycle_report.get("generated_at"),
    }


async def get_storage_backend_status(
    include_capabilities: bool = False,
    backend_types: Optional[List[str]] = None,
    unavailable_backends: Optional[List[str]] = None,
    unavailable_reasons: Optional[Dict[str, str]] = None,
    availability_filter: str = "all",
    include_breakdown: bool = False,
) -> Dict[str, Any]:
    """Return normalized backend availability telemetry via manage_collections(backend_status)."""
    if not isinstance(include_capabilities, bool):
        return _error_result("include_capabilities must be a boolean")
    if not isinstance(include_breakdown, bool):
        return _error_result("include_breakdown must be a boolean")
    if backend_types is not None and (
        not isinstance(backend_types, list)
        or not all(isinstance(item, str) and item.strip() for item in backend_types)
    ):
        return _error_result("backend_types must be an array of non-empty strings")
    if unavailable_backends is not None and (
        not isinstance(unavailable_backends, list)
        or not all(isinstance(item, str) and item.strip() for item in unavailable_backends)
    ):
        return _error_result("unavailable_backends must be an array of non-empty strings")
    if unavailable_reasons is not None and (
        not isinstance(unavailable_reasons, dict)
        or not all(
            isinstance(key, str)
            and key.strip()
            and isinstance(value, str)
            and value.strip()
            for key, value in unavailable_reasons.items()
        )
    ):
        return _error_result("unavailable_reasons must be an object with non-empty string keys/values")

    normalized_availability_filter = str(availability_filter or "all").strip().lower() or "all"
    if normalized_availability_filter not in _VALID_BACKEND_AVAILABILITY_FILTERS:
        return _error_result(
            (
                "availability_filter must be one of: "
                f"{sorted(_VALID_BACKEND_AVAILABILITY_FILTERS)}"
            )
        )

    result = await manage_collections(
        action="backend_status",
        include_capabilities=include_capabilities,
        backend_types=backend_types,
        unavailable_backends=unavailable_backends,
        unavailable_reasons=unavailable_reasons,
        availability_filter=normalized_availability_filter,
        include_breakdown=include_breakdown,
    )
    if result.get("status") == "error":
        return result

    backend_report = result.get("backend_report")
    if not isinstance(backend_report, dict):
        backend_report = {}

    backends = backend_report.get("backends")
    if not isinstance(backends, list):
        backends = []

    return {
        "status": "success",
        "availability_filter": backend_report.get("availability_filter", normalized_availability_filter),
        "backend_count": int(backend_report.get("backend_count", len(backends)) or 0),
        "backends": backends,
        "breakdown": backend_report.get("breakdown", {}),
        "backend_report": backend_report,
        "generated_at": backend_report.get("generated_at"),
    }


async def list_storage_collections(
    include_metadata: bool = True,
    include_timestamps: bool = True,
) -> Dict[str, Any]:
    """Return normalized storage collection inventory via manage_collections(list)."""
    if not isinstance(include_metadata, bool):
        return _error_result("include_metadata must be a boolean")
    if not isinstance(include_timestamps, bool):
        return _error_result("include_timestamps must be a boolean")

    result = await manage_collections(action="list")
    if result.get("status") == "error":
        return result

    collections = result.get("collections")
    if not isinstance(collections, list):
        collections = []

    normalized_collections: List[Dict[str, Any]] = []
    for entry in collections:
        if not isinstance(entry, dict):
            continue
        normalized = dict(entry)
        if not include_metadata:
            normalized.pop("metadata", None)
        if not include_timestamps:
            normalized.pop("created_at", None)
            normalized.pop("updated_at", None)
        normalized_collections.append(normalized)

    return {
        "status": "success",
        "collections": normalized_collections,
        "total_count": int(result.get("total_count", len(normalized_collections)) or 0),
        "generated_at": datetime.now().isoformat(),
    }


async def create_storage_collection(
    collection_name: str,
    description: Optional[str] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Create a storage collection via manage_collections(create)."""
    normalized_collection_name = str(collection_name or "").strip()
    if not normalized_collection_name:
        return _error_result("collection_name is required")
    if description is not None and (not isinstance(description, str) or not description.strip()):
        return _error_result("description must be a non-empty string when provided")
    if metadata is not None and not isinstance(metadata, dict):
        return _error_result("metadata must be an object when provided")
    if metadata is not None and not all(isinstance(key, str) and key.strip() for key in metadata.keys()):
        return _error_result("metadata keys must be non-empty strings when provided")

    result = await manage_collections(
        action="create",
        collection_name=normalized_collection_name,
        description=description,
        metadata=metadata,
    )
    if result.get("status") == "error":
        return result

    collection_payload = result.get("collection")
    if not isinstance(collection_payload, dict):
        collection_payload = {}

    return {
        "status": "success",
        "action": "create",
        "collection_name": normalized_collection_name,
        "created": bool(result.get("success", True)),
        "collection": collection_payload,
    }


async def get_storage_collection(
    collection_name: str,
    include_metadata: bool = True,
    include_timestamps: bool = True,
) -> Dict[str, Any]:
    """Return collection metadata via manage_collections(get)."""
    normalized_collection_name = str(collection_name or "").strip()
    if not normalized_collection_name:
        return _error_result("collection_name is required")
    if not isinstance(include_metadata, bool):
        return _error_result("include_metadata must be a boolean")
    if not isinstance(include_timestamps, bool):
        return _error_result("include_timestamps must be a boolean")

    result = await manage_collections(
        action="get",
        collection_name=normalized_collection_name,
    )
    if result.get("status") == "error":
        return result

    collection_payload = result.get("collection")
    if not isinstance(collection_payload, dict):
        collection_payload = {}
    normalized_collection = dict(collection_payload)
    if not include_metadata:
        normalized_collection.pop("metadata", None)
    if not include_timestamps:
        normalized_collection.pop("created_at", None)
        normalized_collection.pop("updated_at", None)

    return {
        "status": "success",
        "action": "get",
        "collection_name": normalized_collection_name,
        "collection": normalized_collection,
    }


async def delete_storage_collection(
    collection_name: str,
    delete_items: bool = False,
) -> Dict[str, Any]:
    """Delete a storage collection via manage_collections(delete)."""
    normalized_collection_name = str(collection_name or "").strip()
    if not normalized_collection_name:
        return _error_result("collection_name is required")
    if not isinstance(delete_items, bool):
        return _error_result("delete_items must be a boolean")

    result = await manage_collections(
        action="delete",
        collection_name=normalized_collection_name,
        delete_items=delete_items,
    )
    if result.get("status") == "error":
        if "error" not in result:
            result = {
                **result,
                "error": f"Collection '{normalized_collection_name}' could not be deleted",
            }
        return result

    return {
        "status": "success",
        "collection_name": normalized_collection_name,
        "deleted": bool(result.get("success", False)),
        "delete_items": delete_items,
        "action": "delete",
    }


async def delete_data(
    item_ids: List[str],
    missing_ok: bool = False,
) -> Dict[str, Any]:
    """Delete stored items through the source storage manager when available."""
    if not isinstance(item_ids, list) or not item_ids:
        return _error_result(
            "At least one item ID must be provided",
            deleted_count=0,
            missing_count=0,
            deleted_ids=[],
            missing_ids=[],
        )
    if not all(isinstance(item_id, str) and item_id.strip() for item_id in item_ids):
        return _error_result(
            "item_ids must be an array of non-empty strings",
            deleted_count=0,
            missing_count=0,
            deleted_ids=[],
            missing_ids=[],
        )
    if not isinstance(missing_ok, bool):
        return _error_result(
            "missing_ok must be a boolean",
            deleted_count=0,
            missing_count=0,
            deleted_ids=[],
            missing_ids=[],
        )

    storage_manager = _API.get("storage_manager")
    if storage_manager is None or not hasattr(storage_manager, "delete_item"):
        return _error_result(
            "delete_data unavailable: source storage manager is not available",
            deleted_count=0,
            missing_count=len(item_ids),
            deleted_ids=[],
            missing_ids=item_ids,
        )

    deleted_ids: List[str] = []
    missing_ids: List[str] = []
    try:
        for item_id in item_ids:
            normalized_item_id = item_id.strip()
            deleted = bool(storage_manager.delete_item(normalized_item_id))
            if deleted:
                deleted_ids.append(normalized_item_id)
            else:
                missing_ids.append(normalized_item_id)
    except Exception as exc:
        return _error_result(
            f"delete_data failed: {exc}",
            deleted_count=len(deleted_ids),
            missing_count=len(missing_ids),
            deleted_ids=deleted_ids,
            missing_ids=missing_ids,
        )

    if missing_ids and not missing_ok:
        return _error_result(
            "One or more item IDs were not found",
            deleted_count=len(deleted_ids),
            missing_count=len(missing_ids),
            deleted_ids=deleted_ids,
            missing_ids=missing_ids,
        )

    return {
        "status": "success",
        "deleted_count": len(deleted_ids),
        "missing_count": len(missing_ids),
        "deleted_ids": deleted_ids,
        "missing_ids": missing_ids,
        "missing_ok": missing_ok,
    }


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
                "tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "minLength": 1},
                },
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
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
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
                "include_breakdown": {"type": "boolean", "default": False},
                "include_capabilities": {"type": "boolean", "default": False},
                "backend_types": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": sorted(_VALID_STORAGE_TYPES), "minLength": 1},
                },
                "unavailable_backends": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": sorted(_VALID_STORAGE_TYPES), "minLength": 1},
                },
                "unavailable_reasons": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "string", "minLength": 1},
                },
                "availability_filter": {
                    "type": "string",
                    "enum": sorted(_VALID_BACKEND_AVAILABILITY_FILTERS),
                    "default": "all",
                },
                "report_format": {
                    "type": "string",
                    "enum": sorted(_VALID_REPORT_FORMATS),
                    "default": "detailed",
                },
            },
            "required": ["action"],
            "allOf": [
                {
                    "if": {
                        "properties": {
                            "action": {"enum": ["create", "get", "delete"]},
                        },
                    },
                    "then": {
                        "properties": {
                            "collection_name": {"type": "string", "minLength": 1},
                        },
                        "required": ["collection_name"],
                    },
                }
            ],
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
                "tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "minLength": 1},
                },
                "size_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer", "minimum": 0},
                    "minItems": 2,
                    "maxItems": 2,
                },
                "date_range": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "format": "date-time", "minLength": 1},
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

    manager.register_tool(
        category="storage_tools",
        name="list_storage",
        func=list_storage,
        description="List stored items using collection, backend, and tag filters.",
        input_schema={
            "type": "object",
            "properties": {
                "collection": {"type": ["string", "null"]},
                "storage_type": {"type": ["string", "null"], "enum": sorted(_VALID_STORAGE_TYPES) + [None]},
                "tags": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "minLength": 1},
                },
                "limit": {"type": "integer", "minimum": 1, "default": 100},
                "offset": {"type": "integer", "minimum": 0, "default": 0},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="get_storage_stats",
        func=get_storage_stats,
        description="Return normalized storage statistics for all collections or a specific collection.",
        input_schema={
            "type": "object",
            "properties": {
                "collection_name": {"type": ["string", "null"], "minLength": 1},
                "report_format": {
                    "type": "string",
                    "enum": sorted(_VALID_REPORT_FORMATS),
                    "default": "summary",
                },
                "include_breakdown": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="get_storage_collection_stats",
        func=get_storage_collection_stats,
        description="Return normalized collection-scoped storage statistics.",
        input_schema={
            "type": "object",
            "properties": {
                "collection_name": {"type": "string", "minLength": 1},
                "report_format": {
                    "type": "string",
                    "enum": sorted(_VALID_REPORT_FORMATS),
                    "default": "summary",
                },
                "include_breakdown": {"type": "boolean", "default": False},
            },
            "required": ["collection_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="get_storage_lifecycle_report",
        func=get_storage_lifecycle_report,
        description="Return normalized storage lifecycle telemetry for all collections or a specific collection.",
        input_schema={
            "type": "object",
            "properties": {
                "collection_name": {"type": ["string", "null"], "minLength": 1},
                "report_format": {
                    "type": "string",
                    "enum": sorted(_VALID_REPORT_FORMATS),
                    "default": "detailed",
                },
                "include_breakdown": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="get_storage_backend_status",
        func=get_storage_backend_status,
        description="Return normalized backend availability telemetry.",
        input_schema={
            "type": "object",
            "properties": {
                "include_capabilities": {"type": "boolean", "default": False},
                "backend_types": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": sorted(_VALID_STORAGE_TYPES), "minLength": 1},
                },
                "unavailable_backends": {
                    "type": ["array", "null"],
                    "items": {"type": "string", "enum": sorted(_VALID_STORAGE_TYPES), "minLength": 1},
                },
                "unavailable_reasons": {
                    "type": ["object", "null"],
                    "additionalProperties": {"type": "string", "minLength": 1},
                },
                "availability_filter": {
                    "type": "string",
                    "enum": sorted(_VALID_BACKEND_AVAILABILITY_FILTERS),
                    "default": "all",
                },
                "include_breakdown": {"type": "boolean", "default": False},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="list_storage_collections",
        func=list_storage_collections,
        description="Return normalized collection inventory from storage collections list operation.",
        input_schema={
            "type": "object",
            "properties": {
                "include_metadata": {"type": "boolean", "default": True},
                "include_timestamps": {"type": "boolean", "default": True},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="create_storage_collection",
        func=create_storage_collection,
        description="Create a storage collection with optional description and metadata.",
        input_schema={
            "type": "object",
            "properties": {
                "collection_name": {"type": "string", "minLength": 1},
                "description": {"type": ["string", "null"], "minLength": 1},
                "metadata": {
                    "type": ["object", "null"],
                    "propertyNames": {"type": "string", "minLength": 1},
                },
            },
            "required": ["collection_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="get_storage_collection",
        func=get_storage_collection,
        description="Return normalized metadata for a specific storage collection.",
        input_schema={
            "type": "object",
            "properties": {
                "collection_name": {"type": "string", "minLength": 1},
                "include_metadata": {"type": "boolean", "default": True},
                "include_timestamps": {"type": "boolean", "default": True},
            },
            "required": ["collection_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="delete_storage_collection",
        func=delete_storage_collection,
        description="Delete a storage collection with optional cascading item deletion.",
        input_schema={
            "type": "object",
            "properties": {
                "collection_name": {"type": "string", "minLength": 1},
                "delete_items": {"type": "boolean", "default": False},
            },
            "required": ["collection_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )

    manager.register_tool(
        category="storage_tools",
        name="delete_data",
        func=delete_data,
        description="Delete stored items by ID using the underlying storage manager.",
        input_schema={
            "type": "object",
            "properties": {
                "item_ids": {
                    "type": "array",
                    "items": {"type": "string", "minLength": 1},
                    "minItems": 1,
                },
                "missing_ok": {"type": "boolean", "default": False},
            },
            "required": ["item_ids"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "storage"],
    )
