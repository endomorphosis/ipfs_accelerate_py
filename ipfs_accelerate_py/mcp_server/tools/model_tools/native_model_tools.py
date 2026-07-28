"""Native model-tools category implementations for unified mcp_server."""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Mapping, Optional, Sequence

import anyio

logger = logging.getLogger(__name__)

CATALOG_TOOL_SCHEMA_VERSION = "ai.catalog.mcp.v1"
USAGE_TOOL_SCHEMA_VERSION = "ai.usage.mcp.v1"
MAX_CATALOG_PAGE_SIZE = 1_000
MAX_CATALOG_SOURCES = 64
MAX_USAGE_PAGE_SIZE = 100
_REDACTED = "[REDACTED]"

# Process-local usage control service (injected by tests / runtime).
_USAGE_CONTROL_SERVICE: Any = None


def _load_model_tools_api() -> Dict[str, Any]:
    """Resolve source model-tools APIs with compatibility fallback."""
    try:
        from ipfs_accelerate_py.mcp.tools.models import (  # type: ignore
            search_models_tool as _search_models,
            recommend_models_tool as _recommend_models,
            get_model_details_tool as _get_model_details,
            get_model_stats_tool as _get_model_stats,
            list_hf_inference_models_tool as _list_hf_inference_models,
            get_hf_inference_model_metadata_tool as _get_hf_inference_model_metadata,
            build_hf_inference_ipld_document_tool as _build_hf_inference_ipld_document,
            get_hf_inference_ipld_cid_tool as _get_hf_inference_ipld_cid,
            publish_hf_inference_ipld_to_ipfs_tool as _publish_hf_inference_ipld_to_ipfs,
            load_hf_inference_ipld_from_ipfs_tool as _load_hf_inference_ipld_from_ipfs,
        )

        return {
            "search_models": _search_models,
            "recommend_models": _recommend_models,
            "get_model_details": _get_model_details,
            "get_model_stats": _get_model_stats,
            "list_hf_inference_models": _list_hf_inference_models,
            "get_hf_inference_model_metadata": _get_hf_inference_model_metadata,
            "build_hf_inference_ipld_document": _build_hf_inference_ipld_document,
            "get_hf_inference_ipld_cid": _get_hf_inference_ipld_cid,
            "publish_hf_inference_ipld_to_ipfs": _publish_hf_inference_ipld_to_ipfs,
            "load_hf_inference_ipld_from_ipfs": _load_hf_inference_ipld_from_ipfs,
        }
    except Exception:
        logger.warning("Source model_tools import unavailable, using fallback model functions")

        def _search_fallback(
            query: str,
            task_filter: Optional[str] = None,
            limit: int = 10,
        ) -> Dict[str, Any]:
            return {"status": "success", "models": [], "query": query, "count": 0}

        def _recommend_fallback(
            task_type: str,
            hardware: str = "cpu",
            max_size_gb: Optional[float] = None,
        ) -> Dict[str, Any]:
            return {"status": "success", "recommendations": [], "task_type": task_type}

        def _details_fallback(model_id: str) -> Dict[str, Any]:
            return {"status": "success", "model_id": model_id, "details": {}}

        def _stats_fallback() -> Dict[str, Any]:
            return {"status": "success", "stats": {}}

        def _list_hf_fallback(model_kind: Optional[str] = None) -> Dict[str, Any]:
            return {"status": "success", "models": [], "count": 0}

        def _hf_metadata_fallback(model_id: str) -> Dict[str, Any]:
            return {"status": "success", "model_id": model_id, "metadata": {}}

        def _build_ipld_fallback(model_id: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "model_id": model_id, "document": {}}

        def _get_cid_fallback(model_id: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "model_id": model_id, "cid": None}

        def _publish_ipld_fallback(model_id: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "model_id": model_id, "published": False}

        def _load_ipld_fallback(cid: str, **kwargs: Any) -> Dict[str, Any]:
            return {"status": "success", "cid": cid, "document": {}}

        return {
            "search_models": _search_fallback,
            "recommend_models": _recommend_fallback,
            "get_model_details": _details_fallback,
            "get_model_stats": _stats_fallback,
            "list_hf_inference_models": _list_hf_fallback,
            "get_hf_inference_model_metadata": _hf_metadata_fallback,
            "build_hf_inference_ipld_document": _build_ipld_fallback,
            "get_hf_inference_ipld_cid": _get_cid_fallback,
            "publish_hf_inference_ipld_to_ipfs": _publish_ipld_fallback,
            "load_hf_inference_ipld_from_ipfs": _load_ipld_fallback,
        }


_API = _load_model_tools_api()


def _normalize_payload(payload: Any) -> Dict[str, Any]:
    """Normalize delegate payloads to deterministic dict envelopes."""
    if isinstance(payload, dict):
        envelope = dict(payload)
        failed = bool(envelope.get("error")) or envelope.get("success") is False
        if failed:
            envelope["status"] = "error"
        elif "status" not in envelope:
            envelope["status"] = "success"
        return envelope
    if payload is None:
        return {"status": "success"}
    return {"status": "success", "result": payload}


def _error_result(message: str, **context: Any) -> Dict[str, Any]:
    """Build consistent error envelope for wrapper edge failures."""
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "error": message,
    }
    envelope.update(context)
    return envelope


def _catalog_error_result(
    code: str,
    message: str,
    *,
    schema_version: Optional[str] = None,
    catalog_revision: Optional[str] = None,
    **context: Any,
) -> Dict[str, Any]:
    """Build a typed, secret-safe catalog error envelope."""

    error = {"code": code, "message": message}
    envelope: Dict[str, Any] = {
        "status": "error",
        "success": False,
        "tool_schema_version": CATALOG_TOOL_SCHEMA_VERSION,
        "schema_version": schema_version,
        "catalog_revision": catalog_revision,
        "error": error,
        # Flat aliases keep error handling convenient for MCP clients that do
        # not model nested discriminated unions.
        "error_code": code,
        "error_type": code,
    }
    envelope.update(context)
    return envelope


def _catalog_exception_result(
    exc: BaseException,
    *,
    default_code: str,
    schema_version: Optional[str] = None,
    catalog_revision: Optional[str] = None,
    **context: Any,
) -> Dict[str, Any]:
    """Map catalog exceptions without reflecting source- or user-owned text."""

    exception_name = type(exc).__name__
    errors = {
        "StaleCursorError": (
            "cursor_revision_mismatch",
            "The cursor belongs to a different catalog revision.",
        ),
        "InvalidCursorError": (
            "invalid_cursor",
            "The catalog cursor is malformed or does not match this query.",
        ),
        "RefreshPolicyError": (
            "refresh_denied",
            "Catalog refresh was not authorized.",
        ),
        "PermissionError": (
            "refresh_denied",
            "Catalog refresh was not authorized.",
        ),
        "CatalogSourceError": (
            default_code,
            "The catalog source request is invalid.",
        ),
        "ResolutionError": (
            default_code,
            "The catalog resolution constraints are invalid.",
        ),
        "SchemaValidationError": (
            default_code,
            "The catalog request violates the catalog schema.",
        ),
        "ValueError": (
            default_code,
            "The catalog request is invalid.",
        ),
        "TypeError": (
            default_code,
            "The catalog request has an invalid value type.",
        ),
    }
    code, message = errors.get(
        exception_name,
        ("catalog_unavailable", "The catalog request could not be completed."),
    )
    return _catalog_error_result(
        code,
        message,
        schema_version=schema_version,
        catalog_revision=catalog_revision,
        **context,
    )


def _redact_catalog_payload(value: Any) -> Any:
    """Return JSON-safe catalog data without credentials or raw endpoints."""

    from ipfs_accelerate_py.model_catalog.identity import redact_secrets

    redacted = redact_secrets(value)

    def hide_endpoints(item: Any) -> Any:
        if isinstance(item, Mapping):
            return {
                str(key): (
                    _REDACTED
                    if str(key).casefold() == "endpoint_uri"
                    else hide_endpoints(child)
                )
                for key, child in item.items()
            }
        if isinstance(item, (list, tuple)):
            return [hide_endpoints(child) for child in item]
        return item

    return hide_endpoints(redacted)


def _catalog_success(
    *,
    schema_version: str,
    catalog_revision: str,
    **payload: Any,
) -> Dict[str, Any]:
    """Build a versioned, redacted successful catalog envelope."""

    envelope = {
        "status": "success",
        "success": True,
        "tool_schema_version": CATALOG_TOOL_SCHEMA_VERSION,
        "schema_version": schema_version,
        "catalog_revision": catalog_revision,
    }
    envelope.update(payload)
    return _redact_catalog_payload(envelope)


def _snapshot_versions(snapshot: Any) -> Dict[str, str]:
    schema_version = getattr(snapshot, "schema_version", None)
    revision = getattr(snapshot, "revision", None)
    if not isinstance(schema_version, str) or not schema_version:
        raise TypeError("catalog snapshot schema version is invalid")
    if not isinstance(revision, str) or not revision:
        raise TypeError("catalog snapshot revision is invalid")
    return {"schema_version": schema_version, "catalog_revision": revision}


def _catalog_page_payload(page: Any, item_key: str) -> Dict[str, Any]:
    data = page.to_dict()
    items = _redact_catalog_payload(data["items"])
    return {
        "items": items,
        item_key: items,
        "record_type": data["record_type"],
        "count": len(items),
        "total": data["total"],
        "next_cursor": data["next_cursor"],
    }


def _validate_refresh_sources(sources: Any) -> Sequence[str]:
    if (
        isinstance(sources, (str, bytes, Mapping))
        or not isinstance(sources, Sequence)
        or not sources
        or len(sources) > MAX_CATALOG_SOURCES
        or any(not isinstance(item, str) or not item for item in sources)
    ):
        raise ValueError(
            "sources must be a non-empty bounded array of source names"
        )
    return tuple(sources)


async def _run_catalog_read(callback: Any) -> Dict[str, Any]:
    """Run a catalog read against one captured immutable manager snapshot."""

    def run() -> Dict[str, Any]:
        from ipfs_accelerate_py.model_manager import get_default_model_manager

        manager = get_default_model_manager()
        snapshot = manager.snapshot()
        versions = _snapshot_versions(snapshot)
        try:
            return callback(manager, snapshot, versions)
        except Exception as exc:
            return _catalog_exception_result(
                exc,
                default_code="invalid_filter",
                **versions,
            )

    try:
        return await anyio.to_thread.run_sync(run)
    except Exception as exc:
        return _catalog_exception_result(exc, default_code="invalid_filter")


async def model_catalog_list_services(
    limit: int = 100,
    cursor: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    operation: Optional[str] = None,
    modality: Optional[str] = None,
    state: Optional[Dict[str, bool]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """List canonical AI service providers from ModelManager's catalog."""

    def read(manager: Any, snapshot: Any, versions: Dict[str, str]) -> Dict[str, Any]:
        page = manager.list_services(
            limit=limit,
            cursor=cursor,
            provider=provider,
            model=model,
            operation=operation,
            modality=modality,
            state=state,
            labels=labels,
            snapshot=snapshot,
        )
        return _catalog_success(
            **versions,
            **_catalog_page_payload(page, "services"),
        )

    return await _run_catalog_read(read)


async def model_catalog_list_models(
    limit: int = 100,
    cursor: Optional[str] = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    operation: Optional[str] = None,
    modality: Optional[str] = None,
    state: Optional[Dict[str, bool]] = None,
    labels: Optional[Dict[str, str]] = None,
) -> Dict[str, Any]:
    """List canonical model descriptors from ModelManager's catalog."""

    def read(manager: Any, snapshot: Any, versions: Dict[str, str]) -> Dict[str, Any]:
        page = manager.list_catalog_models(
            limit=limit,
            cursor=cursor,
            provider=provider,
            model=model,
            operation=operation,
            modality=modality,
            state=state,
            labels=labels,
            snapshot=snapshot,
        )
        return _catalog_success(
            **versions,
            **_catalog_page_payload(page, "models"),
        )

    return await _run_catalog_read(read)


async def model_catalog_get(
    identifier: str,
    record_type: Optional[str] = None,
) -> Dict[str, Any]:
    """Get one canonical catalog record, failing closed on ambiguous aliases."""

    def read(manager: Any, snapshot: Any, versions: Dict[str, str]) -> Dict[str, Any]:
        result = manager.get(
            identifier,
            record_type=record_type,
            snapshot=snapshot,
        )
        data = _redact_catalog_payload(result.to_dict())
        if not result.found:
            diagnostic = next(iter(data["diagnostics"]), {})
            code = diagnostic.get("code", "no_match")
            message = (
                "The identifier matches more than one canonical record."
                if code == "ambiguous_identifier"
                else "The identifier did not match a canonical record."
            )
            return _catalog_error_result(
                code,
                message,
                **versions,
                record_type=data["record_type"],
                query=data["query"],
                record=None,
                diagnostics=data["diagnostics"],
            )
        return _catalog_success(
            **versions,
            record_type=data["record_type"],
            query=data["query"],
            record=data["record"],
            diagnostics=data["diagnostics"],
        )

    return await _run_catalog_read(read)


async def model_catalog_resolve(
    operation: str,
    modality: Optional[str] = None,
    model: Optional[str] = None,
    provider: Optional[str] = None,
    deployment: Optional[str] = None,
    policy: Optional[Dict[str, Any]] = None,
    device: Optional[str] = None,
    context: Optional[int] = None,
    health: Optional[bool] = None,
    locality: Optional[str] = None,
    configured: Optional[bool] = None,
    authorized: Optional[bool] = None,
    reachable: Optional[bool] = None,
    routable: Optional[bool] = None,
    limit: int = 100,
) -> Dict[str, Any]:
    """Resolve bounded catalog constraints without invoking a provider."""

    constraints = {
        "operation": operation,
        "modality": modality,
        "model": model,
        "provider": provider,
        "deployment": deployment,
        "policy": policy,
        "device": device,
        "context": context,
        "health": health,
        "locality": locality,
        "configured": configured,
        "authorized": authorized,
        "reachable": reachable,
        "routable": routable,
        "limit": limit,
    }

    def read(manager: Any, snapshot: Any, versions: Dict[str, str]) -> Dict[str, Any]:
        result = manager.resolve(snapshot=snapshot, **constraints)
        data = _redact_catalog_payload(result.to_dict())
        if not result.found:
            ambiguous = any(
                "ambiguous" in str(reason).casefold()
                for reason in data.get("reasons", ())
            )
            code = "ambiguous_identifier" if ambiguous else "no_match"
            message = (
                "The constraints contain an ambiguous canonical identifier."
                if ambiguous
                else "No catalog candidate satisfies the constraints."
            )
            return _catalog_error_result(
                code,
                message,
                **versions,
                resolution=data,
            )
        return _catalog_success(**versions, resolution=data)

    return await _run_catalog_read(read)


async def model_catalog_health() -> Dict[str, Any]:
    """Return published catalog health without refreshing or probing sources."""

    def read(manager: Any, snapshot: Any, versions: Dict[str, str]) -> Dict[str, Any]:
        health = manager.health(snapshot=snapshot)
        return _catalog_success(
            **versions,
            health=_redact_catalog_payload(health.to_dict()),
        )

    return await _run_catalog_read(read)


def set_usage_control_service(service: Any) -> None:
    """Inject the process-local :class:`UsageControlService` for MCP tools."""

    global _USAGE_CONTROL_SERVICE
    _USAGE_CONTROL_SERVICE = service


def get_usage_control_service() -> Any:
    """Return the injected usage control service, if any."""

    return _USAGE_CONTROL_SERVICE


def _usage_service_or_error() -> Any:
    service = _USAGE_CONTROL_SERVICE
    if service is None:
        return None
    return service


def _usage_unavailable() -> Dict[str, Any]:
    return {
        "status": "error",
        "success": False,
        "tool_schema_version": USAGE_TOOL_SCHEMA_VERSION,
        "schema_version": "ai.endpoint_usage.control.v1",
        "error": {
            "code": "usage_unavailable",
            "message": "Usage control service is not configured.",
        },
        "error_code": "usage_unavailable",
        "error_type": "usage_unavailable",
        "reason_codes": ["usage_unavailable"],
    }


def _usage_authorities(
    *,
    read: bool = True,
    detail: bool = False,
    admin: bool = False,
    authorities: Optional[Sequence[str]] = None,
) -> List[str]:
    """Normalize authority tokens for the usage control service."""

    if authorities is not None:
        return [str(item) for item in authorities if isinstance(item, str) and item]
    from ipfs_accelerate_py.endpoint_usage.controls import (
        USAGE_ADMIN_AUTHORITY,
        USAGE_READ_AUTHORITY,
        USAGE_READ_DETAIL_AUTHORITY,
    )

    granted: List[str] = []
    if read:
        granted.append(USAGE_READ_AUTHORITY)
    if detail:
        granted.append(USAGE_READ_DETAIL_AUTHORITY)
    if admin:
        granted.append(USAGE_ADMIN_AUTHORITY)
    return granted


async def model_catalog_usage(
    operation: str,
    *,
    scope_id: Optional[str] = None,
    limit: int = 50,
    cursor: Optional[str] = None,
    dimension: Optional[str] = None,
    state: Optional[str] = None,
    expected_usage_revision: Optional[str] = None,
    authorities: Optional[List[str]] = None,
    read_detail: bool = False,
    admin: bool = False,
    idempotency_key: Optional[str] = None,
    lease_id: Optional[str] = None,
    fence: Optional[int] = None,
    expected_effects: Optional[List[str]] = None,
    actor: Optional[str] = None,
    source: str = "operator",
    kind: Optional[str] = None,
    units: Optional[Dict[str, Any]] = None,
    limits: Optional[List[Dict[str, Any]]] = None,
    limits_update: Optional[List[Dict[str, Any]]] = None,
    supersedes_event_id: Optional[str] = None,
    reservation_id: Optional[str] = None,
    reason: Optional[str] = None,
    cooldown_until: Optional[str] = None,
    observation_id: Optional[str] = None,
    adapter_id: Optional[str] = None,
    reason_codes: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Bounded usage status/limits/headroom/reservations/receipts/admin controls.

    Read/query operations are side-effect free. Administrative mutations require
    explicit admin authority, expected revision, idempotency, lease, and fence.
    """

    service = _usage_service_or_error()
    if service is None:
        return _usage_unavailable()

    op = str(operation or "").strip().casefold()
    granted = _usage_authorities(
        read=True,
        detail=bool(read_detail),
        admin=bool(admin),
        authorities=authorities,
    )

    def run() -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {"authorities": granted}
        if op in {"status"}:
            return service.status(
                scope_id=scope_id,
                limit=limit,
                cursor=cursor,
                state=state,
                **kwargs,
            )
        if op in {"health"}:
            return service.health(**kwargs)
        if op in {"limits"}:
            if not scope_id:
                raise ValueError("scope_id is required")
            return service.limits(
                scope_id,
                limit=limit,
                cursor=cursor,
                dimension=dimension,
                expected_usage_revision=expected_usage_revision,
                **kwargs,
            )
        if op in {"headroom"}:
            if not scope_id:
                raise ValueError("scope_id is required")
            return service.headroom(
                scope_id,
                dimension=dimension,
                expected_usage_revision=expected_usage_revision,
                **kwargs,
            )
        if op in {"reservations"}:
            if not scope_id:
                raise ValueError("scope_id is required")
            return service.reservations(
                scope_id,
                limit=limit,
                cursor=cursor,
                expected_usage_revision=expected_usage_revision,
                **kwargs,
            )
        if op in {"receipts"}:
            return service.receipts(
                scope_id=scope_id,
                limit=limit,
                cursor=cursor,
                **kwargs,
            )
        if op in {"adapter_capabilities", "adapters"}:
            return service.adapter_capabilities(
                adapter_id=adapter_id,
                limit=limit,
                cursor=cursor,
                **kwargs,
            )
        if op in {"import", "import_observation"}:
            if not scope_id:
                raise ValueError("scope_id is required")
            return service.import_observation(
                scope_id,
                expected_usage_revision=expected_usage_revision,
                idempotency_key=idempotency_key,
                lease_id=lease_id,
                fence=fence,
                expected_effects=expected_effects,
                actor=actor,
                source=source,
                kind=kind or "observation_success",
                units=units,
                limits_update=limits_update,
                cooldown_until=cooldown_until,
                reason_codes=tuple(reason_codes or ()),
                observation_id=observation_id,
                reservation_id=reservation_id,
                **kwargs,
            )
        if op in {"correct", "correction"}:
            if not scope_id:
                raise ValueError("scope_id is required")
            return service.correct(
                scope_id,
                expected_usage_revision=expected_usage_revision,
                idempotency_key=idempotency_key,
                lease_id=lease_id,
                fence=fence,
                expected_effects=expected_effects,
                actor=actor,
                source=source,
                supersedes_event_id=supersedes_event_id,
                units=units,
                reservation_id=reservation_id,
                reason=reason or "correction",
                **kwargs,
            )
        if op in {"override", "override_limits"}:
            if not scope_id:
                raise ValueError("scope_id is required")
            return service.override_limits(
                scope_id,
                expected_usage_revision=expected_usage_revision,
                idempotency_key=idempotency_key,
                lease_id=lease_id,
                fence=fence,
                expected_effects=expected_effects,
                actor=actor,
                source=source,
                limits=limits,
                **kwargs,
            )
        if op in {"reset"}:
            if not scope_id:
                raise ValueError("scope_id is required")
            return service.reset(
                scope_id,
                expected_usage_revision=expected_usage_revision,
                idempotency_key=idempotency_key,
                lease_id=lease_id,
                fence=fence,
                expected_effects=expected_effects,
                actor=actor,
                source=source,
                reason=reason or "admin_reset",
                **kwargs,
            )
        return {
            "status": "error",
            "success": False,
            "tool_schema_version": USAGE_TOOL_SCHEMA_VERSION,
            "schema_version": "ai.endpoint_usage.control.v1",
            "error": {
                "code": "invalid_request",
                "message": "Unknown usage control operation.",
            },
            "error_code": "invalid_request",
            "error_type": "invalid_request",
            "reason_codes": ["invalid_request"],
        }

    try:
        return await anyio.to_thread.run_sync(run)
    except Exception as exc:
        service = _USAGE_CONTROL_SERVICE
        if service is not None and hasattr(service, "_error"):
            return service._error(exc, authorities=granted)
        return {
            "status": "error",
            "success": False,
            "tool_schema_version": USAGE_TOOL_SCHEMA_VERSION,
            "schema_version": "ai.endpoint_usage.control.v1",
            "error": {
                "code": "invalid_request",
                "message": "Usage control request failed.",
            },
            "error_code": "invalid_request",
            "error_type": "invalid_request",
            "reason_codes": ["invalid_request"],
        }


async def model_catalog_usage_metrics(
    *,
    authorities: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Export event-derived low-cardinality usage metrics (read-only)."""

    service = _usage_service_or_error()
    if service is None:
        return _usage_unavailable()
    granted = _usage_authorities(read=True, authorities=authorities)

    def run() -> Dict[str, Any]:
        from ipfs_accelerate_py.endpoint_usage.controls import require_read

        try:
            require_read(granted)
        except Exception as exc:
            if hasattr(service, "_error"):
                return service._error(exc, authorities=granted)
            raise
        obs = getattr(service, "_observability", None)
        if obs is None:
            return {
                "status": "success",
                "success": True,
                "tool_schema_version": USAGE_TOOL_SCHEMA_VERSION,
                "schema_version": "ai.endpoint_usage.metrics.v1",
                "samples": [],
                "series_count": 0,
                "reason_codes": ["ok"],
            }
        payload = obs.snapshot()
        payload.update(
            {
                "status": "success",
                "success": True,
                "tool_schema_version": USAGE_TOOL_SCHEMA_VERSION,
            }
        )
        return payload

    try:
        return await anyio.to_thread.run_sync(run)
    except Exception as exc:
        if service is not None and hasattr(service, "_error"):
            return service._error(exc, authorities=granted)
        return _usage_unavailable()


async def model_catalog_refresh(
    sources: List[str],
    authority: bool = False,
) -> Dict[str, Any]:
    """Refresh explicitly named catalog sources under explicit caller authority."""

    if authority is not True:
        return _catalog_error_result(
            "refresh_denied",
            "Catalog refresh requires explicit authority.",
        )
    try:
        selected = _validate_refresh_sources(sources)
    except Exception as exc:
        return _catalog_exception_result(exc, default_code="invalid_sources")

    def run() -> Dict[str, Any]:
        from ipfs_accelerate_py.model_catalog.catalog import RefreshPolicy
        from ipfs_accelerate_py.model_manager import get_default_model_manager

        manager = get_default_model_manager()
        before = manager.snapshot()
        before_versions = _snapshot_versions(before)
        try:
            result = manager.refresh(
                selected,
                policy=RefreshPolicy(
                    allow_side_effects=True,
                    allowed_sources=tuple(selected),
                ),
            )
            snapshot = result.snapshot
            versions = _snapshot_versions(snapshot)
            payload = {
                "refreshed": list(result.refreshed),
                "failed": list(result.failed),
                "unchanged": list(result.unchanged),
                "source_states": [
                    item.to_dict() for item in result.source_states
                ],
                "diagnostics": [
                    item.to_dict() for item in result.diagnostics
                ],
            }
            if result.failed:
                return _catalog_error_result(
                    "source_refresh_failed",
                    "One or more named catalog sources failed to refresh.",
                    **versions,
                    **_redact_catalog_payload(payload),
                )
            return _catalog_success(**versions, **payload)
        except Exception as exc:
            return _catalog_exception_result(
                exc,
                default_code="invalid_sources",
                **before_versions,
            )

    try:
        return await anyio.to_thread.run_sync(run)
    except Exception as exc:
        return _catalog_exception_result(exc, default_code="invalid_sources")


async def model_search(
    query: str,
    task_filter: Optional[str] = None,
    limit: int = 10,
) -> Dict[str, Any]:
    """Search for models matching a query."""
    try:
        result = _API["search_models"](query=query, task_filter=task_filter, limit=limit)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), query=query)


async def model_recommend(
    task_type: str,
    hardware: str = "cpu",
    max_size_gb: Optional[float] = None,
) -> Dict[str, Any]:
    """Recommend models for a task and hardware configuration."""
    try:
        result = _API["recommend_models"](
            task_type=task_type, hardware=hardware, max_size_gb=max_size_gb
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), task_type=task_type, hardware=hardware)


async def model_get_details(model_id: str) -> Dict[str, Any]:
    """Get detailed information about a specific model."""
    try:
        result = _API["get_model_details"](model_id=model_id)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_get_stats() -> Dict[str, Any]:
    """Get aggregate statistics about available models."""
    try:
        result = _API["get_model_stats"]()
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def model_list_served(
    endpoint_url: Optional[str] = None,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """List models that are live on configured inference endpoints."""
    try:
        from ipfs_accelerate_py.model_manager import get_default_model_manager

        manager = get_default_model_manager()
        models = await anyio.to_thread.run_sync(
            lambda: manager.list_served_models(endpoint_url=endpoint_url, timeout=timeout)
        )
        return {"status": "success", "models": models, "count": len(models)}
    except Exception as exc:
        return _error_result(str(exc), models=[], count=0)


async def model_get_served(
    model_id: str,
    endpoint_url: Optional[str] = None,
    timeout: float = 2.0,
) -> Dict[str, Any]:
    """Get live serving information for a model ID."""
    try:
        from ipfs_accelerate_py.model_manager import get_default_model_manager

        manager = get_default_model_manager()
        model = await anyio.to_thread.run_sync(
            lambda: manager.get_served_model(
                model_id, endpoint_url=endpoint_url, timeout=timeout
            )
        )
        if model is None:
            return _error_result(f"Model is not currently served: {model_id}", model_id=model_id)
        return {"status": "success", "model": model}
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_list_hf_inference(
    model_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """List HuggingFace inference models."""
    try:
        result = _API["list_hf_inference_models"](model_kind=model_kind)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc))


async def model_get_hf_metadata(model_id: str) -> Dict[str, Any]:
    """Get metadata for a HuggingFace inference model."""
    try:
        result = _API["get_hf_inference_model_metadata"](model_id=model_id)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_build_hf_ipld_document(
    model_id: str,
    include_config: bool = True,
    include_tokenizer: bool = True,
) -> Dict[str, Any]:
    """Build an IPLD document for a HuggingFace inference model."""
    try:
        result = _API["build_hf_inference_ipld_document"](
            model_id=model_id,
            include_config=include_config,
            include_tokenizer=include_tokenizer,
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_get_hf_ipld_cid(
    model_id: str,
    include_config: bool = True,
    include_tokenizer: bool = True,
) -> Dict[str, Any]:
    """Get the IPLD CID for a HuggingFace inference model document."""
    try:
        result = _API["get_hf_inference_ipld_cid"](
            model_id=model_id,
            include_config=include_config,
            include_tokenizer=include_tokenizer,
        )
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_publish_hf_ipld_to_ipfs(
    model_id: str,
    pin: bool = True,
) -> Dict[str, Any]:
    """Publish a HuggingFace model IPLD document to IPFS."""
    try:
        result = _API["publish_hf_inference_ipld_to_ipfs"](model_id=model_id, pin=pin)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), model_id=model_id)


async def model_load_hf_ipld_from_ipfs(cid: str) -> Dict[str, Any]:
    """Load a HuggingFace model IPLD document from IPFS."""
    try:
        result = _API["load_hf_inference_ipld_from_ipfs"](cid=cid)
        return _normalize_payload(result)
    except Exception as exc:
        return _error_result(str(exc), cid=cid)


def register_native_model_tools(manager: Any) -> None:
    """Register native model-tools category tools in unified manager."""
    catalog_filter_properties = {
        "limit": {
            "type": "integer",
            "minimum": 1,
            "maximum": MAX_CATALOG_PAGE_SIZE,
            "default": 100,
        },
        "cursor": {"type": "string", "minLength": 1, "maxLength": 4096},
        "provider": {"type": "string", "minLength": 1, "maxLength": 256},
        "model": {"type": "string", "minLength": 1, "maxLength": 256},
        "operation": {"type": "string", "minLength": 1, "maxLength": 64},
        "modality": {"type": "string", "minLength": 1, "maxLength": 64},
        "state": {
            "type": "object",
            "maxProperties": 6,
            "additionalProperties": {"type": "boolean"},
        },
        "labels": {
            "type": "object",
            "maxProperties": 64,
            "additionalProperties": {
                "type": "string",
                "maxLength": 256,
            },
        },
    }
    for name, func, description in (
        (
            "model_catalog_list_services",
            model_catalog_list_services,
            "List canonical AI services with bounded filters and revision-bound pagination.",
        ),
        (
            "model_catalog_list_models",
            model_catalog_list_models,
            "List canonical AI models with bounded filters and revision-bound pagination.",
        ),
    ):
        manager.register_tool(
            category="model_tools",
            name=name,
            func=func,
            description=description,
            input_schema={
                "type": "object",
                "properties": dict(catalog_filter_properties),
                "required": [],
                "additionalProperties": False,
            },
            runtime="fastapi",
            tags=["native", "mcpp", "model-tools", "catalog", "read-only"],
        )
    manager.register_tool(
        category="model_tools",
        name="model_catalog_get",
        func=model_catalog_get,
        description="Get one canonical catalog record by ID, name, or alias.",
        input_schema={
            "type": "object",
            "properties": {
                "identifier": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 256,
                },
                "record_type": {
                    "type": "string",
                    "enum": [
                        "records",
                        "providers",
                        "models",
                        "deployments",
                        "bindings",
                    ],
                },
            },
            "required": ["identifier"],
            "additionalProperties": False,
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools", "catalog", "read-only"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_catalog_resolve",
        func=model_catalog_resolve,
        description="Resolve canonical providers and models without invoking or probing them.",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                },
                "modality": {"type": "string", "maxLength": 64},
                "model": {"type": "string", "maxLength": 256},
                "provider": {"type": "string", "maxLength": 256},
                "deployment": {"type": "string", "maxLength": 256},
                "policy": {
                    "type": "object",
                    "maxProperties": 64,
                    "additionalProperties": {
                        "type": ["string", "number", "boolean"],
                    },
                },
                "device": {"type": "string", "maxLength": 256},
                "context": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 100000000,
                },
                "health": {"type": "boolean"},
                "locality": {"type": "string", "maxLength": 256},
                "configured": {"type": "boolean"},
                "authorized": {"type": "boolean"},
                "reachable": {"type": "boolean"},
                "routable": {"type": "boolean"},
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_CATALOG_PAGE_SIZE,
                    "default": 100,
                },
            },
            "required": ["operation"],
            "additionalProperties": False,
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools", "catalog", "read-only"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_catalog_health",
        func=model_catalog_health,
        description="Read already-published catalog and source health without active probes.",
        input_schema={
            "type": "object",
            "properties": {},
            "required": [],
            "additionalProperties": False,
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools", "catalog", "read-only"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_catalog_refresh",
        func=model_catalog_refresh,
        description="Privileged refresh of explicitly named catalog sources.",
        input_schema={
            "type": "object",
            "properties": {
                "sources": {
                    "type": "array",
                    "minItems": 1,
                    "maxItems": MAX_CATALOG_SOURCES,
                    "uniqueItems": True,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                },
                "authority": {
                    "type": "boolean",
                    "const": True,
                    "description": "Explicit authorization for this named refresh.",
                },
            },
            "required": ["sources", "authority"],
            "additionalProperties": False,
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools", "catalog", "privileged"],
    )
    usage_operation_enum = [
        "status",
        "health",
        "limits",
        "headroom",
        "reservations",
        "receipts",
        "adapter_capabilities",
        "import",
        "correct",
        "override",
        "reset",
    ]
    manager.register_tool(
        category="model_tools",
        name="model_catalog_usage",
        func=model_catalog_usage,
        description=(
            "Authorized usage controls: status, limits, headroom, reservations, "
            "receipts, adapter capabilities, and privileged import/correct/override/reset. "
            "Reads never reserve, probe, refresh, or invoke."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": usage_operation_enum,
                },
                "scope_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 256,
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": MAX_USAGE_PAGE_SIZE,
                    "default": 50,
                },
                "cursor": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 4096,
                },
                "dimension": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                },
                "state": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                },
                "expected_usage_revision": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 256,
                },
                "authorities": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                },
                "read_detail": {
                    "type": "boolean",
                    "default": False,
                    "description": "Grant exact account/cost/endpoint pseudonym detail.",
                },
                "admin": {
                    "type": "boolean",
                    "default": False,
                    "description": "Grant administrative mutation authority.",
                },
                "idempotency_key": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                },
                "lease_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                },
                "fence": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 9223372036854775807,
                },
                "expected_effects": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                    },
                },
                "actor": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                },
                "source": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                    "default": "operator",
                },
                "kind": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                },
                "units": {
                    "type": "object",
                    "maxProperties": 32,
                    "additionalProperties": True,
                },
                "limits": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "maxProperties": 32,
                        "additionalProperties": True,
                    },
                },
                "limits_update": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {
                        "type": "object",
                        "maxProperties": 32,
                        "additionalProperties": True,
                    },
                },
                "supersedes_event_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                },
                "reservation_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                },
                "reason": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                },
                "cooldown_until": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 64,
                },
                "observation_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                },
                "adapter_id": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 128,
                },
                "reason_codes": {
                    "type": "array",
                    "maxItems": 32,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 64,
                    },
                },
            },
            "required": ["operation"],
            "additionalProperties": False,
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools", "usage", "bounded"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_catalog_usage_metrics",
        func=model_catalog_usage_metrics,
        description=(
            "Export event-derived low-cardinality usage metrics without "
            "request/credential/tenant/alias/model/URL label cardinality."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "authorities": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 128,
                    },
                },
            },
            "required": [],
            "additionalProperties": False,
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools", "usage", "metrics", "read-only"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_search",
        func=model_search,
        description="Search for models matching a query string and optional task filter.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Search query string."},
                "task_filter": {
                    "type": "string",
                    "description": "Optional task type filter (e.g., 'text-generation').",
                },
                "limit": {
                    "type": "integer",
                    "description": "Maximum number of results to return.",
                    "default": 10,
                },
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_recommend",
        func=model_recommend,
        description="Recommend models for a task type and hardware configuration.",
        input_schema={
            "type": "object",
            "properties": {
                "task_type": {"type": "string", "description": "Task type (e.g., 'text-generation')."},
                "hardware": {
                    "type": "string",
                    "description": "Target hardware (e.g., 'cpu', 'cuda').",
                    "default": "cpu",
                },
                "max_size_gb": {
                    "type": "number",
                    "description": "Optional maximum model size in GB.",
                },
            },
            "required": ["task_type"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_get_details",
        func=model_get_details,
        description="Get detailed information about a specific model.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "Model identifier or HuggingFace ID."}
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_get_stats",
        func=model_get_stats,
        description="Get aggregate statistics about available models.",
        input_schema={"type": "object", "properties": {}, "required": []},
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_list_served",
        func=model_list_served,
        description="List models currently exposed by configured inference servers.",
        input_schema={
            "type": "object",
            "properties": {
                "endpoint_url": {"type": "string"},
                "timeout": {"type": "number", "default": 2.0, "minimum": 0.1},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools", "serving"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_get_served",
        func=model_get_served,
        description="Get live serving information for a model ID.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string"},
                "endpoint_url": {"type": "string"},
                "timeout": {"type": "number", "default": 2.0, "minimum": 0.1},
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools", "serving"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_list_hf_inference",
        func=model_list_hf_inference,
        description="List HuggingFace inference API compatible models.",
        input_schema={
            "type": "object",
            "properties": {
                "model_kind": {
                    "type": "string",
                    "description": "Optional model kind filter.",
                }
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_get_hf_metadata",
        func=model_get_hf_metadata,
        description="Get metadata for a HuggingFace inference model.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "HuggingFace model ID."}
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_build_hf_ipld_document",
        func=model_build_hf_ipld_document,
        description="Build an IPLD document for a HuggingFace model.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "HuggingFace model ID."},
                "include_config": {
                    "type": "boolean",
                    "description": "Include model configuration.",
                    "default": True,
                },
                "include_tokenizer": {
                    "type": "boolean",
                    "description": "Include tokenizer configuration.",
                    "default": True,
                },
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_get_hf_ipld_cid",
        func=model_get_hf_ipld_cid,
        description="Get the IPLD CID for a HuggingFace model document.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "HuggingFace model ID."},
                "include_config": {"type": "boolean", "default": True},
                "include_tokenizer": {"type": "boolean", "default": True},
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_publish_hf_ipld_to_ipfs",
        func=model_publish_hf_ipld_to_ipfs,
        description="Publish a HuggingFace model IPLD document to IPFS.",
        input_schema={
            "type": "object",
            "properties": {
                "model_id": {"type": "string", "description": "HuggingFace model ID."},
                "pin": {
                    "type": "boolean",
                    "description": "Pin the published document.",
                    "default": True,
                },
            },
            "required": ["model_id"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
    manager.register_tool(
        category="model_tools",
        name="model_load_hf_ipld_from_ipfs",
        func=model_load_hf_ipld_from_ipfs,
        description="Load a HuggingFace model IPLD document from IPFS by CID.",
        input_schema={
            "type": "object",
            "properties": {
                "cid": {"type": "string", "description": "IPFS content identifier."}
            },
            "required": ["cid"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "model-tools"],
    )
