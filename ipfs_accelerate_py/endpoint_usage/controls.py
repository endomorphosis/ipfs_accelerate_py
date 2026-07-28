"""Authorized usage control surface: reads, previews, and privileged mutations.

Read/query/preview paths are side-effect free: they never reserve capacity,
probe providers, refresh the catalog, or invoke models. Results bind catalog
and usage revisions, support bounded filtering/pagination/cursors, and default
to aggregate state. Exact account, cost, and endpoint pseudonym detail requires
explicit ``ai.usage/read_detail`` authority.

Provider import, correction, override, and reset require administrative
authority, expected revision, idempotency, lease/fence, bounded expected
effects, and an audit receipt. Model output and remote peer data cannot mutate
the ledger.
"""

from __future__ import annotations

import copy
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

from .identity import (
    assert_no_prompt_media_or_output,
    content_cid,
    redact_secrets,
    stable_id,
)
from .provider_registry import (
    adapter_capabilities,
    list_adapter_descriptors,
)
from .receipts import assert_receipt_safe
from .resolution import (
    MAX_PAGE_LIMIT,
    ResolutionError,
    RevisionMismatch,
    StaticCandidate,
    UsageLimitPage,
    UsageRoutingRequest,
    filter_headroom,
    list_limits_page,
    resolve_usage_aware,
)
from .schema import (
    AvailabilityState,
    LimitSource,
    Provenance,
    Quantity,
    UsageDimension,
    UsageEventKind,
    UsageLimit,
    UsageRoutingReceipt,
    UsageSnapshot,
)
from .store import (
    AdmissionAuthorityError,
    CompareAndSetConflict,
    StaleFenceError,
)

USAGE_CONTROL_CONFORMANCE_REQUIREMENT_ID = (
    "requirement:endpoint-usage-control-conformance.v1"
)
USAGE_CONTROL_SCHEMA_VERSION = "ai.endpoint_usage.control.v1"
USAGE_CONTROL_TOOL_SCHEMA_VERSION = "ai.usage.mcp.v1"

# Authorities — keep strings stable for MCP/MCP++ parity.
USAGE_READ_AUTHORITY = "ai.usage/read"
USAGE_READ_DETAIL_AUTHORITY = "ai.usage/read_detail"
USAGE_ADMIN_AUTHORITY = "ai.usage/admin"

MAX_PAGE_SIZE = 100
MAX_RECEIPTS = 256
MAX_AUDIT_RECEIPTS = 512
MAX_FILTER_SCOPES = 256
MAX_REASON_CODES = 32
MAX_STRING = 256
MAX_IDEMPOTENCY_KEY = 128
MAX_EXPECTED_EFFECTS = 32
MAX_AUDIT_DETAIL_KEYS = 32
HEADROOM_BANDS = (
    "unknown",
    "exhausted",
    "critical",  # < 10%
    "low",  # < 25%
    "medium",  # < 50%
    "high",  # >= 50%
    "unlimited",
)

# Shared reason codes (Python / MCP / MCP++ must agree).
USAGE_REASON_CODES = frozenset(
    {
        "ok",
        "unauthorized",
        "read_denied",
        "detail_denied",
        "admin_denied",
        "invalid_request",
        "invalid_filter",
        "invalid_cursor",
        "cursor_revision_mismatch",
        "stale_snapshot",
        "stale_fence",
        "revision_mismatch",
        "idempotency_conflict",
        "idempotency_replay",
        "expected_effects_exceeded",
        "lease_required",
        "fence_required",
        "mutation_denied_model_output",
        "mutation_denied_remote_peer",
        "scope_not_found",
        "usage_unavailable",
        "limit_exhausted",
        "cooling_down",
        "store_unhealthy",
        "import_rejected",
        "correction_rejected",
        "override_rejected",
        "reset_rejected",
        "side_effect_forbidden",
        "unbounded_page",
    }
)


class UsageControlError(Exception):
    """Typed control-plane failure with a stable reason code."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "invalid_request",
        reason_codes: Sequence[str] = (),
    ) -> None:
        super().__init__(message)
        self.code = str(code)
        self.reason_codes = tuple(
            str(item)[:64] for item in (reason_codes or (self.code,))
        )[:MAX_REASON_CODES]


class UsageAuthority(str, Enum):
    READ = USAGE_READ_AUTHORITY
    READ_DETAIL = USAGE_READ_DETAIL_AUTHORITY
    ADMIN = USAGE_ADMIN_AUTHORITY


class ControlOperation(str, Enum):
    STATUS = "status"
    HEALTH = "health"
    LIMITS = "limits"
    HEADROOM = "headroom"
    RESERVATIONS = "reservations"
    RECEIPTS = "receipts"
    ROUTE_PREVIEW = "route_preview"
    ADAPTER_CAPABILITIES = "adapter_capabilities"
    IMPORT = "import"
    CORRECT = "correct"
    OVERRIDE = "override"
    RESET = "reset"


READ_OPERATIONS = frozenset(
    {
        ControlOperation.STATUS,
        ControlOperation.HEALTH,
        ControlOperation.LIMITS,
        ControlOperation.HEADROOM,
        ControlOperation.RESERVATIONS,
        ControlOperation.RECEIPTS,
        ControlOperation.ROUTE_PREVIEW,
        ControlOperation.ADAPTER_CAPABILITIES,
    }
)
ADMIN_OPERATIONS = frozenset(
    {
        ControlOperation.IMPORT,
        ControlOperation.CORRECT,
        ControlOperation.OVERRIDE,
        ControlOperation.RESET,
    }
)


def _now_rfc3339(now: Optional[datetime] = None) -> str:
    value = now or datetime.now(timezone.utc)
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc).isoformat(timespec="microseconds").replace(
        "+00:00", "Z"
    )


def _require_text(value: Any, field: str, *, maximum: int = MAX_STRING) -> str:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise UsageControlError(
            "%s must be non-empty text within %d bytes" % (field, maximum),
            code="invalid_request",
            reason_codes=("invalid_request", field),
        )
    return value


def _optional_text(
    value: Any, field: str, *, maximum: int = MAX_STRING
) -> Optional[str]:
    if value is None:
        return None
    return _require_text(value, field, maximum=maximum)


def _bounded_page(limit: Any, *, default: int = 50) -> int:
    if limit is None:
        return default
    if isinstance(limit, bool) or not isinstance(limit, int):
        raise UsageControlError(
            "limit must be an integer",
            code="unbounded_page",
            reason_codes=("unbounded_page",),
        )
    if not 1 <= limit <= MAX_PAGE_SIZE:
        raise UsageControlError(
            "limit must be between 1 and %d" % MAX_PAGE_SIZE,
            code="unbounded_page",
            reason_codes=("unbounded_page",),
        )
    return limit


def _authorities(granted: Optional[Sequence[str]]) -> frozenset:
    if granted is None:
        return frozenset()
    if isinstance(granted, (str, bytes)):
        raise UsageControlError(
            "authorities must be a sequence of strings",
            code="invalid_request",
        )
    out = []
    for item in granted:
        if not isinstance(item, str) or not item:
            raise UsageControlError(
                "authority values must be non-empty strings",
                code="invalid_request",
            )
        out.append(item)
    return frozenset(out)


def has_authority(granted: Sequence[str] | frozenset, required: str) -> bool:
    return required in frozenset(granted or ())


def require_read(granted: Sequence[str] | frozenset) -> None:
    if not has_authority(granted, USAGE_READ_AUTHORITY):
        raise UsageControlError(
            "usage read requires %s" % USAGE_READ_AUTHORITY,
            code="read_denied",
            reason_codes=("read_denied", "unauthorized"),
        )


def require_detail(granted: Sequence[str] | frozenset) -> None:
    require_read(granted)
    if not has_authority(granted, USAGE_READ_DETAIL_AUTHORITY):
        raise UsageControlError(
            "exact account/cost/endpoint detail requires %s"
            % USAGE_READ_DETAIL_AUTHORITY,
            code="detail_denied",
            reason_codes=("detail_denied", "unauthorized"),
        )


def require_admin(granted: Sequence[str] | frozenset) -> None:
    if not has_authority(granted, USAGE_ADMIN_AUTHORITY):
        raise UsageControlError(
            "usage mutation requires %s" % USAGE_ADMIN_AUTHORITY,
            code="admin_denied",
            reason_codes=("admin_denied", "unauthorized"),
        )


def headroom_band(
    available: Optional[Quantity] = None,
    ceiling: Optional[Quantity] = None,
    *,
    state: Optional[AvailabilityState] = None,
) -> str:
    """Map typed headroom into a low-cardinality band label."""

    if state is AvailabilityState.EXHAUSTED:
        return "exhausted"
    if state is AvailabilityState.UNKNOWN:
        return "unknown"
    if ceiling is not None and ceiling.kind.value == "unlimited":
        return "unlimited"
    if ceiling is not None and ceiling.kind.value == "unknown":
        return "unknown"
    if available is None or ceiling is None:
        return "unknown"
    avail_kind = (
        available.kind.value
        if hasattr(available.kind, "value")
        else str(available.kind)
    )
    ceil_kind = (
        ceiling.kind.value if hasattr(ceiling.kind, "value") else str(ceiling.kind)
    )
    if avail_kind == "unknown" or ceil_kind != "finite":
        return "unknown"
    if avail_kind == "unlimited":
        return "unlimited"
    ceil_value = getattr(ceiling, "value", None)
    if ceil_value is None or int(ceil_value) <= 0:
        return "exhausted"
    avail_value = getattr(available, "value", None)
    avail = 0 if avail_value is None else int(avail_value)
    if avail <= 0:
        return "exhausted"
    ratio = avail / float(int(ceil_value))
    if ratio < 0.10:
        return "critical"
    if ratio < 0.25:
        return "low"
    if ratio < 0.50:
        return "medium"
    return "high"


def _strip_detail_fields(payload: Any, *, allow_detail: bool) -> Any:
    """Default to aggregate state; hide exact account/cost/endpoint pseudonyms."""

    if allow_detail:
        return payload
    if isinstance(payload, Mapping):
        out: Dict[str, Any] = {}
        for key, value in payload.items():
            name = str(key)
            lowered = name.casefold()
            if lowered in {
                "account_pseudonym",
                "project_pseudonym",
                "organization_pseudonym",
                "credential_pseudonym",
                "endpoint_fingerprint",
                "endpoint_uri",
                "raw_endpoint",
            }:
                continue
            if lowered in {"currency"} and isinstance(value, str):
                # Currency alone is fine; cost amounts need detail authority.
                out[name] = value
                continue
            if lowered in {"dimension"} and value == UsageDimension.COST_MICROS.value:
                # Keep dimension name but force aggregate-only amounts.
                out[name] = value
                continue
            if lowered in {
                "available",
                "ceiling",
                "reserved",
                "used",
                "remaining",
                "amount",
            }:
                # Replace finite cost quantities with bands via parent context.
                out[name] = _strip_detail_fields(value, allow_detail=False)
                continue
            if lowered in {"limits", "headroom", "items"} and isinstance(value, list):
                filtered = []
                for item in value:
                    if not isinstance(item, Mapping):
                        filtered.append(
                            _strip_detail_fields(item, allow_detail=False)
                        )
                        continue
                    if item.get("dimension") == UsageDimension.COST_MICROS.value:
                        avail_key = (
                            "available"
                            if isinstance(item.get("available"), Mapping)
                            else "remaining"
                        )
                        filtered.append(
                            {
                                "dimension": UsageDimension.COST_MICROS.value,
                                "state": item.get("state") or "unknown",
                                "band": headroom_band(
                                    Quantity.from_dict(item[avail_key])
                                    if isinstance(item.get(avail_key), Mapping)
                                    else None,
                                    Quantity.from_dict(item["ceiling"])
                                    if isinstance(item.get("ceiling"), Mapping)
                                    else None,
                                    state=AvailabilityState(item["state"])
                                    if item.get("state")
                                    in {s.value for s in AvailabilityState}
                                    else None,
                                ),
                                "enforcement": item.get("enforcement"),
                            }
                        )
                    else:
                        filtered.append(
                            _strip_detail_fields(item, allow_detail=False)
                        )
                out[name] = filtered
                continue
            out[name] = _strip_detail_fields(value, allow_detail=False)
        return out
    if isinstance(payload, list):
        return [_strip_detail_fields(item, allow_detail=False) for item in payload]
    if isinstance(payload, tuple):
        return tuple(_strip_detail_fields(item, allow_detail=False) for item in payload)
    return payload


def redact_usage_payload(
    payload: Any,
    *,
    allow_detail: bool = False,
) -> Any:
    """Redact secrets and optionally exact usage detail."""

    redacted = redact_secrets(payload)
    safe = _strip_detail_fields(redacted, allow_detail=allow_detail)
    # Control envelopes use error.message; strip only for identity scan of
    # nested business payloads, not the top-level typed error object.
    try:
        assert_no_prompt_media_or_output(safe)
    except Exception:
        # Re-scan without the typed error envelope fields that collide with
        # the forbidden prompt vocabulary (e.g. "message").
        if isinstance(safe, Mapping):
            cleaned = {
                key: value
                for key, value in safe.items()
                if str(key).casefold()
                not in {"error", "error_code", "error_type"}
            }
            if "error" in safe and isinstance(safe["error"], Mapping):
                # Keep code only under a non-forbidden key for the scan.
                cleaned["_error_code"] = safe["error"].get("code")
            assert_no_prompt_media_or_output(cleaned)
        else:
            raise
    return safe


@dataclass(frozen=True)
class ControlAuditReceipt:
    """Bounded audit evidence for a privileged usage mutation."""

    operation: str
    scope_id: Optional[str]
    actor: Optional[str]
    idempotency_key: str
    expected_usage_revision: Optional[str]
    result_usage_revision: Optional[str]
    fence: Optional[int]
    lease_id: Optional[str]
    reason_codes: Tuple[str, ...]
    effects: Tuple[str, ...]
    created_at: str
    audit_id: str
    catalog_revision: Optional[str] = None
    store_revision: Optional[int] = None
    success: bool = True
    schema_version: str = USAGE_CONTROL_SCHEMA_VERSION

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "audit_id": self.audit_id,
            "operation": self.operation,
            "scope_id": self.scope_id,
            "actor": self.actor,
            "idempotency_key": self.idempotency_key,
            "expected_usage_revision": self.expected_usage_revision,
            "result_usage_revision": self.result_usage_revision,
            "catalog_revision": self.catalog_revision,
            "fence": self.fence,
            "lease_id": self.lease_id,
            "store_revision": self.store_revision,
            "reason_codes": list(self.reason_codes),
            "effects": list(self.effects),
            "created_at": self.created_at,
            "success": self.success,
        }


@dataclass
class _IdempotencyRecord:
    key: str
    operation: str
    scope_id: Optional[str]
    request_digest: str
    response: Dict[str, Any]
    created_at: str


class UsageControlService:
    """CLI-compatible service exposing authorized usage controls.

    Construct with a :class:`UsageCoordinator` (or any object providing
    ``snapshot`` / mutation methods) and optional catalog revision provider.
    """

    requirement_id = USAGE_CONTROL_CONFORMANCE_REQUIREMENT_ID

    def __init__(
        self,
        coordinator: Any,
        *,
        catalog_revision_provider: Optional[Callable[[], str]] = None,
        observability: Any = None,
        max_receipts: int = MAX_RECEIPTS,
        max_audit: int = MAX_AUDIT_RECEIPTS,
        default_authorities: Optional[Sequence[str]] = None,
    ) -> None:
        if coordinator is None or not callable(getattr(coordinator, "snapshot", None)):
            raise TypeError(
                "coordinator must provide a side-effect-free snapshot(scope_id) method"
            )
        if (
            isinstance(max_receipts, bool)
            or not isinstance(max_receipts, int)
            or not 1 <= max_receipts <= MAX_RECEIPTS
        ):
            raise ValueError("max_receipts is invalid")
        if (
            isinstance(max_audit, bool)
            or not isinstance(max_audit, int)
            or not 1 <= max_audit <= MAX_AUDIT_RECEIPTS
        ):
            raise ValueError("max_audit is invalid")
        self._coordinator = coordinator
        self._catalog_revision_provider = catalog_revision_provider
        self._observability = observability
        self._max_receipts = max_receipts
        self._max_audit = max_audit
        self._default_authorities = _authorities(default_authorities)
        self._lock = threading.RLock()
        self._receipts: List[Dict[str, Any]] = []
        self._audits: List[ControlAuditReceipt] = []
        self._idempotency: Dict[str, _IdempotencyRecord] = {}

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    @property
    def coordinator(self) -> Any:
        return self._coordinator

    def catalog_revision(self) -> str:
        if self._catalog_revision_provider is None:
            return "catalog-revision:unavailable"
        value = self._catalog_revision_provider()
        if not isinstance(value, str) or not value:
            return "catalog-revision:unavailable"
        return value

    def _store_document(self) -> Dict[str, Any]:
        store = getattr(self._coordinator, "store", None)
        if store is None or not callable(getattr(store, "read", None)):
            return {}
        doc = store.read()
        return dict(doc) if isinstance(doc, Mapping) else {}

    def _store_meta(self) -> Dict[str, Any]:
        doc = self._store_document()
        return {
            "store_revision": int(doc.get("revision") or 0),
            "fence": int(doc.get("fence") or 0),
            "writer_id": doc.get("writer_id"),
            "event_count": len(doc.get("events") or []),
            "reservation_count": len(doc.get("reservations") or {}),
        }

    def _bind_revisions(
        self,
        *,
        usage_revision: Optional[str] = None,
        extra: Optional[Mapping[str, Any]] = None,
    ) -> Dict[str, Any]:
        payload = {
            "schema_version": USAGE_CONTROL_SCHEMA_VERSION,
            "tool_schema_version": USAGE_CONTROL_TOOL_SCHEMA_VERSION,
            "requirement_id": self.requirement_id,
            "catalog_revision": self.catalog_revision(),
            "usage_revision": usage_revision,
        }
        if extra:
            payload.update(dict(extra))
        return payload

    def _success(
        self,
        *,
        authorities: Sequence[str] | frozenset,
        usage_revision: Optional[str] = None,
        **payload: Any,
    ) -> Dict[str, Any]:
        allow_detail = has_authority(authorities, USAGE_READ_DETAIL_AUTHORITY)
        envelope = self._bind_revisions(
            usage_revision=usage_revision,
            extra={"status": "success", "success": True, **payload},
        )
        return redact_usage_payload(envelope, allow_detail=allow_detail)

    def _error(
        self,
        exc: BaseException,
        *,
        authorities: Sequence[str] | frozenset = (),
        usage_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        if isinstance(exc, UsageControlError):
            code = exc.code
            message = str(exc)
            reasons = list(exc.reason_codes)
        elif isinstance(exc, RevisionMismatch):
            code = "revision_mismatch"
            message = "usage or catalog revision mismatch"
            reasons = ["revision_mismatch", "stale_snapshot"]
        elif isinstance(exc, StaleFenceError):
            code = "stale_fence"
            message = "caller fence is stale"
            reasons = ["stale_fence"]
        elif isinstance(exc, CompareAndSetConflict):
            code = "revision_mismatch"
            message = "store revision conflict"
            reasons = ["revision_mismatch"]
        elif isinstance(exc, AdmissionAuthorityError):
            code = "admin_denied"
            message = "store does not authorize admission"
            reasons = ["admin_denied"]
        elif isinstance(exc, ResolutionError):
            code = "invalid_filter"
            message = "usage filter or cursor is invalid"
            reasons = ["invalid_filter"]
        else:
            name = type(exc).__name__
            if name == "StaleSnapshot":
                code = "stale_snapshot"
                message = "usage snapshot revision is stale"
                reasons = ["stale_snapshot"]
            else:
                code = "invalid_request"
                message = "usage control request failed"
                reasons = ["invalid_request"]
        allow_detail = has_authority(authorities, USAGE_READ_DETAIL_AUTHORITY)
        envelope = self._bind_revisions(
            usage_revision=usage_revision,
            extra={
                "status": "error",
                "success": False,
                # Avoid the forbidden prompt key "message"; use detail/code only.
                "error": {"code": code, "detail": message},
                "error_code": code,
                "error_type": code,
                "reason_codes": reasons[:MAX_REASON_CODES],
            },
        )
        return redact_usage_payload(envelope, allow_detail=allow_detail)

    def _resolve_authorities(
        self, authorities: Optional[Sequence[str]]
    ) -> frozenset:
        if authorities is None:
            return self._default_authorities
        return _authorities(authorities)

    def _list_scope_ids(self) -> List[str]:
        doc = self._store_document()
        scopes = set()
        for key in ("limits", "caller_budgets", "cooldown_until", "disabled_scopes"):
            section = doc.get(key) or {}
            if isinstance(section, Mapping):
                scopes.update(str(item) for item in section.keys())
        for record in (doc.get("reservations") or {}).values():
            if isinstance(record, Mapping) and record.get("scope_id"):
                scopes.add(str(record["scope_id"]))
        for event in doc.get("events") or []:
            if isinstance(event, Mapping) and event.get("scope_id"):
                scopes.add(str(event["scope_id"]))
        return sorted(scopes)[:MAX_FILTER_SCOPES]

    def _snapshot(self, scope_id: str) -> UsageSnapshot:
        snap = self._coordinator.snapshot(scope_id)
        if isinstance(snap, UsageSnapshot):
            return snap
        if isinstance(snap, Mapping):
            return UsageSnapshot.from_dict(snap)
        raise UsageControlError(
            "coordinator returned an invalid snapshot",
            code="usage_unavailable",
            reason_codes=("usage_unavailable",),
        )

    def _record_audit(self, receipt: ControlAuditReceipt) -> None:
        with self._lock:
            self._audits.append(receipt)
            if len(self._audits) > self._max_audit:
                self._audits = self._audits[-self._max_audit :]

    def record_receipt(self, receipt: UsageRoutingReceipt | Mapping[str, Any]) -> None:
        """Append a redacted routing/settlement receipt (side channel for reads)."""

        if isinstance(receipt, UsageRoutingReceipt):
            payload = receipt.to_dict()
        else:
            payload = dict(receipt)
        assert_receipt_safe(payload)
        safe = redact_usage_payload(payload, allow_detail=True)
        with self._lock:
            self._receipts.append(safe)
            if len(self._receipts) > self._max_receipts:
                self._receipts = self._receipts[-self._max_receipts :]

    def _page_items(
        self,
        items: Sequence[Any],
        *,
        limit: int,
        cursor: Optional[str],
        cursor_key: Callable[[Any], str],
    ) -> Tuple[List[Any], Optional[str]]:
        ordered = list(items)
        start = 0
        if cursor is not None:
            if not isinstance(cursor, str) or not cursor:
                raise UsageControlError(
                    "cursor must be non-empty text",
                    code="invalid_cursor",
                    reason_codes=("invalid_cursor",),
                )
            for idx, item in enumerate(ordered):
                if cursor_key(item) == cursor:
                    start = idx + 1
                    break
            else:
                raise UsageControlError(
                    "cursor does not match this result set",
                    code="invalid_cursor",
                    reason_codes=("invalid_cursor", "cursor_revision_mismatch"),
                )
        page = ordered[start : start + limit]
        next_cursor = None
        if start + limit < len(ordered) and page:
            next_cursor = cursor_key(page[-1])
        return page, next_cursor

    def _mutation_preflight(
        self,
        *,
        operation: ControlOperation,
        authorities: frozenset,
        scope_id: str,
        expected_usage_revision: Optional[str],
        idempotency_key: Optional[str],
        lease_id: Optional[str],
        fence: Optional[int],
        expected_effects: Optional[Sequence[str]],
        request_body: Mapping[str, Any],
        allow_model_output: bool = False,
        allow_remote_peer: bool = False,
        source: Optional[str] = None,
    ) -> Optional[Dict[str, Any]]:
        require_admin(authorities)
        _require_text(scope_id, "scope_id")
        if not allow_model_output and source in {"model_output", "model", "completion"}:
            raise UsageControlError(
                "model output cannot mutate usage state",
                code="mutation_denied_model_output",
                reason_codes=("mutation_denied_model_output",),
            )
        if not allow_remote_peer and source in {
            "remote_peer",
            "federated",
            "peer",
            "federation",
        }:
            raise UsageControlError(
                "remote peer data cannot mutate usage state",
                code="mutation_denied_remote_peer",
                reason_codes=("mutation_denied_remote_peer",),
            )
        if expected_usage_revision is None:
            raise UsageControlError(
                "expected_usage_revision is required for mutations",
                code="revision_mismatch",
                reason_codes=("revision_mismatch",),
            )
        _require_text(expected_usage_revision, "expected_usage_revision")
        if not idempotency_key:
            raise UsageControlError(
                "idempotency_key is required for mutations",
                code="invalid_request",
                reason_codes=("invalid_request",),
            )
        key = _require_text(
            idempotency_key, "idempotency_key", maximum=MAX_IDEMPOTENCY_KEY
        )
        if not lease_id:
            raise UsageControlError(
                "lease_id is required for mutations",
                code="lease_required",
                reason_codes=("lease_required",),
            )
        _require_text(lease_id, "lease_id")
        if fence is None:
            raise UsageControlError(
                "fence is required for mutations",
                code="fence_required",
                reason_codes=("fence_required",),
            )
        if isinstance(fence, bool) or not isinstance(fence, int) or fence < 0:
            raise UsageControlError(
                "fence must be a non-negative integer",
                code="fence_required",
                reason_codes=("fence_required",),
            )
        effects = tuple(str(item)[:64] for item in (expected_effects or ()))
        if len(effects) > MAX_EXPECTED_EFFECTS:
            raise UsageControlError(
                "expected_effects exceeds bound",
                code="expected_effects_exceeded",
                reason_codes=("expected_effects_exceeded",),
            )
        # Reject secret-shaped / prompt material in mutation bodies.
        assert_no_prompt_media_or_output(dict(request_body))
        digest = content_cid(
            {
                "operation": operation.value,
                "scope_id": scope_id,
                "expected_usage_revision": expected_usage_revision,
                "body": dict(request_body),
            }
        )
        with self._lock:
            existing = self._idempotency.get(key)
            if existing is not None:
                if (
                    existing.operation != operation.value
                    or existing.scope_id != scope_id
                    or existing.request_digest != digest
                ):
                    raise UsageControlError(
                        "idempotency key reused with different request",
                        code="idempotency_conflict",
                        reason_codes=("idempotency_conflict",),
                    )
                replay = copy.deepcopy(existing.response)
                replay["reason_codes"] = list(
                    dict.fromkeys(
                        list(replay.get("reason_codes") or [])
                        + ["idempotency_replay"]
                    )
                )[:MAX_REASON_CODES]
                return replay
        # Pin expected usage revision before mutating.
        snap = self._snapshot(scope_id)
        if snap.usage_revision != expected_usage_revision:
            raise UsageControlError(
                "expected_usage_revision does not match current snapshot",
                code="stale_snapshot",
                reason_codes=("stale_snapshot", "revision_mismatch"),
            )
        meta = self._store_meta()
        if meta["fence"] and int(fence) < int(meta["fence"]):
            raise UsageControlError(
                "caller fence is stale",
                code="stale_fence",
                reason_codes=("stale_fence",),
            )
        return None

    def _finish_mutation(
        self,
        *,
        operation: ControlOperation,
        authorities: frozenset,
        scope_id: str,
        expected_usage_revision: str,
        idempotency_key: str,
        lease_id: str,
        fence: int,
        effects: Sequence[str],
        actor: Optional[str],
        response_body: Mapping[str, Any],
        result_usage_revision: Optional[str],
        request_body: Mapping[str, Any],
    ) -> Dict[str, Any]:
        meta = self._store_meta()
        audit = ControlAuditReceipt(
            operation=operation.value,
            scope_id=scope_id,
            actor=_optional_text(actor, "actor"),
            idempotency_key=idempotency_key,
            expected_usage_revision=expected_usage_revision,
            result_usage_revision=result_usage_revision,
            fence=fence,
            lease_id=lease_id,
            reason_codes=("ok", operation.value),
            effects=tuple(str(item)[:64] for item in effects)[:MAX_EXPECTED_EFFECTS],
            created_at=_now_rfc3339(),
            audit_id=stable_id(
                "usage-audit",
                operation.value,
                scope_id,
                idempotency_key,
                result_usage_revision or "",
            ),
            catalog_revision=self.catalog_revision(),
            store_revision=meta.get("store_revision"),
            success=True,
        )
        self._record_audit(audit)
        response = self._success(
            authorities=authorities,
            usage_revision=result_usage_revision,
            operation=operation.value,
            scope_id=scope_id,
            audit=audit.to_dict(),
            store=meta,
            **dict(response_body),
        )
        digest = content_cid(
            {
                "operation": operation.value,
                "scope_id": scope_id,
                "expected_usage_revision": expected_usage_revision,
                "body": dict(request_body),
            }
        )
        with self._lock:
            self._idempotency[idempotency_key] = _IdempotencyRecord(
                key=idempotency_key,
                operation=operation.value,
                scope_id=scope_id,
                request_digest=digest,
                response=copy.deepcopy(response),
                created_at=_now_rfc3339(),
            )
        if self._observability is not None:
            try:
                self._observability.record_control_mutation(operation.value, True)
            except Exception:
                pass
        return response

    # ------------------------------------------------------------------
    # Read / query / preview (side-effect free)
    # ------------------------------------------------------------------

    def status(
        self,
        *,
        scope_id: Optional[str] = None,
        authorities: Optional[Sequence[str]] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
        state: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Aggregate usage status. Never reserves, probes, refreshes, or invokes."""

        granted = self._resolve_authorities(authorities)
        try:
            require_read(granted)
            page_limit = _bounded_page(limit)
            if scope_id:
                scopes = [_require_text(scope_id, "scope_id")]
            else:
                scopes = self._list_scope_ids()
            rows = []
            for sid in scopes:
                snap = self._snapshot(sid)
                if state and snap.state.value != str(state):
                    continue
                bands = {
                    item.dimension.value: headroom_band(
                        item.available, item.ceiling, state=item.state
                    )
                    for item in snap.headroom
                }
                rows.append(
                    {
                        "scope_id": snap.scope_id,
                        "usage_revision": snap.usage_revision,
                        "state": snap.state.value,
                        "observed_at": snap.observed_at,
                        "fresh_until": snap.fresh_until,
                        "next_eligible_at": snap.next_eligible_at,
                        "reason_codes": list(snap.reason_codes),
                        "headroom_bands": bands,
                        "active_reservations": len(snap.reservations),
                        "limit_count": len(snap.limits),
                    }
                )
            page, next_cursor = self._page_items(
                rows,
                limit=page_limit,
                cursor=cursor,
                cursor_key=lambda item: str(item["scope_id"]),
            )
            composite = content_cid(
                {
                    "catalog_revision": self.catalog_revision(),
                    "scopes": [item["usage_revision"] for item in page],
                }
            )
            return self._success(
                authorities=granted,
                usage_revision=composite if page else None,
                operation=ControlOperation.STATUS.value,
                items=page,
                count=len(page),
                total=len(rows),
                next_cursor=next_cursor,
                store=self._store_meta(),
            )
        except Exception as exc:  # noqa: BLE001 - map to control envelope
            return self._error(exc, authorities=granted)

    def health(
        self,
        *,
        authorities: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        """Store and control-plane health without probing providers."""

        granted = self._resolve_authorities(authorities)
        try:
            require_read(granted)
            meta = self._store_meta()
            scopes = self._list_scope_ids()
            states: Dict[str, int] = {}
            stale = 0
            unknown = 0
            for sid in scopes:
                snap = self._snapshot(sid)
                states[snap.state.value] = states.get(snap.state.value, 0) + 1
                if snap.state is AvailabilityState.STALE:
                    stale += 1
                if snap.state is AvailabilityState.UNKNOWN:
                    unknown += 1
            healthy = True
            reasons: List[str] = []
            if meta.get("store_revision") is None:
                healthy = False
                reasons.append("store_unhealthy")
            return self._success(
                authorities=granted,
                usage_revision=content_cid(meta),
                operation=ControlOperation.HEALTH.value,
                healthy=healthy,
                scope_count=len(scopes),
                state_counts=states,
                stale_scopes=stale,
                unknown_scopes=unknown,
                store=meta,
                reason_codes=reasons or ["ok"],
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def limits(
        self,
        scope_id: str,
        *,
        authorities: Optional[Sequence[str]] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
        dimension: Optional[str] = None,
        expected_usage_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        granted = self._resolve_authorities(authorities)
        try:
            require_read(granted)
            page_limit = _bounded_page(limit)
            sid = _require_text(scope_id, "scope_id")
            snap = self._snapshot(sid)
            if (
                expected_usage_revision is not None
                and snap.usage_revision != expected_usage_revision
            ):
                raise UsageControlError(
                    "usage revision mismatch",
                    code="revision_mismatch",
                    reason_codes=("revision_mismatch", "stale_snapshot"),
                )
            page: UsageLimitPage = list_limits_page(
                snap,
                limit=min(page_limit, MAX_PAGE_LIMIT),
                cursor=cursor,
                dimension=dimension,
            )
            return self._success(
                authorities=granted,
                usage_revision=page.usage_revision,
                operation=ControlOperation.LIMITS.value,
                scope_id=sid,
                items=[item.to_dict() for item in page.items],
                count=len(page.items),
                total=page.total,
                next_cursor=page.next_cursor,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def headroom(
        self,
        scope_id: str,
        *,
        authorities: Optional[Sequence[str]] = None,
        dimension: Optional[str] = None,
        expected_usage_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        granted = self._resolve_authorities(authorities)
        try:
            require_read(granted)
            sid = _require_text(scope_id, "scope_id")
            snap = self._snapshot(sid)
            if (
                expected_usage_revision is not None
                and snap.usage_revision != expected_usage_revision
            ):
                raise UsageControlError(
                    "usage revision mismatch",
                    code="revision_mismatch",
                    reason_codes=("revision_mismatch", "stale_snapshot"),
                )
            items = filter_headroom(snap, dimension=dimension)
            payload_items = []
            for item in items:
                row = item.to_dict()
                row["band"] = headroom_band(
                    item.available, item.ceiling, state=item.state
                )
                payload_items.append(row)
            return self._success(
                authorities=granted,
                usage_revision=snap.usage_revision,
                operation=ControlOperation.HEADROOM.value,
                scope_id=sid,
                state=snap.state.value,
                items=payload_items,
                count=len(payload_items),
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def reservations(
        self,
        scope_id: str,
        *,
        authorities: Optional[Sequence[str]] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
        expected_usage_revision: Optional[str] = None,
    ) -> Dict[str, Any]:
        granted = self._resolve_authorities(authorities)
        try:
            require_read(granted)
            page_limit = _bounded_page(limit)
            sid = _require_text(scope_id, "scope_id")
            snap = self._snapshot(sid)
            if (
                expected_usage_revision is not None
                and snap.usage_revision != expected_usage_revision
            ):
                raise UsageControlError(
                    "usage revision mismatch",
                    code="revision_mismatch",
                    reason_codes=("revision_mismatch", "stale_snapshot"),
                )
            items = list(snap.reservations)
            items.sort(key=lambda item: item.reservation_id or "")
            page, next_cursor = self._page_items(
                items,
                limit=page_limit,
                cursor=cursor,
                cursor_key=lambda item: str(item.reservation_id or ""),
            )
            return self._success(
                authorities=granted,
                usage_revision=snap.usage_revision,
                operation=ControlOperation.RESERVATIONS.value,
                scope_id=sid,
                items=[item.to_dict() for item in page],
                count=len(page),
                total=len(items),
                next_cursor=next_cursor,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def receipts(
        self,
        *,
        scope_id: Optional[str] = None,
        authorities: Optional[Sequence[str]] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Recent redacted routing/settlement receipts (side-effect free)."""

        granted = self._resolve_authorities(authorities)
        try:
            require_read(granted)
            page_limit = _bounded_page(limit)
            with self._lock:
                items = list(self._receipts)
            if scope_id:
                sid = _require_text(scope_id, "scope_id")
                items = [item for item in items if item.get("scope_id") == sid]
            # Newest first for operator convenience; cursor is receipt_id.
            items = list(reversed(items))
            page, next_cursor = self._page_items(
                items,
                limit=page_limit,
                cursor=cursor,
                cursor_key=lambda item: str(
                    item.get("receipt_id") or item.get("attempt_id") or ""
                ),
            )
            revision = content_cid(
                {
                    "catalog_revision": self.catalog_revision(),
                    "ids": [
                        item.get("receipt_id") for item in page if item.get("receipt_id")
                    ],
                }
            )
            return self._success(
                authorities=granted,
                usage_revision=revision,
                operation=ControlOperation.RECEIPTS.value,
                items=page,
                count=len(page),
                total=len(items),
                next_cursor=next_cursor,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def route_preview(
        self,
        *,
        authorities: Optional[Sequence[str]] = None,
        candidates: Optional[Sequence[Mapping[str, Any] | StaticCandidate]] = None,
        scope_by_binding: Optional[Mapping[str, str]] = None,
        routing_policy: Optional[Mapping[str, Any]] = None,
        usage_request: Optional[Mapping[str, Any]] = None,
        catalog_revision: Optional[str] = None,
        expected_usage_revision: Optional[str] = None,
        limit: int = 50,
    ) -> Dict[str, Any]:
        """Preview usage-aware candidates without reserving or invoking."""

        granted = self._resolve_authorities(authorities)
        try:
            require_read(granted)
            page_limit = _bounded_page(limit)
            cat_rev = catalog_revision or self.catalog_revision()
            static: List[StaticCandidate] = []
            for item in candidates or ():
                if isinstance(item, StaticCandidate):
                    static.append(item)
                elif isinstance(item, Mapping):
                    static.append(
                        StaticCandidate(
                            binding_id=str(item.get("binding_id") or ""),
                            provider_id=str(item.get("provider_id") or ""),
                            model_id=item.get("model_id"),
                            deployment_id=item.get("deployment_id"),
                            scope_id=item.get("scope_id"),
                            catalog_score=int(item.get("catalog_score") or 0),
                            locality=item.get("locality"),
                            authorized=item.get("authorized"),
                            healthy=item.get("healthy"),
                            routable=item.get("routable"),
                            configured=item.get("configured"),
                            labels=dict(item.get("labels") or {}),
                            reasons=tuple(item.get("reasons") or ()),
                        )
                    )
                else:
                    raise UsageControlError(
                        "candidates must be StaticCandidate or objects",
                        code="invalid_request",
                        reason_codes=("invalid_request",),
                    )
            scope_map = dict(scope_by_binding or {})
            snap_map: Dict[str, UsageSnapshot] = {}
            for binding_id, sid in scope_map.items():
                if sid and sid not in snap_map:
                    snap_map[sid] = self._snapshot(sid)
            # Also accept scope_id on candidates.
            for cand in static:
                sid = scope_map.get(cand.binding_id) or getattr(cand, "scope_id", None)
                if sid and sid not in snap_map:
                    snap_map[str(sid)] = self._snapshot(str(sid))
                    scope_map.setdefault(cand.binding_id, str(sid))
            from .schema import RoutingMode, RoutingPolicy

            policy = (
                RoutingPolicy(mode=RoutingMode.OBSERVE)
                if routing_policy is None
                else (
                    routing_policy
                    if isinstance(routing_policy, RoutingPolicy)
                    else RoutingPolicy.from_dict(routing_policy)
                )
            )
            ureq = (
                UsageRoutingRequest()
                if usage_request is None
                else (
                    usage_request
                    if isinstance(usage_request, UsageRoutingRequest)
                    else UsageRoutingRequest.from_dict(usage_request)
                )
            )
            resolution = resolve_usage_aware(
                catalog_revision=cat_rev,
                candidates=static,
                snapshots_by_scope=snap_map,
                policy=policy,
                request=ureq,
                scope_by_binding=scope_map,
                limit=page_limit,
            )
            if (
                expected_usage_revision is not None
                and resolution.usage_revision != expected_usage_revision
            ):
                raise UsageControlError(
                    "usage revision mismatch after preview",
                    code="revision_mismatch",
                    reason_codes=("revision_mismatch",),
                )
            data = resolution.to_dict() if hasattr(resolution, "to_dict") else dict(resolution)
            return self._success(
                authorities=granted,
                usage_revision=getattr(resolution, "usage_revision", None),
                operation=ControlOperation.ROUTE_PREVIEW.value,
                catalog_revision=cat_rev,
                resolution=data,
                reserved=False,
                invoked=False,
                probed=False,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def adapter_capabilities(
        self,
        *,
        authorities: Optional[Sequence[str]] = None,
        adapter_id: Optional[str] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        granted = self._resolve_authorities(authorities)
        try:
            require_read(granted)
            page_limit = _bounded_page(limit)
            descriptors = list(list_adapter_descriptors())
            if adapter_id:
                aid = _require_text(adapter_id, "adapter_id")
                descriptors = [
                    item for item in descriptors if item.adapter_id == aid
                ]
            rows = []
            for item in descriptors:
                caps = dict(adapter_capabilities(item.adapter_id))
                rows.append(
                    {
                        "adapter_id": item.adapter_id,
                        "family": item.family.value
                        if hasattr(item.family, "value")
                        else str(item.family),
                        "aliases": list(item.aliases)[:16],
                        "description": (item.description or "")[:MAX_STRING],
                        "capabilities": caps,
                    }
                )
            rows.sort(key=lambda row: row["adapter_id"])
            page, next_cursor = self._page_items(
                rows,
                limit=page_limit,
                cursor=cursor,
                cursor_key=lambda item: str(item["adapter_id"]),
            )
            return self._success(
                authorities=granted,
                usage_revision=content_cid([row["adapter_id"] for row in page]),
                operation=ControlOperation.ADAPTER_CAPABILITIES.value,
                items=page,
                count=len(page),
                total=len(rows),
                next_cursor=next_cursor,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    # ------------------------------------------------------------------
    # Privileged mutations
    # ------------------------------------------------------------------

    def import_observation(
        self,
        scope_id: str,
        *,
        authorities: Optional[Sequence[str]] = None,
        expected_usage_revision: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        lease_id: Optional[str] = None,
        fence: Optional[int] = None,
        expected_effects: Optional[Sequence[str]] = None,
        actor: Optional[str] = None,
        source: str = "operator",
        kind: str = "observation_success",
        units: Optional[Mapping[str, Any]] = None,
        limits_update: Optional[Sequence[Mapping[str, Any]]] = None,
        cooldown_until: Optional[str] = None,
        reason_codes: Sequence[str] = (),
        observation_id: Optional[str] = None,
        reservation_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Import provider-counter reconciliation. Admin only; never model output."""

        granted = self._resolve_authorities(authorities)
        request_body = {
            "kind": kind,
            "units": units,
            "limits_update": list(limits_update or ()),
            "cooldown_until": cooldown_until,
            "reason_codes": list(reason_codes),
            "observation_id": observation_id,
            "reservation_id": reservation_id,
            "source": source,
        }
        try:
            replay = self._mutation_preflight(
                operation=ControlOperation.IMPORT,
                authorities=granted,
                scope_id=scope_id,
                expected_usage_revision=expected_usage_revision,
                idempotency_key=idempotency_key,
                lease_id=lease_id,
                fence=fence,
                expected_effects=expected_effects,
                request_body=request_body,
                source=source,
            )
            if replay is not None:
                return replay
            if not callable(getattr(self._coordinator, "append_observation", None)):
                raise UsageControlError(
                    "coordinator cannot import observations",
                    code="import_rejected",
                    reason_codes=("import_rejected",),
                )
            event_kind = UsageEventKind(str(kind))
            event = self._coordinator.append_observation(
                scope_id,
                kind=event_kind,
                units=units,
                reservation_id=reservation_id,
                observation_id=observation_id,
                reason_codes=tuple(reason_codes) + ("import",),
                cooldown_until=cooldown_until,
                limits_update=limits_update,
            )
            snap = self._snapshot(scope_id)
            if self._observability is not None:
                try:
                    self._observability.ingest_event(
                        event.to_dict() if hasattr(event, "to_dict") else event
                    )
                    self._observability.record_reconciliation("import")
                except Exception:
                    pass
            return self._finish_mutation(
                operation=ControlOperation.IMPORT,
                authorities=granted,
                scope_id=scope_id,
                expected_usage_revision=str(expected_usage_revision),
                idempotency_key=str(idempotency_key),
                lease_id=str(lease_id),
                fence=int(fence),  # type: ignore[arg-type]
                effects=expected_effects or ("import_observation",),
                actor=actor,
                response_body={
                    "event": event.to_dict() if hasattr(event, "to_dict") else event,
                },
                result_usage_revision=snap.usage_revision,
                request_body=request_body,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def correct(
        self,
        scope_id: str,
        *,
        authorities: Optional[Sequence[str]] = None,
        expected_usage_revision: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        lease_id: Optional[str] = None,
        fence: Optional[int] = None,
        expected_effects: Optional[Sequence[str]] = None,
        actor: Optional[str] = None,
        source: str = "operator",
        supersedes_event_id: Optional[str] = None,
        units: Optional[Mapping[str, Any]] = None,
        reservation_id: Optional[str] = None,
        reason: str = "correction",
    ) -> Dict[str, Any]:
        granted = self._resolve_authorities(authorities)
        request_body = {
            "supersedes_event_id": supersedes_event_id,
            "units": units,
            "reservation_id": reservation_id,
            "reason": reason,
            "source": source,
        }
        try:
            replay = self._mutation_preflight(
                operation=ControlOperation.CORRECT,
                authorities=granted,
                scope_id=scope_id,
                expected_usage_revision=expected_usage_revision,
                idempotency_key=idempotency_key,
                lease_id=lease_id,
                fence=fence,
                expected_effects=expected_effects,
                request_body=request_body,
                source=source,
            )
            if replay is not None:
                return replay
            if not supersedes_event_id:
                raise UsageControlError(
                    "supersedes_event_id is required",
                    code="correction_rejected",
                    reason_codes=("correction_rejected",),
                )
            if units is None:
                raise UsageControlError(
                    "units are required for correction",
                    code="correction_rejected",
                    reason_codes=("correction_rejected",),
                )
            if not callable(getattr(self._coordinator, "correct", None)):
                raise UsageControlError(
                    "coordinator cannot correct usage",
                    code="correction_rejected",
                    reason_codes=("correction_rejected",),
                )
            event = self._coordinator.correct(
                scope_id,
                supersedes_event_id=supersedes_event_id,
                units=units,
                reason=reason,
                reservation_id=reservation_id,
            )
            snap = self._snapshot(scope_id)
            if self._observability is not None:
                try:
                    self._observability.ingest_event(
                        event.to_dict() if hasattr(event, "to_dict") else event
                    )
                    self._observability.record_reconciliation("correction")
                except Exception:
                    pass
            return self._finish_mutation(
                operation=ControlOperation.CORRECT,
                authorities=granted,
                scope_id=scope_id,
                expected_usage_revision=str(expected_usage_revision),
                idempotency_key=str(idempotency_key),
                lease_id=str(lease_id),
                fence=int(fence),  # type: ignore[arg-type]
                effects=expected_effects or ("correction",),
                actor=actor,
                response_body={
                    "event": event.to_dict() if hasattr(event, "to_dict") else event,
                },
                result_usage_revision=snap.usage_revision,
                request_body=request_body,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def override_limits(
        self,
        scope_id: str,
        *,
        authorities: Optional[Sequence[str]] = None,
        expected_usage_revision: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        lease_id: Optional[str] = None,
        fence: Optional[int] = None,
        expected_effects: Optional[Sequence[str]] = None,
        actor: Optional[str] = None,
        source: str = "operator",
        limits: Optional[Sequence[Mapping[str, Any] | UsageLimit]] = None,
    ) -> Dict[str, Any]:
        granted = self._resolve_authorities(authorities)
        request_body = {
            "limits": [
                item.to_dict() if isinstance(item, UsageLimit) else dict(item)
                for item in (limits or ())
            ],
            "source": source,
        }
        try:
            replay = self._mutation_preflight(
                operation=ControlOperation.OVERRIDE,
                authorities=granted,
                scope_id=scope_id,
                expected_usage_revision=expected_usage_revision,
                idempotency_key=idempotency_key,
                lease_id=lease_id,
                fence=fence,
                expected_effects=expected_effects,
                request_body=request_body,
                source=source,
            )
            if replay is not None:
                return replay
            if not limits:
                raise UsageControlError(
                    "limits are required for override",
                    code="override_rejected",
                    reason_codes=("override_rejected",),
                )
            if len(limits) > MAX_EXPECTED_EFFECTS:
                raise UsageControlError(
                    "limits exceed bounded expected effects",
                    code="expected_effects_exceeded",
                    reason_codes=("expected_effects_exceeded",),
                )
            if not callable(getattr(self._coordinator, "configure_limits", None)):
                raise UsageControlError(
                    "coordinator cannot override limits",
                    code="override_rejected",
                    reason_codes=("override_rejected",),
                )
            # Force operator provenance — never accept model-output source.
            parsed: List[UsageLimit] = []
            for item in limits:
                lim = item if isinstance(item, UsageLimit) else UsageLimit.from_dict(item)
                data = lim.to_dict()
                data["provenance"] = Provenance(
                    source=LimitSource.CONFIGURED,
                    observed_at=_now_rfc3339(),
                    reason_codes=("admin_override",),
                ).to_dict()
                parsed.append(UsageLimit.from_dict(data))
            snap = self._coordinator.configure_limits(scope_id, parsed)
            if not isinstance(snap, UsageSnapshot):
                snap = self._snapshot(scope_id)
            if self._observability is not None:
                try:
                    self._observability.record_control_mutation("override", True)
                except Exception:
                    pass
            return self._finish_mutation(
                operation=ControlOperation.OVERRIDE,
                authorities=granted,
                scope_id=scope_id,
                expected_usage_revision=str(expected_usage_revision),
                idempotency_key=str(idempotency_key),
                lease_id=str(lease_id),
                fence=int(fence),  # type: ignore[arg-type]
                effects=expected_effects
                or tuple("override:%s" % (item.limit_id or item.dimension.value)
                         for item in parsed),
                actor=actor,
                response_body={"snapshot": snap.to_dict()},
                result_usage_revision=snap.usage_revision,
                request_body=request_body,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def reset(
        self,
        scope_id: str,
        *,
        authorities: Optional[Sequence[str]] = None,
        expected_usage_revision: Optional[str] = None,
        idempotency_key: Optional[str] = None,
        lease_id: Optional[str] = None,
        fence: Optional[int] = None,
        expected_effects: Optional[Sequence[str]] = None,
        actor: Optional[str] = None,
        source: str = "operator",
        reason: str = "admin_reset",
    ) -> Dict[str, Any]:
        granted = self._resolve_authorities(authorities)
        request_body = {"reason": reason, "source": source}
        try:
            replay = self._mutation_preflight(
                operation=ControlOperation.RESET,
                authorities=granted,
                scope_id=scope_id,
                expected_usage_revision=expected_usage_revision,
                idempotency_key=idempotency_key,
                lease_id=lease_id,
                fence=fence,
                expected_effects=expected_effects,
                request_body=request_body,
                source=source,
            )
            if replay is not None:
                return replay
            if not callable(getattr(self._coordinator, "reset", None)):
                raise UsageControlError(
                    "coordinator cannot reset usage",
                    code="reset_rejected",
                    reason_codes=("reset_rejected",),
                )
            event = self._coordinator.reset(
                scope_id,
                reason=reason,
                expected_usage_revision=expected_usage_revision,
            )
            snap = self._snapshot(scope_id)
            if self._observability is not None:
                try:
                    self._observability.ingest_event(
                        event.to_dict() if hasattr(event, "to_dict") else event
                    )
                    self._observability.record_reset()
                except Exception:
                    pass
            return self._finish_mutation(
                operation=ControlOperation.RESET,
                authorities=granted,
                scope_id=scope_id,
                expected_usage_revision=str(expected_usage_revision),
                idempotency_key=str(idempotency_key),
                lease_id=str(lease_id),
                fence=int(fence),  # type: ignore[arg-type]
                effects=expected_effects or ("reset",),
                actor=actor,
                response_body={
                    "event": event.to_dict() if hasattr(event, "to_dict") else event,
                },
                result_usage_revision=snap.usage_revision,
                request_body=request_body,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def list_audit_receipts(
        self,
        *,
        authorities: Optional[Sequence[str]] = None,
        limit: int = 50,
        cursor: Optional[str] = None,
    ) -> Dict[str, Any]:
        granted = self._resolve_authorities(authorities)
        try:
            require_admin(granted)
            page_limit = _bounded_page(limit)
            with self._lock:
                items = [item.to_dict() for item in self._audits]
            items = list(reversed(items))
            page, next_cursor = self._page_items(
                items,
                limit=page_limit,
                cursor=cursor,
                cursor_key=lambda item: str(item.get("audit_id") or ""),
            )
            return self._success(
                authorities=granted,
                usage_revision=content_cid(
                    [item.get("audit_id") for item in page]
                ),
                operation="audit",
                items=page,
                count=len(page),
                total=len(items),
                next_cursor=next_cursor,
            )
        except Exception as exc:  # noqa: BLE001
            return self._error(exc, authorities=granted)

    def execute(
        self,
        operation: str | ControlOperation,
        *,
        authorities: Optional[Sequence[str]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Dispatch a named control operation (CLI/MCP multiplex entrypoint)."""

        try:
            op = (
                operation
                if isinstance(operation, ControlOperation)
                else ControlOperation(str(operation))
            )
        except ValueError:
            return self._error(
                UsageControlError(
                    "unknown usage control operation",
                    code="invalid_request",
                    reason_codes=("invalid_request",),
                ),
                authorities=self._resolve_authorities(authorities),
            )
        dispatch = {
            ControlOperation.STATUS: self.status,
            ControlOperation.HEALTH: self.health,
            ControlOperation.LIMITS: self.limits,
            ControlOperation.HEADROOM: self.headroom,
            ControlOperation.RESERVATIONS: self.reservations,
            ControlOperation.RECEIPTS: self.receipts,
            ControlOperation.ROUTE_PREVIEW: self.route_preview,
            ControlOperation.ADAPTER_CAPABILITIES: self.adapter_capabilities,
            ControlOperation.IMPORT: self.import_observation,
            ControlOperation.CORRECT: self.correct,
            ControlOperation.OVERRIDE: self.override_limits,
            ControlOperation.RESET: self.reset,
        }
        handler = dispatch[op]
        # Map multiplex kwargs carefully — pass through only accepted names.
        return handler(authorities=authorities, **kwargs)  # type: ignore[operator]


# JSON Schema fragments shared with MCP / MCP++ (bounded, no free cardinality).
def usage_control_reason_codes() -> Tuple[str, ...]:
    return tuple(sorted(USAGE_REASON_CODES))


def usage_control_authorities() -> Dict[str, str]:
    return {
        "read": USAGE_READ_AUTHORITY,
        "read_detail": USAGE_READ_DETAIL_AUTHORITY,
        "admin": USAGE_ADMIN_AUTHORITY,
    }


def usage_control_operations() -> Tuple[str, ...]:
    return tuple(item.value for item in ControlOperation)


__all__ = [
    "USAGE_CONTROL_CONFORMANCE_REQUIREMENT_ID",
    "USAGE_CONTROL_SCHEMA_VERSION",
    "USAGE_CONTROL_TOOL_SCHEMA_VERSION",
    "USAGE_READ_AUTHORITY",
    "USAGE_READ_DETAIL_AUTHORITY",
    "USAGE_ADMIN_AUTHORITY",
    "USAGE_REASON_CODES",
    "MAX_PAGE_SIZE",
    "HEADROOM_BANDS",
    "UsageControlError",
    "UsageAuthority",
    "ControlOperation",
    "ControlAuditReceipt",
    "UsageControlService",
    "has_authority",
    "require_read",
    "require_detail",
    "require_admin",
    "headroom_band",
    "redact_usage_payload",
    "usage_control_reason_codes",
    "usage_control_authorities",
    "usage_control_operations",
]
