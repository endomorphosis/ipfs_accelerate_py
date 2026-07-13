"""Transport-neutral MCP++ Profile G dispatch for accelerator services.

The datasets package owns goal/risk validation while accelerator owns execution
leases.  This facade intentionally accepts an injected backend so deployments
can compose those providers without making either package a hard dependency.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping
from typing import Any, Callable

PROFILE_G_PROFILE = "mcp++/risk-scheduling"
PROFILE_G_PREFIXES = (
    "mcp++/goals/", "mcp++/tasks/", "mcp++/risk/",
    "mcp++/neighborhood/", "mcp++/schedule/",
)
PROFILE_G_METHODS = (
    "mcp++/risk/profile", "mcp++/goals/create", "mcp++/goals/get",
    "mcp++/goals/list", "mcp++/goals/decompose", "mcp++/goals/select",
    "mcp++/tasks/create", "mcp++/tasks/get", "mcp++/tasks/list",
    "mcp++/tasks/ready", "mcp++/risk/assess", "mcp++/risk/evidence",
    "mcp++/risk/history", "mcp++/neighborhood/query",
    "mcp++/neighborhood/attest", "mcp++/schedule/frontier",
    "mcp++/schedule/status", "mcp++/schedule/propose",
    "mcp++/schedule/claim", "mcp++/schedule/renew",
    "mcp++/schedule/release", "mcp++/schedule/resolve",
    "mcp++/schedule/reconcile",
)

ERROR_NUMBERS = {
    "G_INVALID_ARTIFACT": -32602, "G_CAPABILITY_NOT_NEGOTIATED": -32040,
    "G_CID_MISMATCH": -32041, "G_AUTHORITY_DENIED": -32042,
    "G_POLICY_DENIED": -32043, "G_NOT_READY": -32044,
    "G_IDEMPOTENCY_CONFLICT": -32045, "G_CLAIM_CONFLICT": -32046,
    "G_LEASE_EXPIRED": -32047, "G_QUORUM_UNAVAILABLE": -32049,
    "G_LIMIT_EXCEEDED": -32050, "G_PROVIDER_UNAVAILABLE": -32051,
    "G_EVIDENCE_INVALID": -32052, "G_REDACTED": -32053,
}


class ProfileGTransportError(RuntimeError):
    def __init__(self, code: str, message: str, *, retryable: bool = False,
                 details: Mapping[str, Any] | None = None) -> None:
        super().__init__(message)
        self.code, self.message, self.retryable = code, message, retryable
        self.details = dict(details or {})

    def data(self) -> dict[str, Any]:
        value = {"code": self.code, "message": self.message, "retryable": self.retryable}
        if self.details:
            value["details"] = self.details
        return value


def is_profile_g_method(method: Any) -> bool:
    return isinstance(method, str) and method in PROFILE_G_METHODS


def profile_metadata(provider: str = "ipfs_accelerate_py") -> dict[str, Any]:
    return {
        "version": "1.0", "artifact_schema_major": 1,
        "provider": provider, "transports": ["jsonrpc-http", "mcp+p2p"],
        "methods": list(PROFILE_G_METHODS),
    }


def _default_backend() -> Callable[[str, Mapping[str, Any]], Mapping[str, Any]] | None:
    try:
        from ipfs_datasets_py.mcp_server.profile_g_service import get_profile_g_service
        return get_profile_g_service().dispatch
    except (ImportError, ModuleNotFoundError):
        return None


class ProfileGDispatcher:
    """Validate the wire request and call the configured canonical provider."""

    def __init__(self, backend: Callable[[str, Mapping[str, Any]], Any] | None = None) -> None:
        self.backend = backend

    def dispatch(self, method: str, params: Mapping[str, Any]) -> Any:
        if method not in PROFILE_G_METHODS:
            raise ProfileGTransportError("G_INVALID_ARTIFACT", "unknown Profile G method")
        if not isinstance(params, Mapping):
            raise ProfileGTransportError("G_INVALID_ARTIFACT", "params must be an object")
        if method == "mcp++/risk/profile" and self.backend is None:
            return profile_metadata()
        backend = self.backend or _default_backend()
        if backend is None:
            raise ProfileGTransportError(
                "G_PROVIDER_UNAVAILABLE", "Profile G provider is unavailable",
                retryable=True, details={"method": method},
            )
        try:
            return backend(method, dict(params))
        except ProfileGTransportError:
            raise
        except Exception as error:
            code = str(getattr(error, "code", "G_PROVIDER_UNAVAILABLE"))
            if code not in ERROR_NUMBERS:
                code = "G_PROVIDER_UNAVAILABLE"
            message = str(getattr(error, "message", str(error)))
            details = getattr(error, "details", None)
            raise ProfileGTransportError(
                code, message, retryable=bool(getattr(error, "retryable", False)),
                details=details if isinstance(details, Mapping) else None,
            ) from error


def jsonrpc_error(request_id: Any, error: ProfileGTransportError) -> dict[str, Any]:
    return {"jsonrpc": "2.0", "id": request_id, "error": {
        "code": ERROR_NUMBERS.get(error.code, -32603),
        "message": error.message, "data": error.data(),
    }}


_LOCK = threading.Lock()
_DISPATCHER: ProfileGDispatcher | None = None


def get_profile_g_dispatcher() -> ProfileGDispatcher:
    global _DISPATCHER
    if _DISPATCHER is None:
        with _LOCK:
            if _DISPATCHER is None:
                _DISPATCHER = ProfileGDispatcher()
    return _DISPATCHER


def configure_profile_g_dispatcher(dispatcher: ProfileGDispatcher) -> None:
    global _DISPATCHER
    with _LOCK:
        _DISPATCHER = dispatcher


__all__ = [
    "ERROR_NUMBERS", "PROFILE_G_METHODS", "PROFILE_G_PREFIXES", "PROFILE_G_PROFILE",
    "ProfileGDispatcher", "ProfileGTransportError", "configure_profile_g_dispatcher",
    "get_profile_g_dispatcher", "is_profile_g_method", "jsonrpc_error", "profile_metadata",
]
