"""Multiplex API keys per backend via the secrets manager.

Raw key material lives only in :class:`SecretsManager`. DuckDB stores
fingerprints, session bindings, and per-slot cost/token/latency stats.
"""

from __future__ import annotations

import hashlib
import os
from typing import Any, Mapping, Optional

from .duckdb_store import AllocationStore, get_allocation_store
from .paths import path_for_provider

_SECRET_PREFIX = "api_key"


def fingerprint_api_key(secret: str) -> str:
    raw = str(secret or "").encode("utf-8")
    if not raw:
        return ""
    return hashlib.sha256(raw).hexdigest()[:16]


def secret_name_for(backend: str, slot_id: str) -> str:
    return f"{_SECRET_PREFIX}/{str(backend).strip().lower()}/{str(slot_id).strip()}"


def _secrets(manager: Any = None):
    if manager is not None:
        return manager
    from ipfs_accelerate_py.common.secrets_manager import get_global_secrets_manager

    return get_global_secrets_manager()


def register_api_key(
    backend: str,
    secret: str,
    *,
    slot_id: str = "",
    label: str = "",
    store: Optional[AllocationStore] = None,
    secrets: Any = None,
    persist: bool = True,
) -> dict[str, str]:
    """Store *secret* in the secrets manager and index a DuckDB slot.

    Returns fingerprint metadata only — never the raw key.
    """
    name = str(backend or "").strip().lower()
    value = str(secret or "").strip()
    if not name or not value:
        raise ValueError("backend and secret are required")
    fp = fingerprint_api_key(value)
    slot = str(slot_id or f"key-{fp[:8]}").strip()[:64]
    secret_name = secret_name_for(name, slot)
    _secrets(secrets).set_credential(secret_name, value, persist=persist)
    active = store or get_allocation_store()
    active.upsert_api_key_slot(
        backend=name,
        slot_id=slot,
        key_fingerprint=fp,
        secret_name=secret_name,
        label=label or slot,
        enabled=True,
    )
    return {
        "backend": name,
        "slot_id": slot,
        "key_fingerprint": fp,
        "secret_name": secret_name,
    }


def list_api_key_slots(
    backend: str = "",
    *,
    store: Optional[AllocationStore] = None,
) -> list[dict[str, Any]]:
    active = store or get_allocation_store()
    rows = active.list_api_key_slots(backend)
    for row in rows:
        row.pop("secret", None)
        row.pop("api_key", None)
    return rows


def bind_session_api_key(
    session_id: str,
    backend: str,
    slot_id: str,
    *,
    store: Optional[AllocationStore] = None,
) -> dict[str, str]:
    """Pin an allocation session to one key slot for a backend."""
    sid = str(session_id or "").strip()
    name = str(backend or "").strip().lower()
    slot = str(slot_id or "").strip()
    if not sid or not name or not slot:
        raise ValueError("session_id, backend, and slot_id are required")
    active = store or get_allocation_store()
    active.bind_session_api_key(sid, name, slot)
    return {"session_id": sid, "backend": name, "slot_id": slot}


def _score_slot(row: Mapping[str, Any]) -> tuple[int, float, str]:
    error = str(row.get("last_error_kind") or "")
    skip = 0 if error in {"billing", "authentication"} else 1
    spend = float(row.get("total_cost_usd") or 0.0)
    remaining = row.get("remaining_requests")
    try:
        headroom = float(remaining) if remaining is not None else 1.0
    except (TypeError, ValueError):
        headroom = 1.0
    score = headroom - spend
    return (skip, score, str(row.get("slot_id") or ""))


def select_api_key(
    backend: str,
    *,
    session_id: str = "",
    store: Optional[AllocationStore] = None,
    secrets: Any = None,
    bind: bool = True,
) -> dict[str, Any]:
    """Return slot + secret for a backend.

    Session bindings stick. Unbound sessions get the healthiest enabled slot
    (skip billing/auth, prefer remaining headroom / low spend).
    """
    name = str(backend or "").strip().lower()
    if not name:
        return {}
    active = store or get_allocation_store()
    sid = str(session_id or "").strip()
    bound = active.get_session_api_key(sid, name) if sid else {}
    slot_id = str(bound.get("slot_id") or "")
    rows = [
        row
        for row in active.list_api_key_slots(name)
        if bool(row.get("enabled", True))
    ]
    if not rows:
        return {}
    chosen = None
    if slot_id:
        chosen = next((row for row in rows if str(row.get("slot_id")) == slot_id), None)
    if chosen is None:
        ranked = sorted(rows, key=_score_slot, reverse=True)
        chosen = ranked[0] if ranked else None
        if chosen is not None and sid and bind:
            active.bind_session_api_key(sid, name, str(chosen.get("slot_id")))
    if chosen is None:
        return {}
    slot = str(chosen.get("slot_id") or "")
    secret_name = secret_name_for(name, slot)
    secret = _secrets(secrets).get_credential(secret_name) or ""
    return {
        "backend": name,
        "slot_id": slot,
        "key_fingerprint": str(chosen.get("key_fingerprint") or ""),
        "label": str(chosen.get("label") or slot),
        "secret": secret,
        "secret_name": secret_name,
        "bound": bool(slot_id),
    }


def apply_api_key_to_kwargs(
    provider: str,
    kwargs: dict[str, Any],
    *,
    session_id: str = "",
    store: Optional[AllocationStore] = None,
    secrets: Any = None,
) -> dict[str, Any]:
    """Inject ``api_key`` for API backends when a slot is available. Never logs it."""
    if str(kwargs.get("api_key") or "").strip():
        return kwargs
    if path_for_provider(provider) != "api" and provider not in {
        "muse_code",
        "grok_cli",
        "claude_code",
        "gemini_cli",
        "mistral_vibe",
    }:
        return kwargs
    selected = select_api_key(
        provider, session_id=session_id, store=store, secrets=secrets
    )
    secret = str(selected.get("secret") or "")
    if not secret:
        return kwargs
    kwargs["api_key"] = secret
    kwargs["_api_key_slot"] = selected.get("slot_id")
    kwargs["_api_key_fingerprint"] = selected.get("key_fingerprint")
    return kwargs


__all__ = [
    "apply_api_key_to_kwargs",
    "bind_session_api_key",
    "fingerprint_api_key",
    "list_api_key_slots",
    "register_api_key",
    "secret_name_for",
    "select_api_key",
]
