"""Health and usage status for API LLM backends. Never sends prompts."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from typing import Any, Mapping, Optional

from .duckdb_store import AllocationStore, get_allocation_store
from .paths import API_PROVIDERS, provider_path_metadata


def _env_present(names: object, environ: Mapping[str, str]) -> bool:
    for name in names or ():
        if str(environ.get(str(name)) or "").strip():
            return True
    return False


def _remaining_ok(remaining: object, *, minimum: int = 1) -> bool:
    if remaining is None:
        return True
    try:
        return int(remaining) >= int(minimum)
    except (TypeError, ValueError):
        return True


def api_backends_status(
    *,
    store: Optional[AllocationStore] = None,
    environ: Optional[Mapping[str, str]] = None,
) -> dict[str, Any]:
    """Return auth, health, remaining tokens, and DuckDB stats for API backends."""

    env = os.environ if environ is None else environ
    active = store or get_allocation_store()
    names = sorted(API_PROVIDERS)
    stats: dict[str, Any] = {}
    try:
        stats = active.provider_stats(names, path="api")
    except Exception:
        stats = {}

    backends: dict[str, Any] = {}
    ready: list[str] = []
    missing: list[str] = []
    degraded: list[str] = []
    exhausted: list[str] = []

    for provider in names:
        meta = provider_path_metadata(provider)
        authenticated = _env_present(meta.get("auth_env"), env)
        row = dict(stats.get(provider) or {})
        remaining_tokens = row.get("remaining_tokens")
        remaining_requests = row.get("remaining_requests")
        last_error = str(row.get("last_error_kind") or "")
        tokens_ok = _remaining_ok(remaining_tokens) and _remaining_ok(remaining_requests)
        is_degraded = last_error in {"billing", "authentication"} or int(
            row.get("recent_fail") or 0
        ) >= 3
        if not authenticated:
            label = "unauthenticated"
        elif is_degraded:
            label = "degraded"
        elif not tokens_ok:
            label = "exhausted"
        else:
            label = "ready"
        snapshot = {
            "provider": provider,
            "display_name": str(meta.get("display_name") or provider),
            "status": label,
            "authenticated": authenticated,
            "ready": label == "ready",
            "tokens_ok": tokens_ok,
            "stats": {
                "recent_calls": int(row.get("recent_calls") or 0),
                "recent_ok": int(row.get("recent_ok") or 0),
                "recent_fail": int(row.get("recent_fail") or 0),
                "spend_1d_usd": float(row.get("spend_1d") or 0.0),
                "avg_latency_ms": float(row.get("avg_latency_ms") or 0.0),
                "tokens_1m": int(row.get("tokens_1m") or 0),
                "last_error_kind": last_error,
                "remaining_requests": remaining_requests,
                "remaining_tokens": remaining_tokens,
            },
        }
        backends[provider] = snapshot
        if label == "ready":
            ready.append(provider)
        elif label == "unauthenticated":
            missing.append(provider)
        elif label == "degraded":
            degraded.append(provider)
        else:
            exhausted.append(provider)
        try:
            active.upsert_health(
                provider,
                protocol="http",
                available=authenticated,
                authenticated=authenticated,
                ready=label == "ready",
                reason=label if label != "ready" else last_error,
            )
        except Exception:
            pass

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "degraded": degraded,
        "exhausted": exhausted,
        "unauthenticated": missing,
        "backends": backends,
    }


def render_api_backends_status(report: Optional[Mapping[str, Any]] = None) -> str:
    payload = dict(report or api_backends_status())
    lines = [
        f"API backends status ({payload.get('generated_at') or ''})",
        f"ready={len(payload.get('ready') or [])} "
        f"degraded={len(payload.get('degraded') or [])} "
        f"exhausted={len(payload.get('exhausted') or [])} "
        f"unauthenticated={len(payload.get('unauthenticated') or [])}",
        "",
        f"{'backend':<18} {'status':<18} {'ok/fail':<11} {'remain tok':<12}",
        "-" * 64,
    ]
    backends = payload.get("backends") if isinstance(payload.get("backends"), dict) else {}
    for name in sorted(backends):
        row = backends[name]
        stats = row.get("stats") if isinstance(row.get("stats"), dict) else {}
        remain = stats.get("remaining_tokens")
        remain_s = "-" if remain is None else str(remain)
        lines.append(
            f"{name:<18} {str(row.get('status') or ''):<18} "
            f"{int(stats.get('recent_ok') or 0)}/{int(stats.get('recent_fail') or 0):<7} "
            f"{remain_s:<12}"
        )
    return "\n".join(lines)


__all__ = ["api_backends_status", "render_api_backends_status"]
