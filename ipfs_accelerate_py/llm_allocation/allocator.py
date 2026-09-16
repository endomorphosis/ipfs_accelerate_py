"""Rank unpinned LLM providers from DuckDB health, spend, and remaining limits.

Empty stats preserve the caller's original order (stable tie-break). Billing
and authentication failures in the recent window are skipped when another
candidate remains.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from .duckdb_store import AllocationStore, get_allocation_store, llm_allocate_enabled
from .limits import limit_hint_for
from .paths import filter_names_for_path, normalize_routing_path

_SKIP_KINDS = frozenset({"billing", "authentication"})


def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None:
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any) -> Optional[int]:
    try:
        if value is None:
            return None
        return int(value)
    except (TypeError, ValueError):
        return None


def score_provider(
    name: str,
    stats: Mapping[str, Any],
    *,
    index: int,
    preferred_provider: str = "",
) -> tuple[int, float, int]:
    """Return (skip_rank, score, original_index). Higher score is better.

    skip_rank 1 is eligible, 0 is last-resort skip.
    """
    hint = limit_hint_for(name)
    recent_calls = _as_float(stats.get("recent_calls"))
    recent_ok = _as_float(stats.get("recent_ok"))
    recent_fail = _as_float(stats.get("recent_fail"))
    tokens_1m = _as_float(stats.get("tokens_1m"))
    spend_1d = _as_float(stats.get("spend_1d"))
    avg_latency = _as_float(stats.get("avg_latency_ms"))
    remaining_requests = _as_int(stats.get("remaining_requests"))
    remaining_tokens = _as_int(stats.get("remaining_tokens"))
    last_error = str(stats.get("last_error_kind") or "")
    ready = stats.get("ready")
    available = stats.get("available")

    skip = 1
    if ready is False or available is False:
        skip = 0
    if last_error in _SKIP_KINDS and recent_fail > 0:
        skip = 0
    if remaining_requests is not None and remaining_requests <= 0:
        skip = 0
    if remaining_tokens is not None and remaining_tokens <= 0:
        skip = 0
    tpm = hint.tpm or _as_int(stats.get("tpm"))
    if tpm and tokens_1m >= tpm:
        skip = 0

    success_rate = 1.0 if recent_calls <= 0 else recent_ok / max(1.0, recent_calls)
    score = 50.0 + 40.0 * success_rate
    if remaining_tokens is not None and remaining_tokens > 0:
        score += 20.0 * min(1.0, remaining_tokens / 100_000.0)
    if tpm and tpm > 0:
        score -= 25.0 * min(1.0, tokens_1m / float(tpm))
    rpm = hint.rpm or _as_int(stats.get("rpm"))
    if remaining_requests is not None and rpm and rpm > 0:
        score += 15.0 * min(1.0, remaining_requests / float(rpm))
    score -= min(20.0, spend_1d * 10.0)
    if avg_latency > 0:
        score -= min(15.0, avg_latency / 1000.0)
    score -= recent_fail
    if preferred_provider and name == preferred_provider and skip == 1:
        score += 25.0
    return (skip, score, -index)


def rank_provider_names(
    names: Sequence[str],
    *,
    store: Optional[AllocationStore] = None,
    path: Optional[str] = None,
    session_id: Optional[str] = None,
) -> list[str]:
    """Reorder *names* using DuckDB stats. Unknown providers keep relative order."""
    ordered = [str(name) for name in names if str(name).strip()]
    session: dict[str, Any] = {}
    preferred = ""
    resolved_path = normalize_routing_path(path)
    try:
        active = store or get_allocation_store()
        if session_id:
            session = active.get_session(str(session_id))
            preferred = str(session.get("preferred_provider") or session.get("last_provider") or "")
            if resolved_path is None:
                resolved_path = normalize_routing_path(session.get("path"))
    except Exception:
        active = None
        session = {}
    if resolved_path:
        ordered = filter_names_for_path(ordered, resolved_path)
    if len(ordered) <= 1 or not llm_allocate_enabled():
        return ordered
    if active is None:
        return ordered
    try:
        stats = active.provider_stats(ordered, path=resolved_path)
    except Exception:
        return ordered
    if not stats and not preferred:
        return ordered
    ranked = sorted(
        enumerate(ordered),
        key=lambda item: score_provider(
            item[1],
            stats.get(item[1], {}),
            index=item[0],
            preferred_provider=preferred,
        ),
        reverse=True,
    )
    return [name for _, name in ranked]
