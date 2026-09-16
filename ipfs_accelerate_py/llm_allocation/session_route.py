"""Choose, resume, or migrate CLI coding-tool sessions.

Native CLI session ids are not portable across tools. Resume stays on the
same provider. Migration rebinds the allocation session to a new CLI and
starts a fresh native session there; callers may pass a handoff prompt on
the next generate_text.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Mapping, Optional, Sequence

from .allocator import rank_provider_names
from .duckdb_store import AllocationStore, get_allocation_store, sanitize_session_metadata
from .paths import (
    CLI_PROVIDERS,
    cli_session_contract,
    inject_cli_session_kwargs,
    path_for_provider,
)


def choose_cli_route(
    session_id: Optional[str] = None,
    *,
    candidates: Optional[Sequence[str]] = None,
    store: Optional[AllocationStore] = None,
) -> dict[str, Any]:
    """Pick a CLI provider for a new or existing allocation session."""
    names = [str(n) for n in (candidates or sorted(CLI_PROVIDERS)) if str(n) in CLI_PROVIDERS]
    if not names:
        names = sorted(CLI_PROVIDERS)
    active = store or get_allocation_store()
    sid = str(session_id or "").strip()
    if sid:
        try:
            resolved = active.resolve_session(sid)
            alloc = str((resolved or {}).get("allocation_session_id") or "")
            if alloc:
                sid = alloc
        except Exception:
            pass
    session = active.get_session(sid) if sid else {}
    ranked = rank_provider_names(names, store=active, path="cli", session_id=sid or None)
    preferred = str(session.get("preferred_provider") or session.get("last_provider") or "")
    cli_map = session.get("cli_sessions") if isinstance(session.get("cli_sessions"), dict) else {}
    native = ""
    if preferred in cli_map:
        native = str((cli_map.get(preferred) or {}).get("native_session_id") or "")
    resume = bool(sid and preferred and preferred in ranked and native)
    provider = preferred if resume else (ranked[0] if ranked else "")
    resume_kwargs: dict[str, Any] = {}
    if resume and provider:
        inject_cli_session_kwargs(
            provider,
            resume_kwargs,
            native_session_id=native,
            allocation_session_id=sid,
        )
    reason = "empty"
    if resume:
        reason = "sticky_resume"
    elif ranked:
        reason = "ranked_new" if not sid or not session else "ranked_existing_no_native"
    return {
        "provider": provider,
        "ranked": ranked,
        "resume": resume,
        "native_session_id": native if resume else "",
        "resume_kwargs": resume_kwargs,
        "reason": reason,
        "path": "cli",
        "session_id": sid,
        "preferred_provider": preferred,
    }


def resolve_session(
    any_session_id: str,
    *,
    store: Optional[AllocationStore] = None,
) -> dict[str, Any]:
    """Map an allocation id or any CLI native id to current provider/native state."""
    active = store or get_allocation_store()
    return dict(active.resolve_session(any_session_id) or {})


def resume_kwargs_for_session(
    session_id: str,
    provider: Optional[str] = None,
    *,
    store: Optional[AllocationStore] = None,
) -> dict[str, Any]:
    """Native resume flags for an allocation session, or empty if none."""
    route = choose_cli_route(session_id, store=store)
    if provider and str(provider) != route.get("provider"):
        return {}
    return dict(route.get("resume_kwargs") or {})


def migrate_cli_session(
    session_id: str,
    to_provider: str,
    *,
    store: Optional[AllocationStore] = None,
    handoff: str = "",
    history: str = "full",
) -> dict[str, Any]:
    """Rebind an allocation session to another CLI tool.

    Native histories do not transfer (Muse UUIDs are not Grok/Codex ids).
    Cost/token totals stay on the allocation session. The next generate_text
    with this session id starts a new native session on ``to_provider``.
    Native histories still cannot be resumed on the target CLI. A bounded,
    redacted context packet is stored on the allocation session and injected
    into the next ``generate_text`` for this session id.
    """
    target = str(to_provider or "").strip().lower().replace("-", "_")
    if target not in CLI_PROVIDERS:
        raise ValueError(f"unknown CLI provider {to_provider!r}")
    if path_for_provider(target) != "cli":
        raise ValueError(f"{target} is not a CLI routing path")
    sid = str(session_id or "").strip()
    if not sid:
        raise ValueError("session_id is required")
    active = store or get_allocation_store()
    session = active.get_session(sid)
    if not session:
        raise ValueError(f"unknown allocation session {sid!r}")
    source = str(session.get("preferred_provider") or session.get("last_provider") or "")
    cli_map = session.get("cli_sessions") if isinstance(session.get("cli_sessions"), dict) else {}
    previous_native = str((cli_map.get(source) or {}).get("native_session_id") or "")
    if source == target:
        return {
            "session_id": sid,
            "from_provider": source,
            "to_provider": target,
            "migrated": False,
            "reason": "already_on_provider",
            "native_session_portable": False,
            "handoff_pending": bool(str(handoff or "").strip()),
        }
    meta = dict(session.get("provider_metadata") or session.get("muse") or {})
    meta.update(
        {
            "migrated_from": source,
            "migrated_to": target,
            "previous_native_session_id": previous_native,
            "migrated_at": datetime.now(timezone.utc).isoformat(),
            "native_session_portable": False,
            "handoff_chars": len(str(handoff or "")),
        }
    )
    meta.pop("session_id", None)
    meta.pop("run_id", None)
    from ipfs_accelerate_py.cli_runtime.cli_handoff import collect_cli_handoff_context

    context = collect_cli_handoff_context(
        from_provider=source,
        native_session_id=previous_native,
        handoff=str(handoff or ""),
        history=str(history or "full"),
    )
    meta["history_mode"] = str(history or "full")
    meta["handoff_chars"] = len(context)
    active.rebind_session_provider(
        sid,
        target,
        metadata=sanitize_session_metadata(meta),
        handoff_context=context,
    )
    try:
        active.remap_session_aliases(
            sid,
            current_provider=target,
            previous_native_session_id=previous_native,
            previous_provider=source,
        )
    except Exception:
        pass
    return {
        "session_id": sid,
        "from_provider": source,
        "to_provider": target,
        "migrated": True,
        "reason": "rebound_with_handoff" if context else "rebound_new_native_session",
        "native_session_portable": False,
        "previous_native_session_id": previous_native,
        "resume_kwargs": {},
        "handoff_pending": bool(context),
        "handoff_chars": len(context),
        "history": str(history or "full"),
        "contract": cli_session_contract(target),
    }


__all__ = [
    "choose_cli_route",
    "migrate_cli_session",
    "resolve_session",
    "resume_kwargs_for_session",
]
