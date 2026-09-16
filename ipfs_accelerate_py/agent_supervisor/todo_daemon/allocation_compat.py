"""Bridge the agent supervisor onto the llm_allocation DuckDB schema.

Supervisor provider names (``grok``, ``claude``, …) map onto router names
(``grok_cli``, ``claude_code``, …). Opening the allocation store runs schema
migrations so new tables (sessions, aliases, API key slots, handoff) exist
before the daemon dispatches work.
"""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

from ipfs_accelerate_py.llm_allocation.paths import CLI_PROVIDERS, path_for_provider

SUPERVISOR_TO_ROUTER: dict[str, str] = {
    "grok": "grok_cli",
    "grok_cli": "grok_cli",
    "grok-cli": "grok_cli",
    "grok_build": "grok_cli",
    "xai_cli": "grok_cli",
    "codex": "codex_cli",
    "openai": "codex_cli",
    "codex_cli": "codex_cli",
    "claude": "claude_code",
    "claude_code": "claude_code",
    "claude-code": "claude_code",
    "gemini": "gemini_cli",
    "gemini_cli": "gemini_cli",
    "copilot": "copilot_cli",
    "copilot_cli": "copilot_cli",
    "goose": "goose_cli",
    "goose_cli": "goose_cli",
    "meta_spark": "goose_cli",
    "meta-spark": "goose_cli",
    "muse": "goose_cli",
    "muse-spark": "goose_cli",
    "muse_code": "muse_code",
    "muse-code": "muse_code",
    "muse_cli": "muse_code",
    "mistral": "mistral_vibe",
    "mistral_vibe": "mistral_vibe",
    "vibe": "mistral_vibe",
}


def ensure_allocation_schema() -> bool:
    """Open the allocation store so CREATE/ALTER migrations run. Fail-soft."""
    try:
        from ipfs_accelerate_py.llm_allocation import get_allocation_store

        store = get_allocation_store()
        conn = store._connect()  # noqa: SLF001 — migrate side effect
        if conn is None:
            return False
        conn.close()
        return True
    except Exception:
        return False


def map_supervisor_provider(name: str) -> str:
    key = str(name or "").strip().lower().replace("-", "_")
    return SUPERVISOR_TO_ROUTER.get(key, key)


def allocation_session_id_for_task(task_id: str) -> str:
    tid = str(task_id or "").strip()
    if not tid:
        return ""
    if tid.startswith("asup-"):
        return tid[:128]
    return f"asup-{tid}"[:128]


def supervisor_generate_kwargs(
    *,
    task_id: str = "",
    provider: str = "",
    allocation_session_id: str = "",
    allocation_path: str = "",
) -> dict[str, Any]:
    """Kwargs for ``generate_text`` that keep the supervisor on the new schema."""
    ensure_allocation_schema()
    sid = str(allocation_session_id or "").strip() or allocation_session_id_for_task(
        task_id
    )
    router_provider = map_supervisor_provider(provider)
    path = str(allocation_path or "").strip()
    if not path and router_provider:
        path = path_for_provider(router_provider)
    kwargs: dict[str, Any] = {}
    if sid:
        kwargs["allocation_session_id"] = sid
        try:
            from ipfs_accelerate_py.llm_allocation import resolve_session

            resolved = resolve_session(sid)
            alloc = str((resolved or {}).get("allocation_session_id") or sid)
            kwargs["allocation_session_id"] = alloc
            resume = (resolved or {}).get("current_native_session_id") or ""
            current = str((resolved or {}).get("current_provider") or "")
            if resume and current == router_provider:
                from ipfs_accelerate_py.llm_allocation.paths import (
                    inject_cli_session_kwargs,
                )

                inject_cli_session_kwargs(
                    router_provider,
                    kwargs,
                    native_session_id=str(resume),
                    allocation_session_id=alloc,
                )
        except Exception:
            pass
    if path:
        kwargs["allocation_path"] = path
    if router_provider:
        kwargs["provider"] = router_provider
    return kwargs


def apply_resume_argv(
    command: Sequence[str],
    *,
    provider: str,
    task_id: str = "",
    allocation_session_id: str = "",
) -> list[str]:
    """Insert native resume flags for a supervisor CLI command when mapped."""
    argv = [str(part) for part in command]
    if not argv:
        return argv
    kwargs = supervisor_generate_kwargs(
        task_id=task_id,
        provider=provider,
        allocation_session_id=allocation_session_id,
    )
    router_provider = str(kwargs.get("provider") or map_supervisor_provider(provider))
    resume_id = str(
        kwargs.get("session_id")
        or kwargs.get("resume_session_id")
        or kwargs.get("chat_session_id")
        or ""
    ).strip()
    if not resume_id:
        return argv
    if router_provider == "codex_cli" and "exec" in argv and "resume" not in argv:
        idx = argv.index("exec")
        argv[idx + 1 : idx + 1] = ["resume", resume_id]
        return argv
    if router_provider in {"grok_cli", "claude_code", "copilot_cli"} and "--resume" not in argv:
        insert_at = 1 if len(argv) > 1 else len(argv)
        argv[insert_at:insert_at] = ["--resume", resume_id]
        return argv
    if router_provider == "muse_code" and "--session-id" not in argv:
        insert_at = argv.index("exec") + 1 if "exec" in argv else min(2, len(argv))
        argv[insert_at:insert_at] = ["--session-id", resume_id]
        return argv
    return argv


def supervisor_cli_health() -> dict[str, Any]:
    """cli_tools_status keyed by both router and supervisor names."""
    ensure_allocation_schema()
    try:
        from ipfs_accelerate_py.llm_allocation import cli_tools_status

        report = cli_tools_status()
    except Exception:
        return {"tools": {}, "ready": [], "missing": []}
    tools = dict(report.get("tools") or {})
    aliased = dict(tools)
    for supervisor, router in SUPERVISOR_TO_ROUTER.items():
        if router in tools and supervisor not in aliased:
            aliased[supervisor] = dict(tools[router])
            aliased[supervisor]["router_provider"] = router
    report = dict(report)
    report["tools"] = aliased
    return report


__all__ = [
    "SUPERVISOR_TO_ROUTER",
    "allocation_session_id_for_task",
    "apply_resume_argv",
    "ensure_allocation_schema",
    "map_supervisor_provider",
    "supervisor_cli_health",
    "supervisor_generate_kwargs",
]
