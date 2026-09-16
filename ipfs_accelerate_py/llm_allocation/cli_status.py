"""Health and usage status for every CLI coding tool. Never calls a model."""

from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Optional

from .duckdb_store import AllocationStore, get_allocation_store
from .paths import CLI_PROVIDERS, provider_path_metadata

_LOGIN_FILES = {
    "muse_code": (".config/muse/auth.json",),
    "grok_cli": (".grok/auth.json",),
    "claude_code": (".claude/.credentials.json", ".claude/credentials.json"),
    "codex_cli": (".codex/auth.json", ".codex/config.toml"),
    "copilot_cli": (".copilot/hosts.yml", ".config/github-copilot/hosts.yml"),
    "gemini_cli": (".gemini/oauth_creds.json", ".config/gcloud/application_default_credentials.json"),
    "goose_cli": (".config/goose/config.yaml", ".config/goose/profiles.yaml"),
    "mistral_vibe": (".vibe/credentials.json",),
}


def _env_present(names: object, environ: Mapping[str, str]) -> bool:
    for name in names or ():
        if str(environ.get(str(name)) or "").strip():
            return True
    return False


def _login_file_present(provider: str, *, home: Path) -> bool:
    for relative in _LOGIN_FILES.get(provider, ()):
        path = home / relative
        try:
            if path.is_file() and path.stat().st_size > 2:
                return True
        except OSError:
            continue
    return False


def _auth_status(provider: str, *, environ: Mapping[str, str], home: Path) -> dict[str, Any]:
    meta = provider_path_metadata(provider)
    env_ok = _env_present(meta.get("auth_env"), environ)
    login_ok = _login_file_present(provider, home=home)
    extra = False
    if provider == "muse_code":
        try:
            from ipfs_accelerate_py.cli_runtime.providers.muse import muse_login_present

            extra = muse_login_present(environ=environ)
        except Exception:
            extra = False
    if provider == "mistral_vibe":
        try:
            from ipfs_accelerate_py.utils.mistral_vibe import mistral_vibe_auth_available

            extra = bool(mistral_vibe_auth_available(environ=dict(environ)))
        except Exception:
            extra = False
    authenticated = bool(env_ok or login_ok or extra)
    source = "none"
    if env_ok:
        source = "env"
    elif login_ok or extra:
        source = "login"
    return {"authenticated": authenticated, "auth_source": source}


def _status_label(*, installed: bool, authenticated: bool, degraded: bool) -> str:
    if not installed:
        return "missing"
    if degraded:
        return "degraded"
    if not authenticated:
        return "installed_unauthenticated"
    return "ready"


def cli_tools_status(
    *,
    store: Optional[AllocationStore] = None,
    environ: Optional[Mapping[str, str]] = None,
    home: Optional[Path] = None,
) -> dict[str, Any]:
    """Return install, auth, health, and DuckDB stats for every CLI tool."""
    env = os.environ if environ is None else environ
    home_path = Path(home) if home is not None else Path.home()
    active = store or get_allocation_store()
    names = sorted(CLI_PROVIDERS)
    stats = {}
    counts = {}
    try:
        stats = active.provider_stats(names, path="cli")
    except Exception:
        stats = {}
    try:
        counts = active.native_session_counts()
    except Exception:
        counts = {}
    tools: dict[str, Any] = {}
    try:
        from ipfs_accelerate_py.cli_runtime.installers.catalog import discover_cli_tool
    except Exception:
        discover_cli_tool = None  # type: ignore[assignment]

    ready: list[str] = []
    missing: list[str] = []
    degraded: list[str] = []
    unauthenticated: list[str] = []

    for provider in names:
        meta = provider_path_metadata(provider)
        discovered = None
        if callable(discover_cli_tool):
            try:
                discovered = discover_cli_tool(provider, environ=env)
            except Exception:
                discovered = None
        installed = bool(discovered and discovered.available)
        executable = str(getattr(discovered, "executable", "") or "")
        robustness = str(getattr(discovered, "robustness", "") or meta.get("command") or "")
        auth = _auth_status(provider, environ=env, home=home_path)
        row = dict(stats.get(provider) or {})
        last_error = str(row.get("last_error_kind") or "")
        is_degraded = last_error in {"billing", "authentication"} or int(row.get("recent_fail") or 0) >= 3
        label = _status_label(
            installed=installed,
            authenticated=bool(auth["authenticated"]),
            degraded=is_degraded and installed,
        )
        snapshot = {
            "provider": provider,
            "display_name": str(meta.get("display_name") or provider),
            "command": str(meta.get("command") or ""),
            "status": label,
            "installed": installed,
            "executable": executable,
            "authenticated": bool(auth["authenticated"]),
            "auth_source": auth["auth_source"],
            "installer_robustness": robustness,
            "ready": label == "ready",
            "native_sessions": int(counts.get(provider) or 0),
            "stats": {
                "recent_calls": int(row.get("recent_calls") or 0),
                "recent_ok": int(row.get("recent_ok") or 0),
                "recent_fail": int(row.get("recent_fail") or 0),
                "spend_1d_usd": float(row.get("spend_1d") or 0.0),
                "avg_latency_ms": float(row.get("avg_latency_ms") or 0.0),
                "tokens_1m": int(row.get("tokens_1m") or 0),
                "last_error_kind": last_error,
                "remaining_requests": row.get("remaining_requests"),
                "remaining_tokens": row.get("remaining_tokens"),
            },
        }
        tools[provider] = snapshot
        if label == "ready":
            ready.append(provider)
        elif label == "missing":
            missing.append(provider)
        elif label == "degraded":
            degraded.append(provider)
        else:
            unauthenticated.append(provider)
        try:
            active.upsert_health(
                provider,
                protocol="cli",
                available=installed,
                authenticated=bool(auth["authenticated"]),
                ready=label == "ready",
                reason=label if label != "ready" else last_error,
            )
        except Exception:
            pass

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "ready": ready,
        "degraded": degraded,
        "installed_unauthenticated": unauthenticated,
        "missing": missing,
        "tools": tools,
    }


def render_cli_tools_status(report: Optional[Mapping[str, Any]] = None) -> str:
    """Human-readable table of :func:`cli_tools_status`."""
    payload = dict(report or cli_tools_status())
    lines = [
        f"CLI tools status ({payload.get('generated_at') or ''})",
        f"ready={len(payload.get('ready') or [])} "
        f"degraded={len(payload.get('degraded') or [])} "
        f"unauthenticated={len(payload.get('installed_unauthenticated') or [])} "
        f"missing={len(payload.get('missing') or [])}",
        "",
        f"{'tool':<16} {'status':<26} {'auth':<8} {'ok/fail':<11} {'sessions':<8} {'path'}",
        "-" * 88,
    ]
    tools = payload.get("tools") if isinstance(payload.get("tools"), dict) else {}
    for name in sorted(tools):
        row = tools[name]
        stats = row.get("stats") if isinstance(row.get("stats"), dict) else {}
        ok_fail = f"{int(stats.get('recent_ok') or 0)}/{int(stats.get('recent_fail') or 0)}"
        path = str(row.get("executable") or "-")
        if len(path) > 36:
            path = "…" + path[-35:]
        lines.append(
            f"{name:<16} {str(row.get('status') or ''):<26} "
            f"{str(row.get('auth_source') or 'none'):<8} {ok_fail:<11} "
            f"{int(row.get('native_sessions') or 0):<8} {path}"
        )
    return "\n".join(lines)


__all__ = ["cli_tools_status", "render_cli_tools_status"]
