"""MCP-protocol interop between provider packages.

Cross-package calls **must** use the MCP tool surface (``tools/call`` /
``tools_dispatch`` mediation), not direct Python imports of peer packages.

Packages in scope:

* ``ipfs_accelerate_py`` — this process's unified MCP registry
* ``ipfs_kit_py`` — peer MCP server (HTTP JSON-RPC or in-process registry)
* ``ipfs_datasets_py`` — peer MCP server

Direct imports of peer packages remain compatibility fallbacks only and are
never the preferred interop path for new contract surfaces.
"""

from __future__ import annotations

import json
import logging
import os
import urllib.error
import urllib.request
from typing import Any, Final, Mapping, MutableMapping, Optional

logger = logging.getLogger(__name__)

PACKAGE_MCP_INTEROP_INTERFACE: Final[str] = "PackageMcpInterop@1"
PATH_CLASS_MCP_PLUS_PLUS: Final[str] = "mcp_plus_plus"

# Env overrides for peer MCP JSON-RPC endpoints (tools/call).
_ENV_ENDPOINTS: Final[Mapping[str, str]] = {
    "ipfs_kit_py": "IPFS_KIT_MCP_URL",
    "ipfs_datasets_py": "IPFS_DATASETS_MCP_URL",
    "ipfs_accelerate_py": "IPFS_ACCELERATE_MCP_URL",
}


class PackageMcpInteropError(RuntimeError):
    """Fail-closed interop error (no silent peer import)."""


def package_mcp_endpoint(package_id: str) -> str:
    """Return configured MCP base URL for a package, or empty if unset."""
    key = _ENV_ENDPOINTS.get(str(package_id or "").strip())
    if not key:
        return ""
    return str(os.environ.get(key) or "").strip().rstrip("/")


def _jsonrpc_tools_call(
    endpoint: str,
    *,
    tool_name: str,
    arguments: Mapping[str, Any],
    timeout_seconds: float = 30.0,
) -> dict[str, Any]:
    """Issue MCP-style JSON-RPC ``tools/call`` against a peer endpoint."""

    url = endpoint.rstrip("/")
    # Prefer explicit tools/call routes used by accelerate/datasets FastAPI facades,
    # then JSON-RPC method-on-/mcp (same payload shape).
    if url.endswith("/mcp") or url.endswith("/rpc"):
        candidates = (f"{url}/tools/call", url)
    else:
        candidates = (
            f"{url}/mcp/tools/call",
            f"{url}/mcp",
            f"{url}/mcp/",
            url,
        )

    payload = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": str(tool_name),
            "arguments": dict(arguments or {}),
        },
    }
    body = json.dumps(payload).encode("utf-8")
    last_error = "no_endpoint_candidate"
    for candidate in candidates:
        req = urllib.request.Request(
            candidate,
            data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=timeout_seconds) as resp:
                raw = resp.read().decode("utf-8", errors="replace")
            data = json.loads(raw) if raw else {}
            if not isinstance(data, dict):
                return {
                    "ok": False,
                    "error": "invalid_jsonrpc_response",
                    "tool": tool_name,
                    "mediation": PATH_CLASS_MCP_PLUS_PLUS,
                    "endpoint": candidate,
                }
            if data.get("error"):
                return {
                    "ok": False,
                    "error": data.get("error"),
                    "tool": tool_name,
                    "mediation": PATH_CLASS_MCP_PLUS_PLUS,
                    "endpoint": candidate,
                }
            result = data.get("result", data)
            return {
                "ok": True,
                "tool": tool_name,
                "result": result,
                "mediation": PATH_CLASS_MCP_PLUS_PLUS,
                "endpoint": candidate,
            }
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError) as exc:
            last_error = f"{type(exc).__name__}: {exc}"
            continue
    return {
        "ok": False,
        "error": last_error,
        "tool": tool_name,
        "mediation": PATH_CLASS_MCP_PLUS_PLUS,
    }


async def call_package_mcp_tool(
    package_id: str,
    tool_name: str,
    arguments: Optional[Mapping[str, Any]] = None,
    *,
    local_mcp: Any = None,
    allow_local_same_package: bool = True,
) -> dict[str, Any]:
    """Call a tool on a package via MCP protocol mediation.

    Order:
    1. Same-package local registry via ``invoke_mcp_tool`` when allowed.
    2. Peer package HTTP JSON-RPC ``tools/call`` when endpoint env is set.
    3. Fail closed — never falls through to ``import peer_package``.
    """

    package = str(package_id or "").strip()
    tool = str(tool_name or "").strip()
    args = dict(arguments or {})
    if not package or not tool:
        return {
            "ok": False,
            "error": "package_id and tool_name are required",
            "mediation": PATH_CLASS_MCP_PLUS_PLUS,
        }

    # Resolve GUI/ORB/IDL aliases → preferred package MCP tool name (if bound).
    binding_meta: dict[str, Any] = {}
    try:
        from .surface_identity_bindings import resolve_tool_name_for_interop

        resolved = resolve_tool_name_for_interop(tool)
        if isinstance(resolved, dict) and resolved.get("tool_name"):
            binding_meta = {
                "surface_binding": {
                    "requested_tool_name": tool,
                    "resolved_tool_name": resolved.get("tool_name"),
                    "resolved": bool(resolved.get("resolved")),
                    "reason": resolved.get("reason"),
                    "preferred_path": resolved.get("preferred_path"),
                    "handler": resolved.get("handler"),
                }
            }
            tool = str(resolved.get("tool_name") or tool)
    except Exception as exc:  # pragma: no cover - bindings must not break interop
        logger.debug("surface identity resolve skipped: %s", exc)

    # 1) Same-package local MCP registry
    if allow_local_same_package and local_mcp is not None and package in {
        "ipfs_accelerate_py",
        "local",
        "",
    }:
        try:
            from ipfs_accelerate_py.tool_manifest import invoke_mcp_tool

            result = await invoke_mcp_tool(
                local_mcp, tool_name=tool, args=args, accelerate_instance=None
            )
            if isinstance(result, dict):
                out = dict(result)
                out.setdefault("mediation", PATH_CLASS_MCP_PLUS_PLUS)
                out.setdefault("package_id", package)
                if binding_meta:
                    out.update(binding_meta)
                return out
        except Exception as exc:  # pragma: no cover - defensive
            logger.warning("local MCP invoke failed for %s: %s", tool, exc)

    # 2) Peer HTTP MCP endpoint
    endpoint = package_mcp_endpoint(package)
    if endpoint:
        env = {
            **_jsonrpc_tools_call(endpoint, tool_name=tool, arguments=args),
            "package_id": package,
        }
        if binding_meta:
            env.update(binding_meta)
        return env

    return {
        "ok": False,
        "error": (
            f"mcp_endpoint_unavailable for package={package}; "
            f"set {_ENV_ENDPOINTS.get(package, 'PACKAGE_MCP_URL')} or pass local_mcp "
            "for same-package tools. Direct peer imports are not used for interop."
        ),
        "tool": tool,
        "package_id": package,
        "mediation": PATH_CLASS_MCP_PLUS_PLUS,
        **binding_meta,
    }


def call_package_mcp_tool_sync(
    package_id: str,
    tool_name: str,
    arguments: Optional[Mapping[str, Any]] = None,
    *,
    local_mcp: Any = None,
) -> dict[str, Any]:
    """Sync wrapper for non-async tool handlers."""
    import asyncio

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # Nested loop: run in a new loop threadlessly via asyncio.run fallback
            # is unsafe; fail closed with guidance.
            return {
                "ok": False,
                "error": "async_event_loop_running; use await call_package_mcp_tool",
                "package_id": package_id,
                "tool": tool_name,
                "mediation": PATH_CLASS_MCP_PLUS_PLUS,
            }
        return loop.run_until_complete(
            call_package_mcp_tool(
                package_id,
                tool_name,
                arguments,
                local_mcp=local_mcp,
            )
        )
    except RuntimeError:
        return asyncio.run(
            call_package_mcp_tool(
                package_id,
                tool_name,
                arguments,
                local_mcp=local_mcp,
            )
        )


def mcp_envelope_to_tool_result(envelope: Mapping[str, Any]) -> dict[str, Any]:
    """Normalize interop envelope into tool status/success shape."""
    if not isinstance(envelope, Mapping):
        return {
            "status": "error",
            "success": False,
            "error": "invalid_mcp_envelope",
            "mediation": PATH_CLASS_MCP_PLUS_PLUS,
        }
    if envelope.get("ok") is True:
        result = envelope.get("result")
        if isinstance(result, dict):
            out: MutableMapping[str, Any] = dict(result)
            out.setdefault("status", "success")
            out.setdefault("success", True)
            out["mediation"] = PATH_CLASS_MCP_PLUS_PLUS
            out["mcp_tool"] = envelope.get("tool")
            out["mcp_package"] = envelope.get("package_id")
            return dict(out)
        return {
            "status": "success",
            "success": True,
            "data": result,
            "mediation": PATH_CLASS_MCP_PLUS_PLUS,
            "mcp_tool": envelope.get("tool"),
            "mcp_package": envelope.get("package_id"),
        }
    return {
        "status": "error",
        "success": False,
        "error": envelope.get("error") or "mcp_call_failed",
        "mediation": PATH_CLASS_MCP_PLUS_PLUS,
        "mcp_tool": envelope.get("tool"),
        "mcp_package": envelope.get("package_id"),
    }


def resolve_tool_via_surface_bindings(tool_name: str) -> dict[str, Any]:
    """Public helper: resolve a tool name through surface identity bindings."""
    try:
        from .surface_identity_bindings import resolve_tool_name_for_interop

        return dict(resolve_tool_name_for_interop(tool_name) or {})
    except Exception as exc:  # pragma: no cover
        return {
            "tool_name": str(tool_name or ""),
            "resolved": False,
            "reason": f"resolve_error:{type(exc).__name__}",
            "mediation": PATH_CLASS_MCP_PLUS_PLUS,
        }


__all__ = [
    "PACKAGE_MCP_INTEROP_INTERFACE",
    "PATH_CLASS_MCP_PLUS_PLUS",
    "PackageMcpInteropError",
    "call_package_mcp_tool",
    "call_package_mcp_tool_sync",
    "resolve_tool_via_surface_bindings",
    "mcp_envelope_to_tool_result",
    "package_mcp_endpoint",
]
