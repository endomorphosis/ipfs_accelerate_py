#!/usr/bin/env python3
"""Opt-in interop smoke tests for MCP (HTTP+P2P) and MCP++ (HTTPS+P2P).

These tests are intentionally skipped by default to avoid CI failures:
- They assume systemd (or manual) services are already running.
- They assume libp2p deps are installed for the P2P portion.

Enable locally:
    RUN_MCP_INTEROP_SMOKE=1 ./.venv/bin/python -m pytest -c test/pytest.ini -vv \
        test/integration/test_mcp_mcplusplus_interop_smoke.py

Optional overrides:
    MCP_HTTP_BASE=http://127.0.0.1:9000
    MCPPLUS_HTTP_BASE=https://127.0.0.1:9001

    MCP_P2P_MULTIADDR=/ip4/.../tcp/9100/p2p/...
    MCPPLUS_P2P_MULTIADDR=/ip4/.../tcp/9101/p2p/...

    MCP_ANNOUNCE_FILE=/path/to/task_p2p_announce.json
    MCPPLUS_ANNOUNCE_FILE=/path/to/task_p2p_announce_mcp.json
"""

from __future__ import annotations

import json
import os
import time
from typing import Any, Dict, Optional

import pytest


def _enabled() -> bool:
    return str(os.environ.get("RUN_MCP_INTEROP_SMOKE", "")).strip().lower() in {"1", "true", "yes", "on"}


def _env(name: str, default: str) -> str:
    value = os.environ.get(name)
    text = str(value).strip() if value is not None else ""
    return text or default


def _read_announce_multiaddr(path: str) -> str:
    try:
        if not path or not os.path.exists(path):
            return ""
        text = open(path, "r", encoding="utf-8").read().strip()
        info = json.loads(text) if text else {}
        if not isinstance(info, dict):
            return ""
        ma = str(info.get("multiaddr") or "").strip()
        return ma if ma and "/p2p/" in ma else ""
    except Exception:
        return ""


def _guess_standard_announce_candidates() -> list[str]:
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return [
        os.environ.get("MCP_ANNOUNCE_FILE") or "",
        os.path.join(os.getcwd(), "state", "task_p2p_announce.json"),
        os.path.join(cache_root, "ipfs_accelerate_py", "task_p2p_announce.json"),
        os.path.join(cache_root, "ipfs_datasets_py", "task_p2p_announce.json"),
        "/var/cache/ipfs-accelerate/task_p2p_announce.json",
    ]


def _guess_mcpplus_announce_candidates() -> list[str]:
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return [
        os.environ.get("MCPPLUS_ANNOUNCE_FILE") or "",
        os.path.join(os.getcwd(), "state", "task_p2p_announce_mcp.json"),
        os.path.join(cache_root, "ipfs_accelerate_py", "task_p2p_announce_mcp.json"),
        os.path.join(cache_root, "ipfs_datasets_py", "task_p2p_announce_mcp.json"),
    ]


def _resolve_multiaddr(*, env_var: str, announce_candidates: list[str]) -> str:
    explicit = str(os.environ.get(env_var, "")).strip()
    if explicit:
        return explicit

    best_ma = ""
    best_mtime = -1.0
    for path in announce_candidates:
        path = str(path).strip()
        if not path or not os.path.exists(path):
            continue
        ma = _read_announce_multiaddr(path)
        if not ma:
            continue
        try:
            mtime = float(os.path.getmtime(path))
        except Exception:
            mtime = 0.0
        if mtime > best_mtime:
            best_mtime = mtime
            best_ma = ma

    return best_ma


def _http_get_json(url: str, *, verify_tls: bool) -> Dict[str, Any]:
    import requests

    resp = requests.get(url, timeout=3, verify=verify_tls)
    resp.raise_for_status()
    payload = resp.json()
    if not isinstance(payload, dict):
        raise AssertionError(f"Expected JSON object from {url}, got {type(payload)}")
    return payload


def _http_get_text(url: str, *, verify_tls: bool) -> str:
    import requests

    resp = requests.get(url, timeout=3, verify=verify_tls)
    resp.raise_for_status()
    return resp.text or ""


@pytest.mark.integration
def test_http_mcp_and_mcpplus_reachable() -> None:
    if not _enabled():
        pytest.skip("Set RUN_MCP_INTEROP_SMOKE=1 to enable")

    mcp_http_base = _env("MCP_HTTP_BASE", "http://127.0.0.1:9000")
    mcpplus_http_base = _env("MCPPLUS_HTTP_BASE", "https://127.0.0.1:9001")

    # Standard MCP: JSON health + tools list.
    mcp_health = _http_get_json(f"{mcp_http_base}/mcp/health", verify_tls=True)
    assert str(mcp_health.get("status")) == "ok"

    mcp_tools_text = _http_get_text(f"{mcp_http_base}/mcp/tools", verify_tls=True)
    assert mcp_tools_text.strip(), "Expected non-empty /mcp/tools response"

    # MCP++: JSON health + tools list, self-signed TLS ok.
    mcpplus_health = _http_get_json(f"{mcpplus_http_base}/mcp/health", verify_tls=False)
    assert str(mcpplus_health.get("status")) == "ok"

    mcpplus_tools_text = _http_get_text(f"{mcpplus_http_base}/mcp/tools", verify_tls=False)
    assert mcpplus_tools_text.strip(), "Expected non-empty /mcp/tools response"


def _have_libp2p() -> bool:
    try:
        import libp2p  # noqa: F401

        return True
    except Exception:
        return False


async def _p2p_smoke(mcp_multiaddr: str, mcpplus_multiaddr: str) -> None:
    from ipfs_accelerate_py.p2p_tasks.client import (
        RemoteQueue,
        cache_get,
        cache_set,
        discover_status,
    )

    mcp_remote = RemoteQueue(multiaddr=mcp_multiaddr)
    mcpplus_remote = RemoteQueue(multiaddr=mcpplus_multiaddr)

    mcp_status = await discover_status(remote=mcp_remote, timeout_s=12.0, detail=True)
    assert mcp_status.get("ok"), f"MCP P2P status failed: {mcp_status}"

    mcpplus_status = await discover_status(remote=mcpplus_remote, timeout_s=12.0, detail=True)
    assert mcpplus_status.get("ok"), f"MCP++ P2P status failed: {mcpplus_status}"

    mcp_pid = str(((mcp_status.get("result") or {}) if isinstance(mcp_status.get("result"), dict) else {}).get("peer_id") or "")
    mcpplus_pid = str(((mcpplus_status.get("result") or {}) if isinstance(mcpplus_status.get("result"), dict) else {}).get("peer_id") or "")

    # Helpful invariant when both services run on the same machine.
    if mcp_pid and mcpplus_pid:
        assert mcp_pid != mcpplus_pid, "Both services reported the same peer_id; expected distinct libp2p hosts"

    # Minimal write/read to confirm RPC beyond status.
    ts = int(time.time())
    key = f"interop_smoke/{ts}"

    set_resp_1 = await cache_set(remote=mcp_remote, key=key, value={"service": "mcp", "ts": ts}, ttl_s=60.0, timeout_s=12.0)
    assert set_resp_1.get("ok"), f"MCP cache.set failed: {set_resp_1}"

    get_resp_1 = await cache_get(remote=mcp_remote, key=key, timeout_s=12.0)
    assert get_resp_1.get("ok"), f"MCP cache.get failed: {get_resp_1}"

    set_resp_2 = await cache_set(
        remote=mcpplus_remote,
        key=key,
        value={"service": "mcpplus", "ts": ts},
        ttl_s=60.0,
        timeout_s=12.0,
    )
    assert set_resp_2.get("ok"), f"MCP++ cache.set failed: {set_resp_2}"

    get_resp_2 = await cache_get(remote=mcpplus_remote, key=key, timeout_s=12.0)
    assert get_resp_2.get("ok"), f"MCP++ cache.get failed: {get_resp_2}"


@pytest.mark.integration
def test_libp2p_taskqueue_rpc_reachable_for_both_services() -> None:
    if not _enabled():
        pytest.skip("Set RUN_MCP_INTEROP_SMOKE=1 to enable")

    if not _have_libp2p():
        pytest.skip("libp2p is not installed in this environment")

    mcp_ma = _resolve_multiaddr(env_var="MCP_P2P_MULTIADDR", announce_candidates=_guess_standard_announce_candidates())
    mcpplus_ma = _resolve_multiaddr(env_var="MCPPLUS_P2P_MULTIADDR", announce_candidates=_guess_mcpplus_announce_candidates())

    if not mcp_ma:
        pytest.skip("Could not resolve MCP multiaddr. Set MCP_P2P_MULTIADDR or MCP_ANNOUNCE_FILE")
    if not mcpplus_ma:
        pytest.skip("Could not resolve MCP++ multiaddr. Set MCPPLUS_P2P_MULTIADDR or MCPPLUS_ANNOUNCE_FILE")

    import trio

    trio.run(_p2p_smoke, mcp_ma, mcpplus_ma)
