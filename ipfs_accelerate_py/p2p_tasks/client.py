"""libp2p client for the TaskQueue RPC service.

Supports:
- Explicit dialing by multiaddr.
- Bootstrap dialing using a configured list of peers.
- LAN discovery via mDNS (fallback when no multiaddr is provided).

Environment:
- IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_PEERS (comma-separated multiaddrs)
- IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_DIAL (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_DIAL
    - when enabled, the client will try to speak the TaskQueue protocol to configured bootstrap peers
    - when disabled, bootstrap peers are used only for DHT routing-table seeding
- IPFS_DATASETS_PY_TASK_P2P_DISCOVERY_TIMEOUT_S (compat, default: 5) / IPFS_ACCELERATE_PY_TASK_P2P_DISCOVERY_TIMEOUT_S
- IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT (compat, default: 9710)
    / IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT (used for mDNS)
- IPFS_DATASETS_PY_TASK_P2P_DHT (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_DHT
- IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS (compat, default: 1) / IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS
- IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS (compat, default: ipfs-accelerate-task-queue)
    / IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS
- IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE (compat) / IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE
    - unset: read default XDG cache announce file(s)
    - set to a path: read announce JSON from that path
    - set to 0/false/no: disable announce-file dialing
- IPFS_DATASETS_PY_TASK_P2P_SUBMIT_RETRIES / IPFS_ACCELERATE_PY_TASK_P2P_SUBMIT_RETRIES
    - retry attempts for transient submit transport failures (default: 2)
- IPFS_DATASETS_PY_TASK_P2P_SUBMIT_RETRY_BASE_MS / IPFS_ACCELERATE_PY_TASK_P2P_SUBMIT_RETRY_BASE_MS
    - exponential backoff base in milliseconds for submit retries (default: 50)
- IPFS_DATASETS_PY_TASK_P2P_SUBMIT_DIAL_TIMEOUT_S / IPFS_ACCELERATE_PY_TASK_P2P_SUBMIT_DIAL_TIMEOUT_S
    - per-attempt dial timeout for submit requests in seconds (default: 8.0)
- IPFS_DATASETS_PY_TASK_P2P_STATUS_RETRIES / IPFS_ACCELERATE_PY_TASK_P2P_STATUS_RETRIES
    - retry attempts for transient status transport failures (default: 2)
- IPFS_DATASETS_PY_TASK_P2P_STATUS_RETRY_BASE_MS / IPFS_ACCELERATE_PY_TASK_P2P_STATUS_RETRY_BASE_MS
    - exponential backoff base in milliseconds for status retries (default: 50)
- IPFS_DATASETS_PY_TASK_P2P_STATUS_DIAL_TIMEOUT_S / IPFS_ACCELERATE_PY_TASK_P2P_STATUS_DIAL_TIMEOUT_S
    - per-attempt dial timeout for status requests in seconds (default: 8.0)
- IPFS_DATASETS_PY_TASK_P2P_WAIT_RETRIES / IPFS_ACCELERATE_PY_TASK_P2P_WAIT_RETRIES
    - retry attempts for transient wait transport failures (default: 2)
- IPFS_DATASETS_PY_TASK_P2P_WAIT_RETRY_BASE_MS / IPFS_ACCELERATE_PY_TASK_P2P_WAIT_RETRY_BASE_MS
    - exponential backoff base in milliseconds for wait retries (default: 100)
- IPFS_DATASETS_PY_TASK_P2P_RPC_RETRIES / IPFS_ACCELERATE_PY_TASK_P2P_RPC_RETRIES
    - retry attempts for short control/cache/tool RPC transport failures (default: 2)
- IPFS_DATASETS_PY_TASK_P2P_RPC_RETRY_BASE_MS / IPFS_ACCELERATE_PY_TASK_P2P_RPC_RETRY_BASE_MS
    - exponential backoff base in milliseconds for short RPC retries (default: 50)
- IPFS_DATASETS_PY_TASK_P2P_RPC_DIAL_TIMEOUT_S / IPFS_ACCELERATE_PY_TASK_P2P_RPC_DIAL_TIMEOUT_S
    - per-attempt dial timeout for short control/cache/tool RPCs in seconds (default: 8.0)
- IPFS_DATASETS_PY_TASK_P2P_REMOTE_COOLDOWN_BASE_MS / IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_COOLDOWN_BASE_MS
    - adaptive per-remote cooldown base in milliseconds after retryable transport failures (default: 25)
- IPFS_DATASETS_PY_TASK_P2P_REMOTE_COOLDOWN_MAX_MS / IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_COOLDOWN_MAX_MS
    - cap for adaptive per-remote cooldown in milliseconds (default: 1000)
- IPFS_DATASETS_PY_TASK_P2P_RETRY_DIAL_TIMEOUT_SCALE / IPFS_ACCELERATE_PY_TASK_P2P_RETRY_DIAL_TIMEOUT_SCALE
    - multiplicative dial-timeout scale applied per retry attempt (default: 1.25)
- IPFS_DATASETS_PY_TASK_P2P_RETRY_DIAL_TIMEOUT_MAX_S / IPFS_ACCELERATE_PY_TASK_P2P_RETRY_DIAL_TIMEOUT_MAX_S
    - cap in seconds for scaled retry dial timeouts (default: 30.0)
- IPFS_DATASETS_PY_TASK_P2P_MAX_CONCURRENT_DIALS / IPFS_ACCELERATE_PY_TASK_P2P_MAX_CONCURRENT_DIALS
    - process-level limit for concurrent dial attempts across submit/status/short RPC retries (default: 32)
- IPFS_DATASETS_PY_TASK_P2P_DIAL_SLOT_TIMEOUT_S / IPFS_ACCELERATE_PY_TASK_P2P_DIAL_SLOT_TIMEOUT_S
    - maximum seconds to wait for a dial slot before failing the attempt (default: 10.0)
- IPFS_DATASETS_PY_TASK_P2P_MAX_CONCURRENT_WAIT_DIALS / IPFS_ACCELERATE_PY_TASK_P2P_MAX_CONCURRENT_WAIT_DIALS
    - process-level limit for concurrent dial attempts for long-poll `wait` requests (default: 128)
- IPFS_DATASETS_PY_TASK_P2P_WAIT_DIAL_SLOT_TIMEOUT_S / IPFS_ACCELERATE_PY_TASK_P2P_WAIT_DIAL_SLOT_TIMEOUT_S
    - maximum seconds to wait for a long-poll `wait` dial slot before failing the attempt (default: 30.0)
- IPFS_DATASETS_PY_TASK_P2P_REMOTE_STATE_MAX_KEYS / IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_STATE_MAX_KEYS
    - maximum remote cooldown-state entries retained in-process (default: 2048)
- IPFS_DATASETS_PY_TASK_P2P_REMOTE_STATE_STALE_S / IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_STATE_STALE_S
    - seconds after which idle remote cooldown-state entries may be pruned (default: 600)
- IPFS_DATASETS_PY_TASK_P2P_EXPLICIT_ADDR_COOLDOWN_BASE_MS / IPFS_ACCELERATE_PY_TASK_P2P_EXPLICIT_ADDR_COOLDOWN_BASE_MS
    - adaptive cooldown base in milliseconds for stale explicit multiaddrs after transport failures (default: 250)
- IPFS_DATASETS_PY_TASK_P2P_EXPLICIT_ADDR_COOLDOWN_MAX_MS / IPFS_ACCELERATE_PY_TASK_P2P_EXPLICIT_ADDR_COOLDOWN_MAX_MS
    - cap for stale explicit multiaddr cooldown in milliseconds (default: 5000)
- IPFS_DATASETS_PY_TASK_P2P_CACHE_MAX_KEYS / IPFS_ACCELERATE_PY_TASK_P2P_CACHE_MAX_KEYS
    - maximum discovered multiaddr cache entries retained in-process (default: 1024)
- IPFS_DATASETS_PY_TASK_P2P_CACHE_STALE_S / IPFS_ACCELERATE_PY_TASK_P2P_CACHE_STALE_S
    - seconds after which idle discovered multiaddr cache entries are pruned (default: 1800)
- IPFS_DATASETS_PY_TASK_P2P_RETRY_LIGHTWEIGHT_DISCOVERY / IPFS_ACCELERATE_PY_TASK_P2P_RETRY_LIGHTWEIGHT_DISCOVERY
    - when enabled, retry attempts use lightweight dialing (cache + announce) and skip broad discovery fanout (default: 1)
- IPFS_DATASETS_PY_TASK_P2P_RETRY_DELAY_MAX_MS / IPFS_ACCELERATE_PY_TASK_P2P_RETRY_DELAY_MAX_MS
    - cap in milliseconds for retry backoff delay before jitter (default: 5000)
"""

from __future__ import annotations

import json
import functools
import os
import random
import sys
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

from .protocol import PROTOCOL_V1, get_shared_token


def _truthy(text: str | None, *, default: bool = False) -> bool:
    if text is None:
        return bool(default)
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_bool(*, primary: str, compat: str, default: bool) -> bool:
    raw = os.environ.get(primary)
    if raw is None:
        raw = os.environ.get(compat)
    if raw is None:
        return bool(default)
    return _truthy(str(raw), default=default)


def _env_str(*, primary: str, compat: str, default: str) -> str:
    raw = os.environ.get(primary)
    if raw is None:
        raw = os.environ.get(compat)
    text = str(raw).strip() if raw is not None else ""
    return text or str(default)


def _have_libp2p() -> bool:
    try:
        import libp2p  # noqa: F401
        return True
    except Exception:
        return False


def _dial_debug_enabled() -> bool:
    return _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_DEBUG_DIAL",
        compat="IPFS_DATASETS_PY_TASK_P2P_DEBUG_DIAL",
        default=False,
    )


def _dial_debug(msg: str) -> None:
    if not _dial_debug_enabled():
        return
    try:
        print(f"p2p_tasks.client dial-debug: {msg}", file=sys.stderr, flush=True)
    except Exception:
        pass


def _submit_retry_attempts() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SUBMIT_RETRIES")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_SUBMIT_RETRIES", "2")
        return max(0, int(str(raw).strip()))
    except Exception:
        return 2


def _submit_retry_base_ms() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SUBMIT_RETRY_BASE_MS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_SUBMIT_RETRY_BASE_MS", "50")
        return max(10, int(str(raw).strip()))
    except Exception:
        return 50


def _submit_dial_timeout_s() -> float:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_SUBMIT_DIAL_TIMEOUT_S")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_SUBMIT_DIAL_TIMEOUT_S", "8.0")
        return max(1.0, float(str(raw).strip()))
    except Exception:
        return 8.0


def _status_retry_attempts() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_STATUS_RETRIES")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_STATUS_RETRIES", "2")
        return max(0, int(str(raw).strip()))
    except Exception:
        return 2


def _status_retry_base_ms() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_STATUS_RETRY_BASE_MS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_STATUS_RETRY_BASE_MS", "50")
        return max(10, int(str(raw).strip()))
    except Exception:
        return 50


def _status_dial_timeout_s() -> float:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_STATUS_DIAL_TIMEOUT_S")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_STATUS_DIAL_TIMEOUT_S", "8.0")
        return max(1.0, float(str(raw).strip()))
    except Exception:
        return 8.0


def _wait_retry_attempts() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_WAIT_RETRIES")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_WAIT_RETRIES", "2")
        return max(0, int(str(raw).strip()))
    except Exception:
        return 2


def _wait_retry_base_ms() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_WAIT_RETRY_BASE_MS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_WAIT_RETRY_BASE_MS", "100")
        return max(10, int(str(raw).strip()))
    except Exception:
        return 100


def _rpc_retry_attempts() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_RPC_RETRIES")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_RPC_RETRIES", "2")
        return max(0, int(str(raw).strip()))
    except Exception:
        return 2


def _rpc_retry_base_ms() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_RPC_RETRY_BASE_MS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_RPC_RETRY_BASE_MS", "50")
        return max(10, int(str(raw).strip()))
    except Exception:
        return 50


def _rpc_dial_timeout_s() -> float:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_RPC_DIAL_TIMEOUT_S")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_RPC_DIAL_TIMEOUT_S", "8.0")
        return max(1.0, float(str(raw).strip()))
    except Exception:
        return 8.0


def _remote_cooldown_base_ms() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_COOLDOWN_BASE_MS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_REMOTE_COOLDOWN_BASE_MS", "25")
        return max(10, int(str(raw).strip()))
    except Exception:
        return 25


def _remote_cooldown_max_ms() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_COOLDOWN_MAX_MS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_REMOTE_COOLDOWN_MAX_MS", "1000")
        return max(50, int(str(raw).strip()))
    except Exception:
        return 1000


def _retry_dial_timeout_scale() -> float:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_RETRY_DIAL_TIMEOUT_SCALE")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_RETRY_DIAL_TIMEOUT_SCALE", "1.25")
        return max(1.0, float(str(raw).strip()))
    except Exception:
        return 1.25


def _retry_dial_timeout_max_s() -> float:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_RETRY_DIAL_TIMEOUT_MAX_S")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_RETRY_DIAL_TIMEOUT_MAX_S", "30.0")
        return max(1.0, float(str(raw).strip()))
    except Exception:
        return 30.0


def _dial_timeout_for_attempt(*, base_timeout_s: float, attempt: int) -> float:
    base = max(1.0, float(base_timeout_s))
    scale = _retry_dial_timeout_scale()
    cap = _retry_dial_timeout_max_s()
    if attempt <= 0 or scale <= 1.0:
        return min(base, cap)
    return min(cap, base * (scale ** int(attempt)))


def _max_concurrent_dials() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_MAX_CONCURRENT_DIALS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_MAX_CONCURRENT_DIALS", "32")
        return max(1, int(str(raw).strip()))
    except Exception:
        return 32


def _dial_slot_timeout_s() -> float:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_DIAL_SLOT_TIMEOUT_S")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_DIAL_SLOT_TIMEOUT_S", "10.0")
        return max(0.1, float(str(raw).strip()))
    except Exception:
        return 10.0


def _wait_dial_slot_timeout_s() -> float:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_WAIT_DIAL_SLOT_TIMEOUT_S")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_WAIT_DIAL_SLOT_TIMEOUT_S", "30.0")
        return max(0.1, float(str(raw).strip()))
    except Exception:
        return 30.0


def _max_concurrent_wait_dials() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_MAX_CONCURRENT_WAIT_DIALS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_MAX_CONCURRENT_WAIT_DIALS", "128")
        return max(1, int(str(raw).strip()))
    except Exception:
        return 128


def _remote_state_max_keys() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_STATE_MAX_KEYS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_REMOTE_STATE_MAX_KEYS", "2048")
        return max(64, int(str(raw).strip()))
    except Exception:
        return 2048


def _remote_state_stale_s() -> float:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_REMOTE_STATE_STALE_S")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_REMOTE_STATE_STALE_S", "600")
        return max(30.0, float(str(raw).strip()))
    except Exception:
        return 600.0


def _explicit_addr_cooldown_base_ms() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_EXPLICIT_ADDR_COOLDOWN_BASE_MS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_EXPLICIT_ADDR_COOLDOWN_BASE_MS", "250")
        return max(10, int(str(raw).strip()))
    except Exception:
        return 250


def _explicit_addr_cooldown_max_ms() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_EXPLICIT_ADDR_COOLDOWN_MAX_MS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_EXPLICIT_ADDR_COOLDOWN_MAX_MS", "5000")
        return max(50, int(str(raw).strip()))
    except Exception:
        return 5000


def _cache_state_max_keys() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_CACHE_MAX_KEYS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_CACHE_MAX_KEYS", "1024")
        return max(64, int(str(raw).strip()))
    except Exception:
        return 1024


def _cache_state_stale_s() -> float:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_CACHE_STALE_S")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_CACHE_STALE_S", "1800")
        return max(30.0, float(str(raw).strip()))
    except Exception:
        return 1800.0


def _retry_lightweight_discovery_enabled() -> bool:
    return _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RETRY_LIGHTWEIGHT_DISCOVERY",
        compat="IPFS_DATASETS_PY_TASK_P2P_RETRY_LIGHTWEIGHT_DISCOVERY",
        default=True,
    )


def _prefer_lightweight_first_enabled() -> bool:
    """Prefer lightweight dialing for early attempts in retry loops.

    High-throughput submit/wait patterns can otherwise trigger broad discovery
    fanout (bootstrap/rendezvous/dht/mdns) on every first attempt, which adds
    latency and dial pressure. Keep broad discovery as a late fallback.
    """

    return _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_PREFER_LIGHTWEIGHT_FIRST",
        compat="IPFS_DATASETS_PY_TASK_P2P_PREFER_LIGHTWEIGHT_FIRST",
        default=True,
    )


def _retry_delay_max_ms() -> int:
    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_RETRY_DELAY_MAX_MS")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_RETRY_DELAY_MAX_MS", "5000")
        return max(10, int(str(raw).strip()))
    except Exception:
        return 5000


def _cooldown_failfast_enabled() -> bool:
    """Whether retry loops should fast-fail remotes in cooldown windows."""

    return _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_COOLDOWN_FAILFAST",
        compat="IPFS_DATASETS_PY_TASK_P2P_COOLDOWN_FAILFAST",
        default=True,
    )


def _cooldown_failfast_threshold_s() -> float:
    """Minimum cooldown wait to trigger fast-fail behavior (seconds)."""

    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_COOLDOWN_FAILFAST_THRESHOLD_S")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_COOLDOWN_FAILFAST_THRESHOLD_S", "0.25")
        return max(0.0, float(str(raw).strip()))
    except Exception:
        return 0.25


def _retry_delay_s(*, attempt: int, base_ms: int) -> float:
    base = max(10, int(base_ms))
    # Cap exponent growth so high retry counts do not create runaway sleeps.
    exp = min(10, max(0, int(attempt)))
    delay_cap_ms = max(base, _retry_delay_max_ms())
    stage_delay_ms = min(delay_cap_ms, base * (2**exp))
    jitter_cap_ms = max(1, min(base, stage_delay_ms))
    return (stage_delay_ms + random.randint(0, int(jitter_cap_ms))) / 1000.0


_P2P_RETRY_METRICS_LOCK = threading.Lock()
_P2P_RETRY_METRICS: dict[str, int] = {}
_P2P_REMOTE_COOLDOWN_LOCK = threading.Lock()
_P2P_REMOTE_COOLDOWN_UNTIL_TS: dict[str, float] = {}
_P2P_REMOTE_COOLDOWN_FAILURE_STREAK: dict[str, int] = {}
_P2P_REMOTE_COOLDOWN_LAST_TOUCH_TS: dict[str, float] = {}
_P2P_REMOTE_COOLDOWN_MUTATIONS: int = 0
_P2P_EXPLICIT_ADDR_COOLDOWN_LOCK = threading.Lock()
_P2P_EXPLICIT_ADDR_COOLDOWN_UNTIL_TS: dict[str, float] = {}
_P2P_EXPLICIT_ADDR_COOLDOWN_STREAK: dict[str, int] = {}
_P2P_EXPLICIT_ADDR_COOLDOWN_LAST_TOUCH_TS: dict[str, float] = {}
_P2P_EXPLICIT_ADDR_COOLDOWN_MUTATIONS: int = 0
_P2P_DIAL_SEM_LOCK = threading.Lock()
_P2P_DIAL_SEM: threading.BoundedSemaphore | None = None
_P2P_DIAL_SEM_SIZE: int = 0
_P2P_WAIT_DIAL_SEM_LOCK = threading.Lock()
_P2P_WAIT_DIAL_SEM: threading.BoundedSemaphore | None = None
_P2P_WAIT_DIAL_SEM_SIZE: int = 0
_P2P_LAST_SUCCESS_PEER_LOCK = threading.Lock()
_P2P_LAST_SUCCESS_PEER_ID: str = ""
_P2P_LAST_SUCCESS_PEER_TS: float = 0.0
_DISCOVERED_MULTIADDR_LOCK = threading.Lock()
_DISCOVERED_MULTIADDR_TOUCH_TS: dict[str, float] = {}
_DISCOVERED_MULTIADDR_MUTATIONS: int = 0


def _retry_metric_inc(key: str, amount: int = 1) -> None:
    try:
        metric = str(key or "").strip()
        if not metric:
            return
        delta = int(amount)
        if delta <= 0:
            return
        with _P2P_RETRY_METRICS_LOCK:
            _P2P_RETRY_METRICS[metric] = int(_P2P_RETRY_METRICS.get(metric, 0)) + delta
    except Exception:
        pass


def get_p2p_retry_metrics() -> Dict[str, int]:
    """Return a snapshot of in-process retry counters by operation label."""
    with _P2P_RETRY_METRICS_LOCK:
        return {str(k): int(v) for k, v in _P2P_RETRY_METRICS.items()}


def reset_p2p_retry_metrics() -> None:
    """Reset in-process retry counters."""
    with _P2P_RETRY_METRICS_LOCK:
        _P2P_RETRY_METRICS.clear()


def _is_retryable_transport_error(exc: BaseException) -> bool:
    if isinstance(exc, BaseExceptionGroup):
        return True
    name = type(exc).__name__
    if name in {"TimeoutError", "OSError", "ConnectionError"}:
        return True
    msg = str(exc or "").lower()
    markers = (
        "discovery_timeout",
        "discovery timeout",
        "p2p request failed",
        "no response",
        "failed to negotiate the secure protocol",
        "handshake",
        "failed to upgrade security",
        "connect",
        "connection reset",
        "connection refused",
        "temporarily unavailable",
        "stream",
        "broken pipe",
    )
    return any(m in msg for m in markers)


def _is_retryable_response_error(resp: Dict[str, Any]) -> bool:
    """Return True when a response payload indicates a transient transport issue."""
    if not isinstance(resp, dict):
        return False
    if bool(resp.get("ok")):
        return False
    text = str(resp.get("error") or resp).strip().lower()
    if not text:
        return False
    markers = (
        "discovery_timeout",
        "p2p request failed",
        "no response",
        "unable to connect",
        "failed to negotiate the secure protocol",
        "failed to upgrade security",
        "handshake",
        "connect",
        "stream",
        "swarmexception",
        "timeout",
    )
    return any(m in text for m in markers)


def _exception_group_contains_timeout(exc: BaseException) -> bool:
    """Return True when an exception group tree contains a TimeoutError."""
    if isinstance(exc, TimeoutError):
        return True
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            try:
                if _exception_group_contains_timeout(inner):
                    return True
            except Exception:
                continue
    return False


def _exception_group_contains_dial_slot_timeout(exc: BaseException) -> bool:
    """Return True when an exception group tree includes dial-slot timeout text."""
    msg = str(exc or "").lower()
    if "dial slot timeout" in msg:
        return True
    if isinstance(exc, BaseExceptionGroup):
        for inner in exc.exceptions:
            try:
                if _exception_group_contains_dial_slot_timeout(inner):
                    return True
            except Exception:
                continue
    return False


def _remote_cooldown_key(remote: Any) -> str:
    pid = str(getattr(remote, "peer_id", "") or "").strip()
    ma = str(getattr(remote, "multiaddr", "") or "").strip()
    if pid and ma:
        return f"{pid}::{ma}"
    return pid or ma or "<unknown>"


def _remote_cooldown_wait_s(remote: Any) -> float:
    key = _remote_cooldown_key(remote)
    now = time.monotonic()
    with _P2P_REMOTE_COOLDOWN_LOCK:
        until_ts = float(_P2P_REMOTE_COOLDOWN_UNTIL_TS.get(key, 0.0) or 0.0)
    if until_ts <= now:
        return 0.0
    return max(0.0, until_ts - now)


def _prune_remote_state_locked(now: float) -> None:
    """Prune stale/excess remote cooldown state. Caller must hold lock."""
    max_keys = _remote_state_max_keys()
    stale_s = _remote_state_stale_s()

    if not _P2P_REMOTE_COOLDOWN_LAST_TOUCH_TS:
        return

    stale_cutoff = float(now) - float(stale_s)
    stale_keys = [
        k for (k, ts) in _P2P_REMOTE_COOLDOWN_LAST_TOUCH_TS.items()
        if float(ts) < stale_cutoff and float(_P2P_REMOTE_COOLDOWN_UNTIL_TS.get(k, 0.0) or 0.0) <= float(now)
    ]
    if stale_keys:
        for key in stale_keys:
            _P2P_REMOTE_COOLDOWN_LAST_TOUCH_TS.pop(key, None)
            _P2P_REMOTE_COOLDOWN_UNTIL_TS.pop(key, None)
            _P2P_REMOTE_COOLDOWN_FAILURE_STREAK.pop(key, None)
        _retry_metric_inc("remote_state.pruned", len(stale_keys))

    n = len(_P2P_REMOTE_COOLDOWN_LAST_TOUCH_TS)
    if n <= int(max_keys):
        return

    overflow = n - int(max_keys)
    oldest = sorted(_P2P_REMOTE_COOLDOWN_LAST_TOUCH_TS.items(), key=lambda kv: float(kv[1]))[:overflow]
    if oldest:
        for key, _ in oldest:
            _P2P_REMOTE_COOLDOWN_LAST_TOUCH_TS.pop(key, None)
            _P2P_REMOTE_COOLDOWN_UNTIL_TS.pop(key, None)
            _P2P_REMOTE_COOLDOWN_FAILURE_STREAK.pop(key, None)
        _retry_metric_inc("remote_state.pruned", len(oldest))


def _prune_explicit_addr_state_locked(now: float) -> None:
    """Prune stale/excess explicit-address cooldown state. Caller must hold lock."""
    max_keys = _remote_state_max_keys()
    stale_s = _remote_state_stale_s()

    if not _P2P_EXPLICIT_ADDR_COOLDOWN_LAST_TOUCH_TS:
        return

    stale_cutoff = float(now) - float(stale_s)
    stale_keys = [
        k
        for (k, ts) in _P2P_EXPLICIT_ADDR_COOLDOWN_LAST_TOUCH_TS.items()
        if float(ts) < stale_cutoff and float(_P2P_EXPLICIT_ADDR_COOLDOWN_UNTIL_TS.get(k, 0.0) or 0.0) <= float(now)
    ]
    if stale_keys:
        for key in stale_keys:
            _P2P_EXPLICIT_ADDR_COOLDOWN_LAST_TOUCH_TS.pop(key, None)
            _P2P_EXPLICIT_ADDR_COOLDOWN_UNTIL_TS.pop(key, None)
            _P2P_EXPLICIT_ADDR_COOLDOWN_STREAK.pop(key, None)
        _retry_metric_inc("explicit_addr_state.pruned", len(stale_keys))

    n = len(_P2P_EXPLICIT_ADDR_COOLDOWN_LAST_TOUCH_TS)
    if n <= int(max_keys):
        return

    overflow = n - int(max_keys)
    oldest = sorted(_P2P_EXPLICIT_ADDR_COOLDOWN_LAST_TOUCH_TS.items(), key=lambda kv: float(kv[1]))[:overflow]
    if oldest:
        for key, _ in oldest:
            _P2P_EXPLICIT_ADDR_COOLDOWN_LAST_TOUCH_TS.pop(key, None)
            _P2P_EXPLICIT_ADDR_COOLDOWN_UNTIL_TS.pop(key, None)
            _P2P_EXPLICIT_ADDR_COOLDOWN_STREAK.pop(key, None)
        _retry_metric_inc("explicit_addr_state.pruned", len(oldest))


def _explicit_addr_cooldown_wait_s(peer_multiaddr: str) -> float:
    key = str(peer_multiaddr or "").strip()
    if not key:
        return 0.0
    now = time.monotonic()
    with _P2P_EXPLICIT_ADDR_COOLDOWN_LOCK:
        until_ts = float(_P2P_EXPLICIT_ADDR_COOLDOWN_UNTIL_TS.get(key, 0.0) or 0.0)
    if until_ts <= now:
        return 0.0
    return max(0.0, until_ts - now)


def _explicit_addr_cooldown_mark_failure(peer_multiaddr: str) -> None:
    global _P2P_EXPLICIT_ADDR_COOLDOWN_MUTATIONS
    key = str(peer_multiaddr or "").strip()
    if not key:
        return
    base_ms = _explicit_addr_cooldown_base_ms()
    max_ms = max(_explicit_addr_cooldown_max_ms(), base_ms)
    now = time.monotonic()
    with _P2P_EXPLICIT_ADDR_COOLDOWN_LOCK:
        streak = int(_P2P_EXPLICIT_ADDR_COOLDOWN_STREAK.get(key, 0)) + 1
        _P2P_EXPLICIT_ADDR_COOLDOWN_STREAK[key] = streak
        delay_ms = min(max_ms, base_ms * (2 ** max(0, streak - 1)))
        jitter_ms = random.randint(0, max(1, base_ms // 2))
        _P2P_EXPLICIT_ADDR_COOLDOWN_UNTIL_TS[key] = float(now) + ((delay_ms + jitter_ms) / 1000.0)
        _P2P_EXPLICIT_ADDR_COOLDOWN_LAST_TOUCH_TS[key] = float(now)
        _P2P_EXPLICIT_ADDR_COOLDOWN_MUTATIONS = int(_P2P_EXPLICIT_ADDR_COOLDOWN_MUTATIONS) + 1
        if (
            len(_P2P_EXPLICIT_ADDR_COOLDOWN_LAST_TOUCH_TS) > _remote_state_max_keys()
        ) or (_P2P_EXPLICIT_ADDR_COOLDOWN_MUTATIONS % 64 == 0):
            _prune_explicit_addr_state_locked(float(now))


def _explicit_addr_cooldown_mark_success(peer_multiaddr: str) -> None:
    key = str(peer_multiaddr or "").strip()
    if not key:
        return
    with _P2P_EXPLICIT_ADDR_COOLDOWN_LOCK:
        _P2P_EXPLICIT_ADDR_COOLDOWN_LAST_TOUCH_TS.pop(key, None)
        _P2P_EXPLICIT_ADDR_COOLDOWN_STREAK.pop(key, None)
        _P2P_EXPLICIT_ADDR_COOLDOWN_UNTIL_TS.pop(key, None)


def _remote_cooldown_mark_failure(remote: Any) -> None:
    global _P2P_REMOTE_COOLDOWN_MUTATIONS
    key = _remote_cooldown_key(remote)
    base_ms = _remote_cooldown_base_ms()
    max_ms = max(_remote_cooldown_max_ms(), base_ms)
    now = time.monotonic()
    with _P2P_REMOTE_COOLDOWN_LOCK:
        streak = int(_P2P_REMOTE_COOLDOWN_FAILURE_STREAK.get(key, 0)) + 1
        _P2P_REMOTE_COOLDOWN_FAILURE_STREAK[key] = streak
        delay_ms = min(max_ms, base_ms * (2 ** max(0, streak - 1)))
        jitter_ms = random.randint(0, max(1, base_ms // 2))
        _P2P_REMOTE_COOLDOWN_UNTIL_TS[key] = float(now) + ((delay_ms + jitter_ms) / 1000.0)
        _P2P_REMOTE_COOLDOWN_LAST_TOUCH_TS[key] = float(now)
        _P2P_REMOTE_COOLDOWN_MUTATIONS = int(_P2P_REMOTE_COOLDOWN_MUTATIONS) + 1
        if (len(_P2P_REMOTE_COOLDOWN_LAST_TOUCH_TS) > _remote_state_max_keys()) or (_P2P_REMOTE_COOLDOWN_MUTATIONS % 64 == 0):
            _prune_remote_state_locked(float(now))


def _remote_cooldown_mark_success(remote: Any) -> None:
    key = _remote_cooldown_key(remote)
    with _P2P_REMOTE_COOLDOWN_LOCK:
        _P2P_REMOTE_COOLDOWN_LAST_TOUCH_TS.pop(key, None)
        _P2P_REMOTE_COOLDOWN_FAILURE_STREAK.pop(key, None)
        _P2P_REMOTE_COOLDOWN_UNTIL_TS.pop(key, None)


def _get_dial_semaphore() -> threading.BoundedSemaphore:
    global _P2P_DIAL_SEM, _P2P_DIAL_SEM_SIZE
    size = _max_concurrent_dials()
    with _P2P_DIAL_SEM_LOCK:
        if _P2P_DIAL_SEM is None or int(_P2P_DIAL_SEM_SIZE) != int(size):
            _P2P_DIAL_SEM = threading.BoundedSemaphore(value=int(size))
            _P2P_DIAL_SEM_SIZE = int(size)
        return _P2P_DIAL_SEM


def _get_wait_dial_semaphore() -> threading.BoundedSemaphore:
    global _P2P_WAIT_DIAL_SEM, _P2P_WAIT_DIAL_SEM_SIZE
    size = _max_concurrent_wait_dials()
    with _P2P_WAIT_DIAL_SEM_LOCK:
        if _P2P_WAIT_DIAL_SEM is None or int(_P2P_WAIT_DIAL_SEM_SIZE) != int(size):
            _P2P_WAIT_DIAL_SEM = threading.BoundedSemaphore(value=int(size))
            _P2P_WAIT_DIAL_SEM_SIZE = int(size)
        return _P2P_WAIT_DIAL_SEM


async def _acquire_dial_slot(*, op_label: str) -> Any:
    import anyio

    is_wait = str(op_label) == "wait"
    sem = _get_wait_dial_semaphore() if is_wait else _get_dial_semaphore()
    timeout_s = _wait_dial_slot_timeout_s() if is_wait else _dial_slot_timeout_s()
    t0 = time.monotonic()
    acquired = await anyio.to_thread.run_sync(lambda: sem.acquire(timeout=timeout_s))
    if not acquired:
        _retry_metric_inc(f"{op_label}.dial_slot_timeout")
        raise TimeoutError(f"dial slot timeout after {timeout_s:.3f}s for {op_label}")
    waited_s = max(0.0, time.monotonic() - t0)
    if waited_s > 0.001:
        _retry_metric_inc(f"{op_label}.dial_slot_wait")

    def _release() -> None:
        try:
            sem.release()
        except Exception:
            pass

    return _release


async def _dial_and_request_with_retries(
    *,
    remote: RemoteQueue,
    message: Dict[str, Any],
    retries: int,
    retry_base_ms: int,
    dial_timeout_s: float,
    op_label: str,
    should_retry_response: Optional[Callable[[Dict[str, Any]], bool]] = None,
) -> Dict[str, Any]:
    import anyio

    max_retries = max(0, int(retries))
    for attempt in range(max_retries + 1):
        attempt_dial_timeout_s = _dial_timeout_for_attempt(base_timeout_s=float(dial_timeout_s), attempt=attempt)
        broad_discovery_override: bool | None = None
        # Use lightweight dialing for early attempts to reduce network fanout
        # under sustained request rates, then allow broad discovery on the last
        # attempt as a recovery path.
        prefer_lightweight_first = _prefer_lightweight_first_enabled()
        if prefer_lightweight_first and attempt < max_retries:
            broad_discovery_override = False
            if attempt == 0:
                _retry_metric_inc(f"{op_label}.lightweight_first")
            else:
                _retry_metric_inc(f"{op_label}.retry_lightweight_discovery")
        elif attempt > 0 and _retry_lightweight_discovery_enabled():
            broad_discovery_override = False
            _retry_metric_inc(f"{op_label}.retry_lightweight_discovery")
        cooldown_wait_s = _remote_cooldown_wait_s(remote)
        if cooldown_wait_s > 0:
            failfast_threshold_s = _cooldown_failfast_threshold_s()
            if (
                _cooldown_failfast_enabled()
                and cooldown_wait_s >= failfast_threshold_s
                and attempt < max_retries
            ):
                # Under high concurrency, sleeping for full cooldown on each
                # call can stall worker throughput. Fast-fail this attempt so
                # callers with multi-remote logic can fail over immediately.
                _retry_metric_inc(f"{op_label}.cooldown_fast_fail")
                quick_delay_s = min(0.2, _retry_delay_s(attempt=attempt, base_ms=retry_base_ms))
                _dial_debug(
                    f"{op_label} cooldown fast-fail attempt={attempt + 1}/{max_retries + 1} "
                    f"cooldown_wait_s={cooldown_wait_s:.3f} quick_delay_s={quick_delay_s:.3f}"
                )
                if quick_delay_s > 0:
                    await anyio.sleep(quick_delay_s)
                continue

            _retry_metric_inc(f"{op_label}.cooldown_wait")
            _dial_debug(
                f"{op_label} cooldown wait before attempt={attempt + 1}/{max_retries + 1} wait_s={cooldown_wait_s:.3f}"
            )
            await anyio.sleep(cooldown_wait_s)
        try:
            release_slot = await _acquire_dial_slot(op_label=op_label)
            try:
                resp = await _dial_and_request(
                    remote=remote,
                    message=message,
                    dial_timeout_s=attempt_dial_timeout_s,
                    allow_broad_discovery_override=broad_discovery_override,
                )
            finally:
                release_slot()
            retryable_response = False
            if callable(should_retry_response):
                try:
                    retryable_response = bool(should_retry_response(resp))
                except Exception:
                    retryable_response = False
            if not retryable_response:
                retryable_response = _is_retryable_response_error(resp)
            if retryable_response:
                _remote_cooldown_mark_failure(remote)
                if attempt >= max_retries:
                    _retry_metric_inc(f"{op_label}.failed")
                    return resp
                delay_s = _retry_delay_s(attempt=attempt, base_ms=retry_base_ms)
                _retry_metric_inc(f"{op_label}.retry")
                _retry_metric_inc(f"{op_label}.retry_response")
                _dial_debug(
                    f"{op_label} retry after retryable response attempt={attempt + 1}/{max_retries + 1} delay_s={delay_s:.3f}"
                )
                await anyio.sleep(delay_s)
                continue
            _remote_cooldown_mark_success(remote)
            if attempt > 0:
                _retry_metric_inc(f"{op_label}.recovered")
            return resp
        except BaseExceptionGroup as exc:
            retryable = _is_retryable_transport_error(exc)
            if retryable:
                _remote_cooldown_mark_failure(remote)
            if attempt >= max_retries or not retryable:
                _retry_metric_inc(f"{op_label}.failed")
                raise
            delay_s = _retry_delay_s(attempt=attempt, base_ms=retry_base_ms)
            _retry_metric_inc(f"{op_label}.retry")
            _dial_debug(
                f"{op_label} retry after BaseExceptionGroup attempt={attempt + 1}/{max_retries + 1} delay_s={delay_s:.3f}"
            )
            await anyio.sleep(delay_s)
        except Exception as exc:
            retryable = _is_retryable_transport_error(exc)
            if retryable:
                _remote_cooldown_mark_failure(remote)
            if attempt >= max_retries or not retryable:
                _retry_metric_inc(f"{op_label}.failed")
                raise
            delay_s = _retry_delay_s(attempt=attempt, base_ms=retry_base_ms)
            _retry_metric_inc(f"{op_label}.retry")
            _dial_debug(
                f"{op_label} retry after {type(exc).__name__} attempt={attempt + 1}/{max_retries + 1} delay_s={delay_s:.3f}"
            )
            await anyio.sleep(delay_s)

    raise RuntimeError(f"{op_label} failed after retries")


@dataclass
class RemoteQueue:
    peer_id: str = ""
    multiaddr: str = ""


# Best-effort in-process cache for discovery-derived dial multiaddrs.
#
# The convenience helpers in this module create a new libp2p host per request.
# When relying on mDNS, recreating discovery repeatedly can race announcements
# and yield `discovery_timeout` on submit/wait loops. Caching the last known
# dialable multiaddr per peer makes repeated RPCs much more stable.
_DISCOVERED_MULTIADDR_CACHE: dict[str, str] = {}


def _cache_prune_locked(now: float) -> None:
    """Prune stale/excess discovered multiaddr cache entries. Caller holds lock."""
    max_keys = _cache_state_max_keys()
    stale_s = _cache_state_stale_s()

    if not _DISCOVERED_MULTIADDR_TOUCH_TS:
        return

    stale_cutoff = float(now) - float(stale_s)
    stale_keys = [k for (k, ts) in _DISCOVERED_MULTIADDR_TOUCH_TS.items() if float(ts) < stale_cutoff]
    if stale_keys:
        for key in stale_keys:
            _DISCOVERED_MULTIADDR_TOUCH_TS.pop(key, None)
            _DISCOVERED_MULTIADDR_CACHE.pop(key, None)
        _retry_metric_inc("cache.pruned", len(stale_keys))

    n = len(_DISCOVERED_MULTIADDR_TOUCH_TS)
    if n <= int(max_keys):
        return

    overflow = n - int(max_keys)
    oldest = sorted(_DISCOVERED_MULTIADDR_TOUCH_TS.items(), key=lambda kv: float(kv[1]))[:overflow]
    if oldest:
        for key, _ in oldest:
            _DISCOVERED_MULTIADDR_TOUCH_TS.pop(key, None)
            _DISCOVERED_MULTIADDR_CACHE.pop(key, None)
        _retry_metric_inc("cache.pruned", len(oldest))


def _cache_get_multiaddr(peer_id: str) -> str:
    try:
        pid = str(peer_id).strip()
        if not pid:
            return ""
        now = time.monotonic()
        with _DISCOVERED_MULTIADDR_LOCK:
            ts = float(_DISCOVERED_MULTIADDR_TOUCH_TS.get(pid, 0.0) or 0.0)
            if ts > 0.0 and (float(now) - ts) > _cache_state_stale_s():
                _DISCOVERED_MULTIADDR_TOUCH_TS.pop(pid, None)
                _DISCOVERED_MULTIADDR_CACHE.pop(pid, None)
                _retry_metric_inc("cache.expired")
                return ""
            ma = str(_DISCOVERED_MULTIADDR_CACHE.get(pid) or "")
            if ma:
                _DISCOVERED_MULTIADDR_TOUCH_TS[pid] = float(now)
            return ma
    except Exception:
        return ""


def _cache_set_multiaddr(peer_id: str, multiaddr: str) -> None:
    global _DISCOVERED_MULTIADDR_MUTATIONS
    pid = str(peer_id or "").strip()
    ma = str(multiaddr or "").strip()
    if not pid or not ma:
        return
    now = time.monotonic()
    with _DISCOVERED_MULTIADDR_LOCK:
        _DISCOVERED_MULTIADDR_CACHE[pid] = ma
        _DISCOVERED_MULTIADDR_TOUCH_TS[pid] = float(now)
        _DISCOVERED_MULTIADDR_MUTATIONS = int(_DISCOVERED_MULTIADDR_MUTATIONS) + 1
        if (len(_DISCOVERED_MULTIADDR_TOUCH_TS) > _cache_state_max_keys()) or (_DISCOVERED_MULTIADDR_MUTATIONS % 64 == 0):
            _cache_prune_locked(float(now))


def _cache_del_multiaddr(peer_id: str) -> None:
    try:
        pid = str(peer_id).strip()
        with _DISCOVERED_MULTIADDR_LOCK:
            _DISCOVERED_MULTIADDR_CACHE.pop(pid, None)
            _DISCOVERED_MULTIADDR_TOUCH_TS.pop(pid, None)
    except Exception:
        pass


def _addr_to_peer_multiaddr_text(addr: object, peer_id: str) -> str:
    """Convert a peerstore address to a dialable /p2p/... multiaddr string."""
    pid = str(peer_id or "").strip()
    if not pid:
        return ""
    try:
        text = str(addr).strip()
    except Exception:
        return ""
    if not text:
        return ""
    # Avoid undialable wildcard/unspecified addresses.
    if text.startswith("/ip4/0.0.0.0/") or text.startswith("/ip6/::/"):
        return ""
    # Avoid loopback / link-local addresses as they are not dialable from
    # other hosts (common in mDNS TXT records). Allow opt-in for single-host
    # test setups that expect to dial 127.0.0.1.
    if not _mdns_allow_loopback() and (text.startswith("/ip4/127.") or text.startswith("/ip6/::1/")):
        return ""
    if text.startswith("/ip4/169.254.") or text.startswith("/ip6/fe80:"):
        return ""
    if "/p2p/" not in text:
        text = f"{text}/p2p/{pid}"
    return text


def _mdns_allow_loopback() -> bool:
    return _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_MDNS_ALLOW_LOOPBACK",
        compat="IPFS_DATASETS_PY_TASK_P2P_MDNS_ALLOW_LOOPBACK",
        default=False,
    )


def _mdns_max_attempts_per_poll() -> int:
    """Maximum number of mDNS peers to dial per discovery poll iteration."""

    try:
        raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_MDNS_MAX_ATTEMPTS_PER_POLL")
        if raw is None:
            raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_MDNS_MAX_ATTEMPTS_PER_POLL", "4")
        return max(1, int(str(raw).strip()))
    except Exception:
        return 4


def _mark_last_success_peer(peer_id: str) -> None:
    pid = str(peer_id or "").strip()
    if not pid:
        return
    with _P2P_LAST_SUCCESS_PEER_LOCK:
        global _P2P_LAST_SUCCESS_PEER_ID, _P2P_LAST_SUCCESS_PEER_TS
        _P2P_LAST_SUCCESS_PEER_ID = pid
        _P2P_LAST_SUCCESS_PEER_TS = float(time.monotonic())


def _get_last_success_peer(*, max_age_s: float = 300.0) -> str:
    with _P2P_LAST_SUCCESS_PEER_LOCK:
        pid = str(_P2P_LAST_SUCCESS_PEER_ID or "").strip()
        if not pid:
            return ""
        age_s = float(time.monotonic()) - float(_P2P_LAST_SUCCESS_PEER_TS)
        if age_s > max(1.0, float(max_age_s)):
            return ""
        return pid


def _pick_best_peer_multiaddr_text(addrs: object, peer_id: str) -> str:
    """Pick the best dial target from a peerstore address set.

    When loopback dialing is explicitly allowed, prefer loopback addresses for
    determinism in single-host environments.
    """

    pid = str(peer_id or "").strip()
    if not pid:
        return ""
    allow_loopback = _mdns_allow_loopback()

    candidates: list[str] = []
    try:
        items = list(addrs or [])
    except Exception:
        items = []

    for a in items:
        ma = _addr_to_peer_multiaddr_text(a, pid)
        if ma:
            candidates.append(ma)

    if not candidates:
        return ""

    def _score(ma: str) -> tuple[int, int, str]:
        text = str(ma or "")
        is_loop = text.startswith("/ip4/127.") or text.startswith("/ip6/::1/")
        # Lower is better.
        loop_rank = 0 if (allow_loopback and is_loop) else 1
        ip_rank = 0 if text.startswith("/ip4/") else (1 if text.startswith("/ip6/") else 2)
        return (loop_rank, ip_rank, text)

    candidates.sort(key=_score)
    return candidates[0]


def _multiaddr_peer_id(multiaddr: str) -> str:
    try:
        return str(multiaddr).rsplit("/p2p/", 1)[-1].strip()
    except Exception:
        return ""


def _dht_value_record_key(ns: str) -> str:
    key = str(ns or "").strip()
    return f"/ipfs-accelerate/task-queue/ns/{key}" if key else "/ipfs-accelerate/task-queue/ns"


def _best_effort_peerinfo_multiaddrs(peer_info: Any) -> list[str]:
    try:
        pid = getattr(peer_info, "peer_id", None)
        pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid or "")
    except Exception:
        pid_text = ""

    addrs: list[str] = []
    try:
        for a in list(getattr(peer_info, "addrs", None) or []):
            try:
                a_text = str(a)
                if pid_text and "/p2p/" not in a_text:
                    a_text = f"{a_text}/p2p/{pid_text}"
                addrs.append(a_text)
            except Exception:
                continue
    except Exception:
        pass
    return addrs


async def _try_peer_info(*, host, peer_info: Any, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    try:
        await host.connect(peer_info)
        stream = await host.new_stream(peer_info.peer_id, [PROTOCOL_V1])
        try:
            resp = await _request_over_stream(stream=stream, message=message)
            return resp if isinstance(resp, dict) else None
        finally:
            try:
                await stream.close()
            except Exception:
                pass
    except Exception:
        return None


def _parse_bootstrap_peers() -> list[str]:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_PEERS")
    )
    if raw is not None and str(raw).strip().lower() in {"0", "false", "no", "off"}:
        return []

    # Default to the public libp2p bootstrap set when not provided.
    # This is required for internet-wide DHT discovery to function.
    default = [
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmNnooDu7bfjPFoTZYxMNLWUQJyrVwtbZg5gBMjTezGAJN",
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmQCU2EcMqAqQPR2i9bChDtGNJchTbq5TbXJJ16u19uLTa",
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmbLHAnMoJPWSCR5Zhtx6BHJX9KiKNN6tpvbUcqanj75Nb",
        "/dnsaddr/bootstrap.libp2p.io/p2p/QmcZf59bWwK5XFi76CZX8cbJ4BhTzzA3gU1ZjYZcYW3dwt",
    ]
    text = str(raw).strip() if raw is not None else ""
    if not text:
        return _expand_dnsaddr_peers(list(default))

    parts = [p.strip() for p in text.split(",")]
    return _expand_dnsaddr_peers([p for p in parts if p])


def _dnsaddr_resolution_enabled() -> bool:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_DNSADDR_RESOLVE")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_DNSADDR_RESOLVE")
    )
    if raw is None:
        return True
    return str(raw).strip().lower() not in {"0", "false", "no", "off"}


@functools.lru_cache(maxsize=32)
def _resolve_dnsaddr_txt(hostname: str) -> list[str]:
    if not hostname or not _dnsaddr_resolution_enabled():
        return []

    try:
        import dns.resolver  # type: ignore
    except Exception:
        return []

    qname = f"_dnsaddr.{hostname.strip().strip('.')}"
    try:
        resolver = dns.resolver.Resolver()
        resolver.lifetime = 2.5
        answers = resolver.resolve(qname, "TXT")
    except Exception:
        return []

    out: list[str] = []
    for rdata in answers:
        try:
            txt = "".join(str(rdata).strip().strip('"').split('" "'))
        except Exception:
            continue
        txt = (txt or "").strip()
        if not txt.startswith("dnsaddr="):
            continue
        ma = txt[len("dnsaddr=") :].strip()
        if ma:
            out.append(ma)
    return out


def _expand_dnsaddr_peers(peers: list[str]) -> list[str]:
    if not peers or not _dnsaddr_resolution_enabled():
        return list(peers or [])

    def _expand_one(text: str, *, depth: int, seen_dns: set[str]) -> list[str]:
        text = str(text or "").strip()
        if not text:
            return []
        if depth <= 0 or not text.startswith("/dnsaddr/"):
            return [text]

        if text in seen_dns:
            return [text]
        seen_dns.add(text)

        remainder = text[len("/dnsaddr/") :]
        host = remainder
        peer_id = ""
        if "/p2p/" in remainder:
            host, peer_id = remainder.split("/p2p/", 1)
            peer_id = peer_id.strip().strip("/")
        host = (host or "").strip().strip("/")

        candidates = _resolve_dnsaddr_txt(host)
        if peer_id:
            candidates = [ma for ma in candidates if f"/p2p/{peer_id}" in str(ma)]

        def _candidate_score(addr: str) -> tuple[int, int, int, int]:
            a = str(addr or "")
            has_tcp = 0 if "/tcp/" in a else 1
            is_ws = 1 if ("/ws" in a or "/wss" in a) else 0
            port_pref = 0 if "/tcp/4001" in a else 1
            if a.startswith("/ip4/"):
                net = 0
            elif a.startswith("/ip6/"):
                net = 1
            elif a.startswith("/dns4/"):
                net = 2
            elif a.startswith("/dns6/"):
                net = 3
            else:
                net = 4
            return (has_tcp, is_ws, port_pref, net)

        candidates = sorted([str(ma) for ma in candidates if str(ma).strip()], key=_candidate_score)

        out: list[str] = []
        if not candidates:
            return [text]
        for ma in candidates:
            out.extend(_expand_one(str(ma), depth=depth - 1, seen_dns=seen_dns))
        return out

    expanded: list[str] = []
    for peer_addr in peers:
        expanded.extend(_expand_one(str(peer_addr), depth=3, seen_dns=set()))

    seen: set[str] = set()
    out: list[str] = []
    for ma in expanded:
        if not ma or ma in seen:
            continue
        seen.add(ma)
        out.append(ma)
    return out


def _bootstrap_peers_explicitly_configured() -> bool:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_PEERS")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_PEERS")
    )
    if raw is None:
        return False
    text = str(raw).strip().lower()
    if not text:
        return False
    if text in {"0", "false", "no", "off"}:
        return False
    return True


def _bootstrap_dial_enabled() -> bool:
    # By default, preserve existing behavior: if bootstrap peers are explicitly
    # configured, attempt to dial them as TaskQueue endpoints.
    #
    # Operators/tests that use bootstrap peers *only* for DHT routing-table
    # seeding should set this to 0 to avoid slow protocol negotiation timeouts.
    return _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_BOOTSTRAP_DIAL",
        compat="IPFS_DATASETS_PY_TASK_P2P_BOOTSTRAP_DIAL",
        default=True,
    )


async def _best_effort_connect_multiaddrs(*, host, addrs: list[str]) -> None:
    if not addrs:
        return
    try:
        from multiaddr import Multiaddr
        from libp2p.peer.peerinfo import info_from_p2p_addr
    except Exception:
        return

    for addr in list(addrs or []):
        try:
            peer_info = info_from_p2p_addr(Multiaddr(addr))
            await host.connect(peer_info)
        except Exception:
            continue


def _mdns_port() -> int:
    # Keep backwards-compatible behavior: in py-libp2p, the mDNS discovery
    # implementation expects a port value aligned with the peer's libp2p
    # listen port. Tests that run multiple peers on one host can do so by
    # binding to distinct loopback IPs while sharing the same port.
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_MDNS_PORT")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_MDNS_PORT")
        or os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_PORT")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_LISTEN_PORT")
        or os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_PORT")
        or "9710"
    )
    try:
        return int(str(raw).strip())
    except Exception:
        return 9710


def _mcp_single_port() -> int | None:
    """Return the canonical libp2p port used in MCP single-port deployments."""

    raw = os.environ.get("IPFS_ACCELERATE_PY_MCP_P2P_PORT")
    text = str(raw or "").strip()
    if not text:
        return None
    try:
        port = int(text)
    except Exception:
        return None
    return port if port > 0 else None


def _normalize_peer_multiaddr_port(peer_multiaddr: str) -> str:
    """Best-effort normalize legacy TaskQueue multiaddrs in MCP mode.

    In MCP deployments we standardize on a single libp2p port (typically 9100).
    Older discovery caches may still surface /tcp/9710 addresses; rewriting that
    legacy default avoids persistent dial failures after migrating to MCP port.

    This is intentionally conservative: it only rewrites the legacy default port
    9710 to the MCP canonical port when the latter is explicitly configured.
    """

    ma = str(peer_multiaddr or "").strip()
    if not ma:
        return ma

    desired = _mcp_single_port()
    if not desired or desired == 9710:
        return ma

    # Only rewrite the legacy default port to avoid surprising behavior for
    # custom ports.
    legacy = "/tcp/9710"
    if legacy not in ma:
        return ma

    return ma.replace(legacy, f"/tcp/{int(desired)}", 1)


def _client_listen_host() -> str:
    # Reuse the service listen host env var for consistency in local/LAN setups
    # and tests. Default remains 0.0.0.0.
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_LISTEN_HOST")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_LISTEN_HOST")
        or "0.0.0.0"
    )
    text = str(raw).strip()
    return text or "0.0.0.0"


def _dht_key_for_namespace(ns: str) -> str:
    # The installed py-libp2p KadDHT in this environment expects *string* keys
    # (see KadDHT.put_value/get_value/provide/find_providers signatures).
    # We still keep bytes fallbacks at call sites for older layouts.
    return str(ns or "").strip()


def _default_announce_files() -> list[str]:
    cache_root = os.environ.get("XDG_CACHE_HOME") or os.path.join(os.path.expanduser("~"), ".cache")
    return [
        os.path.join(cache_root, "ipfs_accelerate_py", "task_p2p_announce.json"),
        os.path.join(cache_root, "ipfs_datasets_py", "task_p2p_announce.json"),
        # Common systemd deployment location (see deployments/systemd/*.service)
        "/var/cache/ipfs-accelerate/task_p2p_announce.json",
    ]


def _repo_local_announce_files() -> list[str]:
    # Best-effort: when running from a repo checkout (common for LAN ops),
    # systemd-friendly deployments write announce state under ./state.
    try:
        state_dir = os.path.join(os.getcwd(), "state")
        return [
            os.path.join(state_dir, "task_p2p_announce.json"),
            os.path.join(state_dir, "task_p2p_announce_mcp.json"),
        ]
    except Exception:
        return []


def _read_announce_peer_id_hint() -> str:
    # Used to avoid "discovering" and dialing the local node's own TaskQueue
    # service when using mDNS/DHT/rendezvous for LAN/public discovery.
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE")
    )
    if raw is not None and str(raw).strip().lower() in {"0", "false", "no", "off"}:
        raw = None

    explicit_path = str(raw).strip() if raw is not None else ""
    if explicit_path:
        try:
            if os.path.exists(explicit_path):
                text = open(explicit_path, "r", encoding="utf-8").read().strip()
                info = json.loads(text) if text else {}
                if isinstance(info, dict):
                    pid = str(info.get("peer_id") or "").strip()
                    if pid:
                        return pid
        except Exception:
            pass

    candidates: list[str] = []
    candidates.extend(_repo_local_announce_files())
    candidates.extend(_default_announce_files())

    best_path = ""
    best_mtime = -1.0
    for path in candidates:
        try:
            if not path or not os.path.exists(path):
                continue
            mtime = float(os.path.getmtime(path))
            if mtime > best_mtime:
                best_mtime = mtime
                best_path = path
        except Exception:
            continue

    if not best_path:
        return ""

    try:
        text = open(best_path, "r", encoding="utf-8").read().strip()
        info = json.loads(text) if text else {}
        if isinstance(info, dict):
            return str(info.get("peer_id") or "").strip()
    except Exception:
        return ""
    return ""


def _read_announce_multiaddr() -> str:
    raw = (
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ANNOUNCE_FILE")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_ANNOUNCE_FILE")
    )
    if raw is not None and str(raw).strip().lower() in {"0", "false", "no", "off"}:
        return ""

    explicit_path = str(raw).strip() if raw is not None else ""
    if explicit_path:
        try:
            if os.path.exists(explicit_path):
                text = open(explicit_path, "r", encoding="utf-8").read().strip()
                info = json.loads(text) if text else {}
                if isinstance(info, dict):
                    ma = str(info.get("multiaddr") or "").strip()
                    if ma and "/p2p/" in ma:
                        return ma
        except Exception:
            pass

    candidates: list[str] = []
    candidates.extend(_repo_local_announce_files())
    candidates.extend(_default_announce_files())

    best_path = ""
    best_mtime = -1.0
    for path in candidates:
        try:
            if not path or not os.path.exists(path):
                continue
            mtime = float(os.path.getmtime(path))
            if mtime > best_mtime:
                best_mtime = mtime
                best_path = path
        except Exception:
            continue

    if not best_path:
        return ""

    try:
        text = open(best_path, "r", encoding="utf-8").read().strip()
        info = json.loads(text) if text else {}
        if isinstance(info, dict):
            ma = str(info.get("multiaddr") or "").strip()
            if ma and "/p2p/" in ma:
                return ma
    except Exception:
        return ""
    return ""


async def _read_one_json_line(stream) -> Dict[str, Any]:
    raw = bytearray()
    max_bytes = 1024 * 1024
    while len(raw) < max_bytes:
        chunk = await stream.read(1024)
        if not chunk:
            break
        raw.extend(chunk)
        if b"\n" in chunk:
            break
    try:
        return json.loads((bytes(raw) or b"{}").rstrip(b"\n").decode("utf-8"))
    except Exception:
        return {"ok": False, "error": "invalid_json_response"}


async def _request_over_stream(*, stream, message: Dict[str, Any]) -> Dict[str, Any]:
    token = get_shared_token()
    if token and "token" not in message:
        message = dict(message)
        message["token"] = token
    await stream.write(json.dumps(message).encode("utf-8") + b"\n")
    return await _read_one_json_line(stream)


async def _try_peer_multiaddr(*, host, peer_multiaddr: str, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from multiaddr import Multiaddr
    from libp2p.peer.peerinfo import info_from_p2p_addr

    peer_multiaddr = _normalize_peer_multiaddr_port(peer_multiaddr)
    peer_info = info_from_p2p_addr(Multiaddr(peer_multiaddr))
    peer_text = ""
    try:
        peer_text = peer_info.peer_id.pretty() if hasattr(peer_info.peer_id, "pretty") else str(peer_info.peer_id)
    except Exception:
        peer_text = str(getattr(peer_info, "peer_id", "") or "")

    op = str(message.get("op") or "").strip()
    _dial_debug(
        f"connect begin peer={peer_text} multiaddr={peer_multiaddr} op={op or 'unknown'}"
    )
    try:
        await host.connect(peer_info)
    except Exception as exc:
        _dial_debug(
            f"connect failed peer={peer_text} multiaddr={peer_multiaddr} exc={type(exc).__name__}: {exc}"
        )
        raise

    _dial_debug(f"connect ok peer={peer_text} multiaddr={peer_multiaddr}")
    try:
        stream = await host.new_stream(peer_info.peer_id, [PROTOCOL_V1])
    except Exception as exc:
        _dial_debug(
            f"new_stream failed peer={peer_text} protocol={PROTOCOL_V1} exc={type(exc).__name__}: {exc}"
        )
        raise

    _dial_debug(f"new_stream ok peer={peer_text} protocol={PROTOCOL_V1}")
    try:
        return await _request_over_stream(stream=stream, message=message)
    finally:
        try:
            await stream.close()
        except Exception:
            pass


async def _dial_via_bootstrap(*, host, message: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    # Only attempt direct dialing of "bootstrap" peers when the user explicitly
    # configured them as TaskQueue endpoints. The default public libp2p bootstrap
    # set is meant for DHT routing, not for speaking our application protocol.
    if not _bootstrap_peers_explicitly_configured():
        return None
    if not _bootstrap_dial_enabled():
        return None
    for addr in _parse_bootstrap_peers():
        wait_s = _explicit_addr_cooldown_wait_s(addr)
        if wait_s > 0:
            _retry_metric_inc("dial.bootstrap_cooldown_skip")
            continue
        try:
            resp = await _try_peer_multiaddr(host=host, peer_multiaddr=addr, message=message)
            if isinstance(resp, dict):
                _explicit_addr_cooldown_mark_success(addr)
                return resp
        except Exception:
            _explicit_addr_cooldown_mark_failure(addr)
            continue
    return None


async def _dial_via_announce_file(
    *,
    host,
    message: Dict[str, Any],
    require_peer_id: str = "",
) -> Optional[Dict[str, Any]]:
    ma = _read_announce_multiaddr()
    if not ma:
        return None

    # If the caller requested a specific peer, avoid dialing the announce hint
    # when it clearly doesn't match.
    if require_peer_id:
        try:
            pid = str(ma).rsplit("/p2p/", 1)[-1].strip()
            if pid and pid != require_peer_id:
                return None
        except Exception:
            pass

    announce_wait_s = _explicit_addr_cooldown_wait_s(ma)
    if announce_wait_s > 0:
        _retry_metric_inc("dial.announce_cooldown_skip")
        return None

    try:
        resp = await _try_peer_multiaddr(host=host, peer_multiaddr=ma, message=message)
        if isinstance(resp, dict):
            _explicit_addr_cooldown_mark_success(ma)
            pid = _multiaddr_peer_id(ma)
            if pid:
                _cache_set_multiaddr(pid, ma)
            return resp
        return None
    except Exception:
        _retry_metric_inc("dial.announce_failed")
        _explicit_addr_cooldown_mark_failure(ma)
        return None


async def _dial_via_mdns(*, host, message: Dict[str, Any], require_peer_id: str = "") -> Dict[str, Any]:
    import anyio

    if not _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_MDNS",
        compat="IPFS_DATASETS_PY_TASK_P2P_MDNS",
        default=True,
    ):
        return {"ok": False, "error": "mdns_disabled"}

    try:
        from libp2p.discovery.mdns.mdns import MDNSDiscovery
        from libp2p.abc import PeerInfo
    except Exception as exc:
        return {"ok": False, "error": f"mdns_unavailable: {exc}"}

    discover_timeout_s = float(
        os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_DISCOVERY_TIMEOUT_S")
        or os.environ.get("IPFS_DATASETS_PY_TASK_P2P_DISCOVERY_TIMEOUT_S", "5.0")
    )

    mdns = MDNSDiscovery(host.get_network(), port=_mdns_port())
    try:
        mdns.start()
    except Exception as exc:
        return {"ok": False, "error": f"mdns_start_failed: {exc}"}

    # Avoid dialing the local node's own TaskQueue service if it is advertising
    # on the LAN (common when both client+server run on the same box).
    exclude_peer_id = _read_announce_peer_id_hint()

    try:
        deadline = anyio.current_time() + max(0.1, discover_timeout_s)
        attempted: set[str] = set()
        max_attempts_per_poll = _mdns_max_attempts_per_poll()

        while anyio.current_time() < deadline:
            discovered_peer_ids = list(mdns.listener.discovered_services.values())
            preferred_peer_id = "" if require_peer_id else _get_last_success_peer(max_age_s=300.0)
            if preferred_peer_id:
                discovered_peer_ids = sorted(
                    discovered_peer_ids,
                    key=lambda pid: 0
                    if ((pid.pretty() if hasattr(pid, "pretty") else str(pid)) == preferred_peer_id)
                    else 1,
                )

            attempts_this_poll = 0
            for pid in discovered_peer_ids:
                if attempts_this_poll >= max_attempts_per_poll:
                    _retry_metric_inc("dial.mdns_attempt_limit")
                    break

                pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid)

                if pid_text in attempted:
                    continue
                if exclude_peer_id and pid_text == exclude_peer_id:
                    attempted.add(pid_text)
                    continue
                if require_peer_id and pid_text != require_peer_id:
                    continue

                addrs = host.get_network().peerstore.addrs(pid)
                attempted.add(pid_text)
                if not addrs:
                    continue

                dial_addrs = []
                dial_ma = _pick_best_peer_multiaddr_text(addrs, pid_text)
                for a in list(addrs or []):
                    # Filter out undialable wildcard addresses.
                    if str(a).startswith("/ip4/0.0.0.0/") or str(a).startswith("/ip6/::/"):
                        continue
                    dial_addrs.append(a)

                if not dial_addrs:
                    continue

                peer_info = PeerInfo(peer_id=pid, addrs=dial_addrs)
                remote_ref = RemoteQueue(peer_id=pid_text, multiaddr=(dial_ma or ""))
                remote_wait_s = _remote_cooldown_wait_s(remote_ref)
                if remote_wait_s > 0:
                    _retry_metric_inc("dial.mdns_cooldown_skip")
                    continue
                attempts_this_poll += 1
                try:
                    await host.connect(peer_info)
                    stream = await host.new_stream(peer_info.peer_id, [PROTOCOL_V1])
                    try:
                        resp = await _request_over_stream(stream=stream, message=message)
                        _remote_cooldown_mark_success(remote_ref)
                        _mark_last_success_peer(pid_text)
                        if dial_ma:
                            _cache_set_multiaddr(pid_text, dial_ma)
                        return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}
                    finally:
                        try:
                            await stream.close()
                        except Exception:
                            pass
                except Exception:
                    _remote_cooldown_mark_failure(remote_ref)
                    continue

            await anyio.sleep(0.1)

        return {"ok": False, "error": "discovery_timeout"}
    finally:
        try:
            try:
                mdns.listener.stop()
            except Exception:
                pass
            mdns.stop()
        except Exception:
            pass


async def _dial_via_rendezvous(*, host, message: Dict[str, Any], require_peer_id: str = "") -> Optional[Dict[str, Any]]:
    if not _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS",
        compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS",
        default=True,
    ):
        return None

    ns = _env_str(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS",
        compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS",
        default="ipfs-accelerate-task-queue",
    )

    exclude_peer_id = "" if require_peer_id else _read_announce_peer_id_hint()

    candidates = [
        ("libp2p.discovery.rendezvous.rendezvous", "RendezvousClient"),
        ("libp2p.discovery.rendezvous", "RendezvousClient"),
        ("libp2p.rendezvous", "RendezvousClient"),
    ]
    for module_name, symbol in candidates:
        try:
            mod = __import__(module_name, fromlist=[symbol])
            cls = getattr(mod, symbol)
            cli = cls(host)
            discover = getattr(cli, "discover", None)
            if not callable(discover):
                continue

            # Newer py-libp2p returns (peers, cookie).
            cookie: bytes = b""
            try:
                peers, cookie = await discover(ns, limit=100, cookie=cookie)
            except TypeError:
                # Older implementations may have a simpler signature.
                found = discover(ns)
                peers = list(found or [])

            for peer_info in list(peers or []):
                try:
                    pid = getattr(peer_info, "peer_id", None)
                    pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid or "")
                    if require_peer_id and pid_text != require_peer_id:
                        continue
                    if exclude_peer_id and pid_text and pid_text == exclude_peer_id:
                        continue
                    await host.connect(peer_info)
                    stream = await host.new_stream(peer_info.peer_id, [PROTOCOL_V1])
                    try:
                        resp = await _request_over_stream(stream=stream, message=message)
                        return resp if isinstance(resp, dict) else None
                    finally:
                        try:
                            await stream.close()
                        except Exception:
                            pass
                except Exception:
                    continue
        except Exception:
            continue
    return None


async def _dial_via_dht(*, host, message: Dict[str, Any], require_peer_id: str = "") -> Optional[Dict[str, Any]]:
    if not _env_bool(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_DHT",
        compat="IPFS_DATASETS_PY_TASK_P2P_DHT",
        default=True,
    ):
        return None

    ns = _env_str(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS",
        compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS",
        default="ipfs-accelerate-task-queue",
    )
    ns_key = _dht_key_for_namespace(ns)
    ns_key_bytes = ns_key.encode("utf-8") if ns_key else b""

    exclude_peer_id = "" if require_peer_id else _read_announce_peer_id_hint()

    candidates = [
        ("libp2p.kad_dht.kad_dht", "KadDHT"),
        ("libp2p.kad_dht", "KadDHT"),
    ]
    for module_name, symbol in candidates:
        try:
            mod = __import__(module_name, fromlist=[symbol])
            cls = getattr(mod, symbol)

            # KadDHT requires a DHTMode in some builds.
            try:
                from libp2p.kad_dht.kad_dht import DHTMode  # type: ignore

                dht = cls(host, DHTMode.CLIENT)
            except Exception:
                dht = cls(host)

            from libp2p.tools.async_service.trio_service import background_trio_service

            async with background_trio_service(dht):
                # Seed routing table: connect to bootstrap peers.
                try:
                    await _best_effort_connect_multiaddrs(host=host, addrs=_parse_bootstrap_peers())
                except Exception:
                    pass

                # Some KadDHT implementations require explicit start/bootstrap.
                for method_name in ("start", "bootstrap"):
                    try:
                        fn = getattr(dht, method_name, None)
                        if not callable(fn):
                            continue
                        maybe = fn()
                        if hasattr(maybe, "__await__"):
                            await maybe
                    except Exception:
                        continue

                # Prefer deterministic namespace -> multiaddr record when doing
                # namespace-based discovery.
                if not require_peer_id:
                    get_value = getattr(dht, "get_value", None)
                    if callable(get_value):
                        raw = None
                        try:
                            raw = await get_value(_dht_value_record_key(ns_key))
                        except Exception:
                            raw = None

                        value_text = ""
                        try:
                            if isinstance(raw, (bytes, bytearray)):
                                value_text = bytes(raw).decode("utf-8", errors="ignore")
                            elif isinstance(raw, str):
                                value_text = raw
                        except Exception:
                            value_text = ""

                        if value_text:
                            try:
                                data = json.loads(value_text)
                            except Exception:
                                data = None
                            if isinstance(data, dict):
                                ma = str(data.get("multiaddr") or "").strip()
                                if ma:
                                    try:
                                        resp = await _try_peer_multiaddr(host=host, peer_multiaddr=ma, message=message)
                                        if isinstance(resp, dict):
                                            return resp
                                    except Exception:
                                        pass

                if require_peer_id:
                    find_peer = getattr(dht, "find_peer", None)
                    if not callable(find_peer):
                        continue
                    try:
                        peer_info = await find_peer(require_peer_id)
                    except Exception:
                        peer_info = None
                    if not peer_info:
                        continue
                    try:
                        resp = await _try_peer_info(host=host, peer_info=peer_info, message=message)
                        if isinstance(resp, dict):
                            return resp
                    except Exception:
                        continue
                    continue

                # Namespace-based provider discovery: ask DHT for providers and try them.
                providers: list[Any] = []
                find_providers = getattr(dht, "find_providers", None)
                if callable(find_providers):
                    for key_candidate in (ns_key, ns_key_bytes):
                        try:
                            providers = list(await find_providers(key_candidate, 20) or [])
                            break
                        except Exception:
                            continue

                for peer_info in list(providers or []):
                    try:
                        pid_text = ""
                        try:
                            pid = getattr(peer_info, "peer_id", None)
                            pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid or "")
                        except Exception:
                            pid_text = ""
                        if exclude_peer_id and pid_text and pid_text == exclude_peer_id:
                            continue
                        resp = await _try_peer_info(host=host, peer_info=peer_info, message=message)
                        if isinstance(resp, dict):
                            return resp
                    except Exception:
                        continue
        except Exception:
            continue
    return None


async def _dial_and_request(
    *,
    remote: RemoteQueue,
    message: Dict[str, Any],
    dial_timeout_s: float = 20.0,
    allow_broad_discovery_override: bool | None = None,
) -> Dict[str, Any]:
    if not _have_libp2p():
        raise RuntimeError("libp2p is not installed")

    import anyio
    import inspect
    from ipfs_accelerate_py.github_cli.libp2p_compat import ensure_libp2p_compatible
    from libp2p import new_host
    from multiaddr import Multiaddr
    from libp2p.tools.async_service import background_trio_service

    if not ensure_libp2p_compatible():
        raise RuntimeError(
            "libp2p is installed but dependency compatibility patches could not be applied. "
            "This environment likely has an incompatible `multihash` module."
        )

    host_obj = new_host()
    host = await host_obj if inspect.isawaitable(host_obj) else host_obj

    resp: Optional[Dict[str, Any]] = None
    try:
        async with background_trio_service(host.get_network()):
            await host.get_network().listen(Multiaddr(f"/ip4/{_client_listen_host()}/tcp/0"))

            with anyio.fail_after(float(dial_timeout_s)):
                require_peer_id = (remote.peer_id or "").strip()
                explicit_multiaddr = (remote.multiaddr or "").strip()

                if explicit_multiaddr:
                    # Prefer explicit address first, but do not hard-fail this
                    # attempt if the address is stale or transport/security
                    # negotiation fails under load.
                    explicit_wait_s = _explicit_addr_cooldown_wait_s(explicit_multiaddr)
                    if explicit_wait_s > 0:
                        _retry_metric_inc("dial.explicit_multiaddr_cooldown_skip")
                        _dial_debug(
                            "explicit multiaddr cooldown skip; falling back to discovery paths "
                            f"peer_id={require_peer_id or '<none>'} wait_s={explicit_wait_s:.3f}"
                        )
                        resp = None
                    else:
                        try:
                            resp = await _try_peer_multiaddr(
                                host=host,
                                peer_multiaddr=explicit_multiaddr,
                                message=message,
                            )  # type: ignore[assignment]
                            if isinstance(resp, dict):
                                _explicit_addr_cooldown_mark_success(explicit_multiaddr)
                        except Exception as exc:
                            _retry_metric_inc("dial.explicit_multiaddr_failed")
                            _explicit_addr_cooldown_mark_failure(explicit_multiaddr)
                            _dial_debug(
                                "explicit multiaddr dial failed; falling back to discovery paths "
                                f"peer_id={require_peer_id or '<none>'} err={type(exc).__name__}: {exc}"
                            )
                            resp = None

                # Whether we started with explicit multiaddr or not, attempt
                # peer-id based paths when no response has been obtained.
                if not isinstance(resp, dict):
                    # If we've recently discovered this peer via mDNS/DHT/etc in this
                    # process, try the cached multiaddr first to avoid rediscovery
                    # races during submit/wait loops.
                    if require_peer_id:
                        cached_ma = _cache_get_multiaddr(require_peer_id)
                        if cached_ma:
                            cached_wait_s = _explicit_addr_cooldown_wait_s(cached_ma)
                            if cached_wait_s > 0:
                                _retry_metric_inc("dial.cache_multiaddr_cooldown_skip")
                            else:
                                try:
                                    resp = await _try_peer_multiaddr(host=host, peer_multiaddr=cached_ma, message=message)
                                    if isinstance(resp, dict):
                                        _explicit_addr_cooldown_mark_success(cached_ma)
                                        return resp
                                except Exception:
                                    _retry_metric_inc("dial.cache_multiaddr_failed")
                                    _explicit_addr_cooldown_mark_failure(cached_ma)
                                _cache_del_multiaddr(require_peer_id)

                    # Zero-config: if a local service is running, it writes an
                    # announce file in XDG cache; dial it first.
                    ann = await _dial_via_announce_file(host=host, message=message, require_peer_id=require_peer_id)
                    if isinstance(ann, dict):
                        resp = ann
                    else:
                        # Broad discovery fallback can be expensive under high
                        # throughput. For explicit multiaddr targets, keep
                        # fallback lightweight by default (cache + announce)
                        # unless explicitly enabled.
                        allow_broad_discovery = True
                        if explicit_multiaddr:
                            allow_broad_discovery = _env_bool(
                                primary="IPFS_ACCELERATE_PY_TASK_P2P_EXPLICIT_DISCOVERY_FALLBACK",
                                compat="IPFS_DATASETS_PY_TASK_P2P_EXPLICIT_DISCOVERY_FALLBACK",
                                default=False,
                            )
                        if allow_broad_discovery_override is not None:
                            allow_broad_discovery = bool(allow_broad_discovery) and bool(allow_broad_discovery_override)

                        if allow_broad_discovery:
                            # Then try cross-subnet mechanisms, and finally LAN mDNS.
                            boot = await _dial_via_bootstrap(host=host, message=message)
                            if isinstance(boot, dict):
                                resp = boot
                            else:
                                rv = await _dial_via_rendezvous(host=host, message=message, require_peer_id=require_peer_id)
                                if isinstance(rv, dict):
                                    resp = rv
                                else:
                                    dht = await _dial_via_dht(host=host, message=message, require_peer_id=require_peer_id)
                                    if isinstance(dht, dict):
                                        resp = dht
                                    else:
                                        resp = await _dial_via_mdns(host=host, message=message, require_peer_id=require_peer_id)
                        else:
                            _retry_metric_inc("dial.broad_discovery_skipped")
                            if explicit_multiaddr:
                                _retry_metric_inc("dial.explicit_discovery_skipped")
    except BaseExceptionGroup as exc:
        # Under high concurrency, background_trio_service teardown can raise
        # grouped transport errors after a successful request/response cycle.
        if isinstance(resp, dict):
            _dial_debug(f"background service teardown raised {type(exc).__name__}; preserving successful response")
        else:
            raise

    try:
        await host.close()
    except Exception:
        pass

    if not isinstance(resp, dict):
        raise RuntimeError("p2p request failed: no response")
    return resp


async def discover_status(
    *,
    remote: RemoteQueue,
    timeout_s: float = 10.0,
    detail: bool = False,
) -> Dict[str, Any]:
    """Attempt to discover and contact a TaskQueue peer, returning a trace.

    This is intended for debugging / operator UX.

    Returns:
        {
          "ok": bool,
          "result": {..status response..} | None,
                    "nat": {..} | None,
          "attempts": [
             {"method": "announce-file", "multiaddr": "...", "peer_id": "...", "ok": bool, "error": "..."},
             ...
          ]
        }
    """

    if not _have_libp2p():
        raise RuntimeError("libp2p is not installed")

    import anyio
    import inspect
    import time

    from ipfs_accelerate_py.github_cli.libp2p_compat import ensure_libp2p_compatible
    from libp2p import new_host
    from libp2p.tools.async_service import background_trio_service
    from multiaddr import Multiaddr

    if not ensure_libp2p_compatible():
        raise RuntimeError(
            "libp2p is installed but dependency compatibility patches could not be applied. "
            "This environment likely has an incompatible `multihash` module."
        )

    message: Dict[str, Any] = {"op": "status", "timeout_s": float(timeout_s), "detail": bool(detail)}
    require_peer_id = (remote.peer_id or "").strip()
    attempts: list[Dict[str, Any]] = []

    def _nat_from_resp(resp: Dict[str, Any] | None) -> Dict[str, Any] | None:
        if not isinstance(resp, dict):
            return None
        nat = resp.get("nat")
        return nat if isinstance(nat, dict) else None

    host_obj = new_host()
    host = await host_obj if inspect.isawaitable(host_obj) else host_obj

    async with background_trio_service(host.get_network()):
        await host.get_network().listen(Multiaddr(f"/ip4/{_client_listen_host()}/tcp/0"))

        deadline = time.time() + max(0.1, float(timeout_s))

        async def _record(
            *,
            method: str,
            ok: bool,
            peer_id: str = "",
            multiaddr: str = "",
            error: str = "",
            response: Dict[str, Any] | None = None,
        ) -> None:
            attempts.append(
                {
                    "method": str(method),
                    "ok": bool(ok),
                    "peer_id": str(peer_id or "").strip(),
                    "multiaddr": str(multiaddr or "").strip(),
                    "error": str(error or "").strip(),
                    "response": response if isinstance(response, dict) else None,
                }
            )

        async def _try_multiaddr(method: str, ma: str) -> Optional[Dict[str, Any]]:
            ma = str(ma or "").strip()
            if not ma:
                await _record(method=method, ok=False, error="missing_multiaddr")
                return None
            pid = _multiaddr_peer_id(ma)
            if require_peer_id and pid and pid != require_peer_id:
                await _record(method=method, ok=False, peer_id=pid, multiaddr=ma, error="peer_id_mismatch")
                return None
            try:
                resp = await _try_peer_multiaddr(host=host, peer_multiaddr=ma, message=message)
                if isinstance(resp, dict) and resp.get("ok"):
                    await _record(
                        method=method,
                        ok=True,
                        peer_id=str(resp.get("peer_id") or pid),
                        multiaddr=ma,
                        response=resp,
                    )
                    return resp
                await _record(
                    method=method,
                    ok=False,
                    peer_id=str((resp or {}).get("peer_id") or pid),
                    multiaddr=ma,
                    error=str((resp or {}).get("error") or "request_failed"),
                    response=resp if isinstance(resp, dict) else None,
                )
                return None
            except Exception as exc:
                await _record(method=method, ok=False, peer_id=pid, multiaddr=ma, error=str(exc))
                return None

        # 1) Explicit multiaddr
        if (remote.multiaddr or "").strip():
            resp = await _try_multiaddr("explicit", str(remote.multiaddr))
            return {
                "ok": bool(resp and resp.get("ok")),
                "result": resp,
                "nat": _nat_from_resp(resp),
                "attempts": attempts,
            }

        # 2) Announce file hint (local)
        ann_ma = _read_announce_multiaddr()
        if ann_ma:
            resp = await _try_multiaddr("announce-file", ann_ma)
            if resp is not None:
                return {"ok": True, "result": resp, "nat": _nat_from_resp(resp), "attempts": attempts}
        else:
            await _record(method="announce-file", ok=False, error="no_announce_multiaddr")

        # 3) Direct dialing of explicitly configured bootstrap peers
        if _bootstrap_peers_explicitly_configured() and _bootstrap_dial_enabled():
            for ma in _parse_bootstrap_peers():
                if time.time() > deadline:
                    await _record(method="bootstrap", ok=False, error="timeout")
                    break
                resp = await _try_multiaddr("bootstrap", ma)
                if resp is not None:
                    return {"ok": True, "result": resp, "nat": _nat_from_resp(resp), "attempts": attempts}
        elif _bootstrap_peers_explicitly_configured() and not _bootstrap_dial_enabled():
            await _record(method="bootstrap", ok=False, error="disabled")
        else:
            await _record(method="bootstrap", ok=False, error="not_explicitly_configured")

        # 4) Rendezvous discovery
        if time.time() <= deadline and _env_bool(
            primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS",
            compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS",
            default=True,
        ):
            ns = _env_str(
                primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS",
                compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS",
                default="ipfs-accelerate-task-queue",
            )
            candidates = [
                ("libp2p.discovery.rendezvous.rendezvous", "RendezvousClient"),
                ("libp2p.discovery.rendezvous", "RendezvousClient"),
                ("libp2p.rendezvous", "RendezvousClient"),
            ]
            rendezvous_attempted = False
            for module_name, symbol in candidates:
                try:
                    mod = __import__(module_name, fromlist=[symbol])
                    cls = getattr(mod, symbol)
                    cli = cls(host)
                    discover = getattr(cli, "discover", None)
                    if not callable(discover):
                        continue
                    rendezvous_attempted = True

                    cookie: bytes = b""
                    try:
                        peers, cookie = await discover(ns, limit=100, cookie=cookie)
                    except TypeError:
                        found = discover(ns)
                        peers = list(found or [])

                    for peer_info in list(peers or []):
                        if time.time() > deadline:
                            await _record(method="rendezvous", ok=False, error="timeout")
                            break

                        pid_text = ""
                        try:
                            pid = getattr(peer_info, "peer_id", None)
                            pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid or "")
                        except Exception:
                            pid_text = ""
                        if require_peer_id and pid_text and pid_text != require_peer_id:
                            await _record(method="rendezvous", ok=False, peer_id=pid_text, error="peer_id_mismatch")
                            continue

                        resp = await _try_peer_info(host=host, peer_info=peer_info, message=message)
                        ma = ""
                        addrs = _best_effort_peerinfo_multiaddrs(peer_info)
                        if addrs:
                            ma = addrs[0]
                        if isinstance(resp, dict) and resp.get("ok"):
                            await _record(
                                method="rendezvous",
                                ok=True,
                                peer_id=str(resp.get("peer_id") or pid_text),
                                multiaddr=ma,
                                response=resp,
                            )
                            return {"ok": True, "result": resp, "nat": _nat_from_resp(resp), "attempts": attempts}
                        await _record(
                            method="rendezvous",
                            ok=False,
                            peer_id=str((resp or {}).get("peer_id") or pid_text),
                            multiaddr=ma,
                            error=str((resp or {}).get("error") or "request_failed"),
                            response=resp if isinstance(resp, dict) else None,
                        )
                except Exception:
                    continue

            if not rendezvous_attempted:
                await _record(method="rendezvous", ok=False, error="unavailable")
        else:
            await _record(method="rendezvous", ok=False, error="disabled_or_timeout")

        # 5) DHT provider discovery
        if time.time() <= deadline and _env_bool(
            primary="IPFS_ACCELERATE_PY_TASK_P2P_DHT",
            compat="IPFS_DATASETS_PY_TASK_P2P_DHT",
            default=True,
        ):
            ns = _env_str(
                primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS",
                compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS",
                default="ipfs-accelerate-task-queue",
            )
            ns_key = _dht_key_for_namespace(ns)
            candidates = [
                ("libp2p.kad_dht.kad_dht", "KadDHT"),
                ("libp2p.kad_dht", "KadDHT"),
            ]

            dht_attempted = False
            for module_name, symbol in candidates:
                try:
                    mod = __import__(module_name, fromlist=[symbol])
                    cls = getattr(mod, symbol)
                    dht_attempted = True

                    try:
                        from libp2p.kad_dht.kad_dht import DHTMode  # type: ignore

                        dht = cls(host, DHTMode.CLIENT)
                    except Exception:
                        dht = cls(host)

                    import trio
                    from libp2p.tools.async_service.trio_service import background_trio_service as bg_trio

                    async def _run_dht_service() -> None:
                        async with bg_trio(dht):
                            await trio.sleep_forever()

                    async with anyio.create_task_group() as tg:
                        tg.start_soon(_run_dht_service)
                        try:
                            await _best_effort_connect_multiaddrs(host=host, addrs=_parse_bootstrap_peers())
                        except Exception:
                            pass

                        # Some KadDHT implementations require explicit
                        # start/bootstrap to populate the routing table.
                        try:
                            start = getattr(dht, "start", None)
                            if callable(start):
                                maybe = start()
                                if hasattr(maybe, "__await__"):
                                    await maybe
                        except Exception:
                            pass
                        try:
                            bootstrap = getattr(dht, "bootstrap", None)
                            if callable(bootstrap):
                                maybe = bootstrap()
                                if hasattr(maybe, "__await__"):
                                    await maybe
                        except Exception:
                            pass

                        if require_peer_id:
                            find_peer = getattr(dht, "find_peer", None)
                            if not callable(find_peer):
                                await _record(method="dht", ok=False, error="find_peer_unavailable")
                            else:
                                peer_info = await find_peer(require_peer_id)
                                if not peer_info:
                                    await _record(
                                        method="dht",
                                        ok=False,
                                        peer_id=require_peer_id,
                                        error="peer_not_found",
                                    )
                                else:
                                    resp = await _try_peer_info(host=host, peer_info=peer_info, message=message)
                                    addrs = _best_effort_peerinfo_multiaddrs(peer_info)
                                    ma = addrs[0] if addrs else ""
                                    if isinstance(resp, dict) and resp.get("ok"):
                                        await _record(
                                            method="dht",
                                            ok=True,
                                            peer_id=str(resp.get("peer_id") or require_peer_id),
                                            multiaddr=ma,
                                            response=resp,
                                        )
                                        tg.cancel_scope.cancel()
                                        return {"ok": True, "result": resp, "nat": _nat_from_resp(resp), "attempts": attempts}
                                    await _record(
                                        method="dht",
                                        ok=False,
                                        peer_id=str((resp or {}).get("peer_id") or require_peer_id),
                                        multiaddr=ma,
                                        error=str((resp or {}).get("error") or "request_failed"),
                                        response=resp if isinstance(resp, dict) else None,
                                    )
                        else:
                            find_providers = getattr(dht, "find_providers", None)
                            if not callable(find_providers):
                                await _record(method="dht", ok=False, error="find_providers_unavailable")
                            else:
                                get_value = getattr(dht, "get_value", None)

                                # Retry until deadline: DHT routing tables can
                                # take a couple seconds to populate in tiny
                                # local test networks.
                                while time.time() <= deadline:
                                    providers = None
                                    try:
                                        providers = await find_providers(ns_key, 20)
                                    except Exception:
                                        try:
                                            providers = await find_providers(ns, 20)
                                        except Exception:
                                            providers = None

                                    if providers:
                                        for peer_info in list(providers or []):
                                            if time.time() > deadline:
                                                await _record(method="dht", ok=False, error="timeout")
                                                break
                                            resp = await _try_peer_info(host=host, peer_info=peer_info, message=message)
                                            addrs = _best_effort_peerinfo_multiaddrs(peer_info)
                                            ma = addrs[0] if addrs else ""
                                            pid_text = ""
                                            try:
                                                pid = getattr(peer_info, "peer_id", None)
                                                pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid or "")
                                            except Exception:
                                                pid_text = ""
                                            if isinstance(resp, dict) and resp.get("ok"):
                                                await _record(
                                                    method="dht",
                                                    ok=True,
                                                    peer_id=str(resp.get("peer_id") or pid_text),
                                                    multiaddr=ma,
                                                    response=resp,
                                                )
                                                tg.cancel_scope.cancel()
                                                return {
                                                    "ok": True,
                                                    "result": resp,
                                                    "nat": _nat_from_resp(resp),
                                                    "attempts": attempts,
                                                }
                                            await _record(
                                                method="dht",
                                                ok=False,
                                                peer_id=str((resp or {}).get("peer_id") or pid_text),
                                                multiaddr=ma,
                                                error=str((resp or {}).get("error") or "request_failed"),
                                                response=resp if isinstance(resp, dict) else None,
                                            )

                                    # Provider records can be flaky in tiny test
                                    # networks. Fall back to a deterministic DHT
                                    # key that stores {peer_id, multiaddr}.
                                    if callable(get_value):
                                        raw = None
                                        try:
                                            raw = await get_value(_dht_value_record_key(ns_key))
                                        except Exception:
                                            raw = None

                                        value_text = ""
                                        try:
                                            if isinstance(raw, (bytes, bytearray)):
                                                value_text = bytes(raw).decode("utf-8", errors="ignore")
                                            elif isinstance(raw, str):
                                                value_text = raw
                                        except Exception:
                                            value_text = ""

                                        if value_text:
                                            data = None
                                            try:
                                                data = json.loads(value_text)
                                            except Exception:
                                                data = None
                                            if isinstance(data, dict):
                                                ma = str(data.get("multiaddr") or "").strip()
                                                if ma:
                                                    resp = await _try_peer_multiaddr(
                                                        host=host,
                                                        peer_multiaddr=ma,
                                                        message=message,
                                                    )
                                                    if isinstance(resp, dict) and resp.get("ok"):
                                                        await _record(
                                                            method="dht",
                                                            ok=True,
                                                            peer_id=str(resp.get("peer_id") or ""),
                                                            multiaddr=ma,
                                                            response=resp,
                                                        )
                                                        tg.cancel_scope.cancel()
                                                        return {
                                                            "ok": True,
                                                            "result": resp,
                                                            "nat": _nat_from_resp(resp),
                                                            "attempts": attempts,
                                                        }

                                    await anyio.sleep(0.2)

                                await _record(method="dht", ok=False, error="no_providers")

                        tg.cancel_scope.cancel()
                except Exception:
                    continue

            if not dht_attempted:
                await _record(method="dht", ok=False, error="unavailable")
        else:
            await _record(method="dht", ok=False, error="disabled_or_timeout")

        # 6) LAN mDNS
        if time.time() <= deadline:
            try:
                mdns_resp = await _dial_via_mdns(host=host, message=message, require_peer_id=require_peer_id)
                if isinstance(mdns_resp, dict) and mdns_resp.get("ok"):
                    await _record(
                        method="mdns",
                        ok=True,
                        peer_id=str(mdns_resp.get("peer_id") or ""),
                        response=mdns_resp,
                    )
                    return {"ok": True, "result": mdns_resp, "nat": _nat_from_resp(mdns_resp), "attempts": attempts}
                await _record(
                    method="mdns",
                    ok=False,
                    peer_id=str((mdns_resp or {}).get("peer_id") or ""),
                    error=str((mdns_resp or {}).get("error") or "request_failed"),
                    response=mdns_resp if isinstance(mdns_resp, dict) else None,
                )
            except Exception as exc:
                await _record(method="mdns", ok=False, error=str(exc))
        else:
            await _record(method="mdns", ok=False, error="timeout")

    try:
        await host.close()
    except Exception:
        pass

    ok = any(a.get("ok") for a in attempts)
    result = None
    for a in attempts:
        if a.get("ok") and isinstance(a.get("response"), dict):
            result = a.get("response")
            break
    return {"ok": bool(ok), "result": result, "nat": _nat_from_resp(result), "attempts": attempts}


async def discover_multiaddr_via_mdns(*, peer_id: str, timeout_s: float = 10.0) -> str:
    """Discover a peer's dialable multiaddr via mDNS.

    This is intended as a one-time resolver for callers that want to rely on
    mDNS for zero-config peer discovery, but then dial by explicit multiaddr for
    repeated RPCs (submit/wait loops).
    """

    if not _have_libp2p():
        raise RuntimeError("libp2p is not installed")

    import anyio
    import inspect

    from ipfs_accelerate_py.github_cli.libp2p_compat import ensure_libp2p_compatible
    from libp2p import new_host
    from libp2p.tools.async_service import background_trio_service
    from multiaddr import Multiaddr

    if not ensure_libp2p_compatible():
        raise RuntimeError(
            "libp2p is installed but dependency compatibility patches could not be applied. "
            "This environment likely has an incompatible `multihash` module."
        )

    require_peer_id = str(peer_id or "").strip()
    if not require_peer_id:
        raise ValueError("peer_id is required")

    try:
        from libp2p.discovery.mdns.mdns import MDNSDiscovery
    except Exception as exc:
        raise RuntimeError(f"mdns_unavailable: {exc}")

    host_obj = new_host()
    host = await host_obj if inspect.isawaitable(host_obj) else host_obj

    async with background_trio_service(host.get_network()):
        await host.get_network().listen(Multiaddr(f"/ip4/{_client_listen_host()}/tcp/0"))

        mdns = MDNSDiscovery(host.get_network(), port=_mdns_port())
        try:
            mdns.start()
        except Exception as exc:
            raise RuntimeError(f"mdns_start_failed: {exc}")

        try:
            deadline = anyio.current_time() + max(0.1, float(timeout_s))
            while anyio.current_time() < deadline:
                for pid in list(mdns.listener.discovered_services.values()):
                    pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid)
                    if pid_text != require_peer_id:
                        continue
                    addrs = host.get_network().peerstore.addrs(pid)
                    ma = _pick_best_peer_multiaddr_text(addrs, pid_text)
                    if ma:
                        _cache_set_multiaddr(pid_text, ma)
                        return ma
                await anyio.sleep(0.1)
        finally:
            try:
                try:
                    mdns.listener.stop()
                except Exception:
                    pass
                mdns.stop()
            except Exception:
                pass

    try:
        await host.close()
    except Exception:
        pass
    return ""


def discover_multiaddr_via_mdns_sync(*, peer_id: str, timeout_s: float = 10.0) -> str:
    import trio

    result: str = ""

    async def _main() -> None:
        nonlocal result
        result = await discover_multiaddr_via_mdns(peer_id=str(peer_id), timeout_s=float(timeout_s))

    trio.run(_main)
    return str(result or "")


async def discover_peers_via_mdns(*, timeout_s: float = 10.0, limit: int = 10, exclude_self: bool = True) -> list[RemoteQueue]:
    """Discover peers on the LAN via mDNS and return dialable RemoteQueue targets.

    Notes:
    - Returns best-effort results; order is discovery order.
    - When `exclude_self=True`, excludes the local peer-id if available via the
      local announce file hint.
    """

    if not _have_libp2p():
        raise RuntimeError("libp2p is not installed")

    import anyio
    import inspect

    from ipfs_accelerate_py.github_cli.libp2p_compat import ensure_libp2p_compatible
    from libp2p import new_host
    from libp2p.tools.async_service import background_trio_service
    from multiaddr import Multiaddr

    if not ensure_libp2p_compatible():
        raise RuntimeError(
            "libp2p is installed but dependency compatibility patches could not be applied. "
            "This environment likely has an incompatible `multihash` module."
        )

    try:
        from libp2p.discovery.mdns.mdns import MDNSDiscovery
    except Exception as exc:
        raise RuntimeError(f"mdns_unavailable: {exc}")

    limit = max(1, int(limit))
    exclude_peer_id = _read_announce_peer_id_hint() if bool(exclude_self) else ""

    host_obj = new_host()
    host = await host_obj if inspect.isawaitable(host_obj) else host_obj

    found: dict[str, str] = {}

    async with background_trio_service(host.get_network()):
        await host.get_network().listen(Multiaddr(f"/ip4/{_client_listen_host()}/tcp/0"))

        mdns = MDNSDiscovery(host.get_network(), port=_mdns_port())
        try:
            mdns.start()
        except Exception as exc:
            raise RuntimeError(f"mdns_start_failed: {exc}")

        try:
            deadline = anyio.current_time() + max(0.1, float(timeout_s))
            while anyio.current_time() < deadline and len(found) < limit:
                for pid in list(mdns.listener.discovered_services.values()):
                    pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid)
                    pid_text = str(pid_text or "").strip()
                    if not pid_text or pid_text in found:
                        continue
                    if exclude_peer_id and pid_text == exclude_peer_id:
                        continue

                    addrs = host.get_network().peerstore.addrs(pid)
                    dial_ma = _pick_best_peer_multiaddr_text(addrs, pid_text)
                    if not dial_ma:
                        continue

                    found[pid_text] = dial_ma
                    _cache_set_multiaddr(pid_text, dial_ma)

                await anyio.sleep(0.1)
        finally:
            try:
                try:
                    mdns.listener.stop()
                except Exception:
                    pass
                mdns.stop()
            except Exception:
                pass

    try:
        await host.close()
    except Exception:
        pass

    return [RemoteQueue(peer_id=pid, multiaddr=ma) for (pid, ma) in found.items()]


async def discover_peers_via_rendezvous(
    *,
    timeout_s: float = 10.0,
    limit: int = 10,
    exclude_self: bool = True,
    namespace: str = "",
) -> list[RemoteQueue]:
    """Discover peers via libp2p rendezvous.

    This requires at least one reachable rendezvous service in the network.
    When unavailable, this returns an empty list.
    """

    if not _have_libp2p():
        raise RuntimeError("libp2p is not installed")

    import anyio
    import inspect

    from ipfs_accelerate_py.github_cli.libp2p_compat import ensure_libp2p_compatible
    from libp2p import new_host
    from libp2p.tools.async_service import background_trio_service
    from multiaddr import Multiaddr

    if not ensure_libp2p_compatible():
        raise RuntimeError(
            "libp2p is installed but dependency compatibility patches could not be applied. "
            "This environment likely has an incompatible `multihash` module."
        )

    ns = (namespace or "").strip() or _env_str(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS",
        compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS",
        default="ipfs-accelerate-task-queue",
    )

    candidates = [
        ("libp2p.discovery.rendezvous.rendezvous", "RendezvousClient"),
        ("libp2p.discovery.rendezvous", "RendezvousClient"),
        ("libp2p.rendezvous", "RendezvousClient"),
    ]
    cli = None
    for module_name, symbol in candidates:
        try:
            mod = __import__(module_name, fromlist=[symbol])
            cls = getattr(mod, symbol)
            cli = cls
            break
        except Exception:
            continue
    if cli is None:
        return []

    limit = max(1, int(limit))
    exclude_peer_id = _read_announce_peer_id_hint() if bool(exclude_self) else ""

    host_obj = new_host()
    host = await host_obj if inspect.isawaitable(host_obj) else host_obj

    found: dict[str, str] = {}

    async with background_trio_service(host.get_network()):
        await host.get_network().listen(Multiaddr(f"/ip4/{_client_listen_host()}/tcp/0"))

        deadline = anyio.current_time() + max(0.1, float(timeout_s))
        cookie: bytes = b""

        while anyio.current_time() < deadline and len(found) < limit:
            try:
                inst = cli(host)
                discover = getattr(inst, "discover", None)
                if not callable(discover):
                    break
                try:
                    remaining = max(0.1, float(deadline - anyio.current_time()))
                    with anyio.fail_after(remaining):
                        peers, cookie = await discover(ns, limit=100, cookie=cookie)
                except TypeError:
                    remaining = max(0.1, float(deadline - anyio.current_time()))
                    with anyio.fail_after(remaining):
                        peers = list(discover(ns) or [])
                except Exception:
                    peers = []

                for peer_info in list(peers or []):
                    try:
                        pid = getattr(peer_info, "peer_id", None)
                        pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid or "")
                        pid_text = str(pid_text or "").strip()
                        if not pid_text or pid_text in found:
                            continue
                        if exclude_peer_id and pid_text == exclude_peer_id:
                            continue

                        dial_ma = ""
                        for a_text in _best_effort_peerinfo_multiaddrs(peer_info):
                            if a_text.startswith("/ip4/0.0.0.0/") or a_text.startswith("/ip6/::/"):
                                continue
                            dial_ma = str(a_text).strip()
                            if dial_ma:
                                break
                        if not dial_ma:
                            continue

                        found[pid_text] = dial_ma
                        _cache_set_multiaddr(pid_text, dial_ma)
                    except Exception:
                        continue
            except Exception:
                break

            await anyio.sleep(0.1)

    try:
        await host.close()
    except Exception:
        pass

    return [RemoteQueue(peer_id=pid, multiaddr=ma) for (pid, ma) in found.items()]


def discover_peers_via_rendezvous_sync(
    *,
    timeout_s: float = 10.0,
    limit: int = 10,
    exclude_self: bool = True,
    namespace: str = "",
) -> list[RemoteQueue]:
    import trio

    result: list[RemoteQueue] = []

    async def _main() -> None:
        nonlocal result
        result = await discover_peers_via_rendezvous(
            timeout_s=float(timeout_s),
            limit=int(limit),
            exclude_self=bool(exclude_self),
            namespace=str(namespace or ""),
        )

    trio.run(_main)
    return list(result or [])


async def discover_peers_via_dht(
    *,
    timeout_s: float = 10.0,
    limit: int = 10,
    exclude_self: bool = True,
    namespace: str = "",
) -> list[RemoteQueue]:
    """Discover peers via KadDHT provider records.

    Peers that run the TaskQueue P2P service publish provider records for the
    rendezvous namespace key (see `service.py`). This helper queries the DHT for
    providers and returns dialable multiaddrs.
    """

    if not _have_libp2p():
        raise RuntimeError("libp2p is not installed")

    import anyio
    import inspect

    from ipfs_accelerate_py.github_cli.libp2p_compat import ensure_libp2p_compatible
    from libp2p import new_host
    from libp2p.tools.async_service import background_trio_service
    from multiaddr import Multiaddr

    if not ensure_libp2p_compatible():
        raise RuntimeError(
            "libp2p is installed but dependency compatibility patches could not be applied. "
            "This environment likely has an incompatible `multihash` module."
        )

    ns = (namespace or "").strip() or _env_str(
        primary="IPFS_ACCELERATE_PY_TASK_P2P_RENDEZVOUS_NS",
        compat="IPFS_DATASETS_PY_TASK_P2P_RENDEZVOUS_NS",
        default="ipfs-accelerate-task-queue",
    )
    ns_key = _dht_key_for_namespace(ns)
    ns_key_bytes = ns_key.encode("utf-8") if ns_key else b""

    exclude_peer_id = _read_announce_peer_id_hint() if bool(exclude_self) else ""
    limit = max(1, int(limit))

    candidates = [
        ("libp2p.kad_dht.kad_dht", "KadDHT"),
        ("libp2p.kad_dht", "KadDHT"),
    ]

    host_obj = new_host()
    host = await host_obj if inspect.isawaitable(host_obj) else host_obj

    found: dict[str, str] = {}

    async with background_trio_service(host.get_network()):
        await host.get_network().listen(Multiaddr(f"/ip4/{_client_listen_host()}/tcp/0"))

        deadline = anyio.current_time() + max(0.1, float(timeout_s))

        for module_name, symbol in candidates:
            if anyio.current_time() >= deadline or len(found) >= limit:
                break
            try:
                mod = __import__(module_name, fromlist=[symbol])
                cls = getattr(mod, symbol)

                try:
                    from libp2p.kad_dht.kad_dht import DHTMode  # type: ignore

                    dht = cls(host, DHTMode.CLIENT)
                except Exception:
                    dht = cls(host)

                from libp2p.tools.async_service.trio_service import background_trio_service as background_trio_service2

                async with background_trio_service2(dht):
                    try:
                        remaining = max(0.1, float(deadline - anyio.current_time()))
                        with anyio.fail_after(remaining):
                            await _best_effort_connect_multiaddrs(host=host, addrs=_parse_bootstrap_peers())
                    except Exception:
                        pass

                    for method_name in ("start", "bootstrap"):
                        try:
                            fn = getattr(dht, method_name, None)
                            if not callable(fn):
                                continue
                            maybe = fn()
                            if hasattr(maybe, "__await__"):
                                await maybe
                        except Exception:
                            continue

                    find_providers = getattr(dht, "find_providers", None)
                    if not callable(find_providers):
                        continue

                    providers = None
                    for key_candidate in (ns_key, ns_key_bytes, ns):
                        try:
                            remaining = max(0.1, float(deadline - anyio.current_time()))
                            with anyio.fail_after(remaining):
                                providers = await find_providers(key_candidate, 20)
                            break
                        except Exception:
                            continue

                    for peer_info in list(providers or []):
                        if len(found) >= limit:
                            break
                        try:
                            pid = getattr(peer_info, "peer_id", None)
                            pid_text = pid.pretty() if hasattr(pid, "pretty") else str(pid or "")
                            pid_text = str(pid_text or "").strip()
                            if not pid_text or pid_text in found:
                                continue
                            if exclude_peer_id and pid_text == exclude_peer_id:
                                continue

                            dial_ma = ""
                            for a_text in _best_effort_peerinfo_multiaddrs(peer_info):
                                if a_text.startswith("/ip4/0.0.0.0/") or a_text.startswith("/ip6/::/"):
                                    continue
                                dial_ma = str(a_text).strip()
                                if dial_ma:
                                    break
                            if not dial_ma:
                                continue

                            found[pid_text] = dial_ma
                            _cache_set_multiaddr(pid_text, dial_ma)
                        except Exception:
                            continue
            except Exception:
                continue

            await anyio.sleep(0.05)

    try:
        await host.close()
    except Exception:
        pass

    return [RemoteQueue(peer_id=pid, multiaddr=ma) for (pid, ma) in found.items()]


def discover_peers_via_dht_sync(
    *,
    timeout_s: float = 10.0,
    limit: int = 10,
    exclude_self: bool = True,
    namespace: str = "",
) -> list[RemoteQueue]:
    import trio

    result: list[RemoteQueue] = []

    async def _main() -> None:
        nonlocal result
        result = await discover_peers_via_dht(
            timeout_s=float(timeout_s),
            limit=int(limit),
            exclude_self=bool(exclude_self),
            namespace=str(namespace or ""),
        )

    trio.run(_main)
    return list(result or [])


def discover_peers_via_mdns_sync(*, timeout_s: float = 10.0, limit: int = 10, exclude_self: bool = True) -> list[RemoteQueue]:
    import trio

    result: list[RemoteQueue] = []

    async def _main() -> None:
        nonlocal result
        result = await discover_peers_via_mdns(timeout_s=float(timeout_s), limit=int(limit), exclude_self=bool(exclude_self))

    trio.run(_main)
    return list(result or [])


def discover_status_sync(*, remote: RemoteQueue, timeout_s: float = 10.0, detail: bool = False) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await discover_status(remote=remote, timeout_s=timeout_s, detail=detail)

    trio.run(_main)
    return result


async def submit_task(*, remote: RemoteQueue, task_type: str, model_name: str, payload: Dict[str, Any]) -> str:
    def _should_retry_submit_response(resp: Dict[str, Any]) -> bool:
        if not isinstance(resp, dict):
            return False
        if bool(resp.get("ok")):
            return False
        text = str(resp.get("error") or resp).strip().lower()
        if not text:
            return False
        transient_markers = (
            "discovery_timeout",
            "p2p request failed: no response",
            "unable to connect",
            "failed to upgrade security",
            "failed to negotiate the secure protocol",
            "handshake",
            "timeout",
            "stream",
            "swarmexception",
        )
        return any(marker in text for marker in transient_markers)

    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={"op": "submit", "task_type": task_type, "model_name": model_name, "payload": payload},
        retries=_submit_retry_attempts(),
        retry_base_ms=_submit_retry_base_ms(),
        dial_timeout_s=_submit_dial_timeout_s(),
        op_label="submit",
        should_retry_response=_should_retry_submit_response,
    )
    if not resp.get("ok"):
        raise RuntimeError(f"submit failed: {resp}")
    return str(resp.get("task_id"))


def _maybe_str_dict(value: Any) -> Dict[str, str]:
    if not isinstance(value, dict):
        return {}
    out: Dict[str, str] = {}
    for k, v in value.items():
        try:
            out[str(k)] = str(v)
        except Exception:
            continue
    return out


def _maybe_str_dict2(value: Any) -> Dict[str, str]:
    # Alias for readability in docker helpers.
    return _maybe_str_dict(value)


def _normalize_cmd(value: Any) -> Any:
    # Preserve as list[str] if provided, else accept str.
    if value is None:
        return None
    if isinstance(value, list) and all(isinstance(x, str) for x in value):
        return [x for x in value if x]
    if isinstance(value, str):
        return value
    return value


async def submit_docker_hub_task(
    *,
    remote: RemoteQueue,
    image: str,
    command: Any = None,
    entrypoint: Any = None,
    environment: Dict[str, Any] | None = None,
    volumes: Dict[str, Any] | None = None,
    model_name: str = "docker",
    task_type: str = "docker.execute",
    **kwargs: Any,
) -> str:
    """Submit a Docker Hub container run to the TaskQueue.

    This only submits the task; execution depends on workers enabling Docker via
    `IPFS_ACCELERATE_PY_TASK_WORKER_ENABLE_DOCKER=1`.
    """

    payload: Dict[str, Any] = {
        "image": str(image),
    }
    if command is not None:
        payload["command"] = _normalize_cmd(command)
    if entrypoint is not None:
        payload["entrypoint"] = _normalize_cmd(entrypoint)
    if environment is not None:
        payload["environment"] = _maybe_str_dict(environment)
    if volumes is not None:
        payload["volumes"] = _maybe_str_dict2(volumes)

    # Pass through common docker execution settings (memory_limit, cpu_limit, timeout,
    # network_mode, working_dir, read_only, no_new_privileges, user, stream_output, etc.)
    for k, v in kwargs.items():
        if v is None:
            continue
        payload[str(k)] = v

    return await submit_task(remote=remote, task_type=str(task_type), model_name=str(model_name), payload=payload)


def submit_docker_hub_task_sync(
    *,
    remote: RemoteQueue,
    image: str,
    command: Any = None,
    entrypoint: Any = None,
    environment: Dict[str, Any] | None = None,
    volumes: Dict[str, Any] | None = None,
    model_name: str = "docker",
    task_type: str = "docker.execute",
    **kwargs: Any,
) -> str:
    import anyio

    async def _do() -> str:
        return await submit_docker_hub_task(
            remote=remote,
            image=image,
            command=command,
            entrypoint=entrypoint,
            environment=environment,
            volumes=volumes,
            model_name=model_name,
            task_type=task_type,
            **kwargs,
        )

    return anyio.run(_do, backend="trio")


async def submit_docker_github_task(
    *,
    remote: RemoteQueue,
    repo_url: str,
    branch: str = "main",
    dockerfile_path: str = "Dockerfile",
    context_path: str = ".",
    command: Any = None,
    entrypoint: Any = None,
    environment: Dict[str, Any] | None = None,
    build_args: Dict[str, Any] | None = None,
    model_name: str = "docker",
    task_type: str = "docker.github",
    **kwargs: Any,
) -> str:
    """Submit a GitHub repo build+run (Dockerfile) to the TaskQueue."""

    payload: Dict[str, Any] = {
        "repo_url": str(repo_url),
        "branch": str(branch),
        "dockerfile_path": str(dockerfile_path),
        "context_path": str(context_path),
    }
    if command is not None:
        payload["command"] = _normalize_cmd(command)
    if entrypoint is not None:
        payload["entrypoint"] = _normalize_cmd(entrypoint)
    if environment is not None:
        payload["environment"] = _maybe_str_dict(environment)
    if build_args is not None:
        payload["build_args"] = _maybe_str_dict(build_args)

    for k, v in kwargs.items():
        if v is None:
            continue
        payload[str(k)] = v

    return await submit_task(remote=remote, task_type=str(task_type), model_name=str(model_name), payload=payload)


def submit_docker_github_task_sync(
    *,
    remote: RemoteQueue,
    repo_url: str,
    branch: str = "main",
    dockerfile_path: str = "Dockerfile",
    context_path: str = ".",
    command: Any = None,
    entrypoint: Any = None,
    environment: Dict[str, Any] | None = None,
    build_args: Dict[str, Any] | None = None,
    model_name: str = "docker",
    task_type: str = "docker.github",
    **kwargs: Any,
) -> str:
    import anyio

    async def _do() -> str:
        return await submit_docker_github_task(
            remote=remote,
            repo_url=repo_url,
            branch=branch,
            dockerfile_path=dockerfile_path,
            context_path=context_path,
            command=command,
            entrypoint=entrypoint,
            environment=environment,
            build_args=build_args,
            model_name=model_name,
            task_type=task_type,
            **kwargs,
        )

    return anyio.run(_do, backend="trio")


def submit_task_sync(*, remote: RemoteQueue, task_type: str, model_name: str, payload: Dict[str, Any]) -> str:
    import anyio

    async def _do() -> str:
        return await submit_task(remote=remote, task_type=task_type, model_name=model_name, payload=payload)

    return anyio.run(_do, backend="trio")


async def submit_task_with_info(
    *,
    remote: RemoteQueue,
    task_type: str,
    model_name: str,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={"op": "submit", "task_type": task_type, "model_name": model_name, "payload": payload},
        retries=_submit_retry_attempts(),
        retry_base_ms=_submit_retry_base_ms(),
        dial_timeout_s=_submit_dial_timeout_s(),
        op_label="submit_with_info",
    )
    if not resp.get("ok"):
        raise RuntimeError(f"submit failed: {resp}")
    return {"task_id": str(resp.get("task_id")), "peer_id": str(resp.get("peer_id") or "").strip()}


def submit_task_with_info_sync(
    *,
    remote: RemoteQueue,
    task_type: str,
    model_name: str,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    import anyio

    async def _do() -> Dict[str, str]:
        return await submit_task_with_info(remote=remote, task_type=task_type, model_name=model_name, payload=payload)

    return anyio.run(_do, backend="trio")


async def claim_next(
    *,
    remote: RemoteQueue,
    worker_id: str,
    supported_task_types: list[str] | None = None,
    session_id: str | None = None,
    peer_id: str | None = None,
    clock: Dict[str, Any] | None = None,
) -> Optional[Dict[str, Any]]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={
            "op": "claim",
            "worker_id": str(worker_id),
            "supported_task_types": list(supported_task_types or []),
            "session_id": str(session_id or "").strip(),
            "peer_id": str(peer_id) if peer_id else "",
            "clock": clock,
        },
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=_rpc_dial_timeout_s(),
        op_label="claim",
    )
    if not resp.get("ok"):
        raise RuntimeError(f"claim failed: {resp}")
    task = resp.get("task")
    return task if isinstance(task, dict) else None


async def claim_many(
    *,
    remote: RemoteQueue,
    worker_id: str,
    supported_task_types: list[str] | None = None,
    max_tasks: int = 1,
    same_task_type: bool = True,
    session_id: str | None = None,
    peer_id: str | None = None,
    clock: Dict[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={
            "op": "claim_many",
            "worker_id": str(worker_id),
            "supported_task_types": list(supported_task_types or []),
            "max_tasks": int(max_tasks),
            "same_task_type": bool(same_task_type),
            "session_id": str(session_id or "").strip(),
            "peer_id": str(peer_id) if peer_id else "",
            "clock": clock,
        },
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=_rpc_dial_timeout_s(),
        op_label="claim_many",
    )
    if not resp.get("ok"):
        raise RuntimeError(f"claim_many failed: {resp}")
    tasks = resp.get("tasks")
    return list(tasks) if isinstance(tasks, list) else []


def claim_next_sync(
    *,
    remote: RemoteQueue,
    worker_id: str,
    supported_task_types: list[str] | None = None,
    session_id: str | None = None,
    peer_id: str | None = None,
    clock: Dict[str, Any] | None = None,
) -> Optional[Dict[str, Any]]:
    import anyio

    async def _do() -> Optional[Dict[str, Any]]:
        return await claim_next(
            remote=remote,
            worker_id=worker_id,
            supported_task_types=supported_task_types,
            session_id=session_id,
            peer_id=peer_id,
            clock=clock,
        )

    return anyio.run(_do, backend="trio")


def claim_many_sync(
    *,
    remote: RemoteQueue,
    worker_id: str,
    supported_task_types: list[str] | None = None,
    max_tasks: int = 1,
    same_task_type: bool = True,
    session_id: str | None = None,
    peer_id: str | None = None,
    clock: Dict[str, Any] | None = None,
) -> list[Dict[str, Any]]:
    import anyio

    async def _do() -> list[Dict[str, Any]]:
        return await claim_many(
            remote=remote,
            worker_id=worker_id,
            supported_task_types=supported_task_types,
            max_tasks=max_tasks,
            same_task_type=same_task_type,
            session_id=session_id,
            peer_id=peer_id,
            clock=clock,
        )

    return anyio.run(_do, backend="trio")


async def heartbeat(*, remote: RemoteQueue, peer_id: str, clock: Dict[str, Any] | None = None) -> Dict[str, Any]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={"op": "peer.heartbeat", "peer_id": str(peer_id), "clock": clock},
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=_rpc_dial_timeout_s(),
        op_label="heartbeat",
    )
    if not resp.get("ok"):
        raise RuntimeError(f"heartbeat failed: {resp}")
    return resp


def heartbeat_sync(*, remote: RemoteQueue, peer_id: str, clock: Dict[str, Any] | None = None) -> Dict[str, Any]:
    import anyio

    async def _do() -> Dict[str, Any]:
        return await heartbeat(remote=remote, peer_id=peer_id, clock=clock)

    return anyio.run(_do, backend="trio")


async def list_tasks(
    *,
    remote: RemoteQueue,
    status: str | None = None,
    limit: int = 50,
    task_types: list[str] | None = None,
) -> Dict[str, Any]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={"op": "list", "status": status, "limit": int(limit), "task_types": list(task_types or [])},
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=_rpc_dial_timeout_s(),
        op_label="list",
    )
    if not resp.get("ok"):
        raise RuntimeError(f"list failed: {resp}")
    return resp


def list_tasks_sync(
    *,
    remote: RemoteQueue,
    status: str | None = None,
    limit: int = 50,
    task_types: list[str] | None = None,
) -> Dict[str, Any]:
    import anyio

    async def _do() -> Dict[str, Any]:
        return await list_tasks(remote=remote, status=status, limit=limit, task_types=task_types)

    return anyio.run(_do, backend="trio")


async def complete_task(
    *,
    remote: RemoteQueue,
    task_id: str,
    status: str = "completed",
    result: Dict[str, Any] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={
            "op": "complete",
            "task_id": str(task_id),
            "status": str(status),
            "result": result,
            "error": error,
        },
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=_rpc_dial_timeout_s(),
        op_label="complete",
    )
    if not resp.get("ok"):
        raise RuntimeError(f"complete failed: {resp}")
    return resp


async def release_task(
    *,
    remote: RemoteQueue,
    task_id: str,
    worker_id: str,
    reason: str | None = None,
) -> Dict[str, Any]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={
            "op": "release",
            "task_id": str(task_id),
            "worker_id": str(worker_id),
            "reason": str(reason) if reason else "",
        },
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=_rpc_dial_timeout_s(),
        op_label="release",
    )
    if not resp.get("ok"):
        raise RuntimeError(f"release failed: {resp}")
    return resp


def release_task_sync(
    *,
    remote: RemoteQueue,
    task_id: str,
    worker_id: str,
    reason: str | None = None,
) -> Dict[str, Any]:
    import anyio

    async def _do() -> Dict[str, Any]:
        return await release_task(remote=remote, task_id=task_id, worker_id=worker_id, reason=reason)

    return anyio.run(_do, backend="trio")


def complete_task_sync(
    *,
    remote: RemoteQueue,
    task_id: str,
    status: str = "completed",
    result: Dict[str, Any] | None = None,
    error: str | None = None,
) -> Dict[str, Any]:
    import anyio

    async def _do() -> Dict[str, Any]:
        return await complete_task(remote=remote, task_id=task_id, status=status, result=result, error=error)

    return anyio.run(_do, backend="trio")


async def get_task(*, remote: RemoteQueue, task_id: str) -> Optional[Dict[str, Any]]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={"op": "get", "task_id": task_id},
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=_rpc_dial_timeout_s(),
        op_label="get",
    )
    if not resp.get("ok"):
        raise RuntimeError(f"get failed: {resp}")
    task = resp.get("task")
    return task if isinstance(task, dict) else None


async def wait_task(*, remote: RemoteQueue, task_id: str, timeout_s: float = 60.0) -> Optional[Dict[str, Any]]:
    import anyio

    retries = _wait_retry_attempts()
    base_ms = _wait_retry_base_ms()

    # `wait` is a long-poll RPC: the peer may keep the stream open until the
    # task completes or until its own timeout fires.
    dial_timeout_s = max(20.0, float(timeout_s) + 15.0)

    for attempt in range(retries + 1):
        attempt_dial_timeout_s = _dial_timeout_for_attempt(base_timeout_s=float(dial_timeout_s), attempt=attempt)
        cooldown_wait_s = _remote_cooldown_wait_s(remote)
        if cooldown_wait_s > 0:
            _retry_metric_inc("wait.cooldown_wait")
            _dial_debug(f"wait cooldown wait before attempt={attempt + 1}/{retries + 1} wait_s={cooldown_wait_s:.3f}")
            await anyio.sleep(cooldown_wait_s)
        try:
            release_slot = await _acquire_dial_slot(op_label="wait")
            try:
                resp = await _dial_and_request(
                    remote=remote,
                    message={"op": "wait", "task_id": task_id, "timeout_s": float(timeout_s)},
                    dial_timeout_s=attempt_dial_timeout_s,
                )
            finally:
                release_slot()
            _remote_cooldown_mark_success(remote)
            if not resp.get("ok"):
                raise RuntimeError(f"wait failed: {resp}")
            task = resp.get("task")
            if attempt > 0:
                _retry_metric_inc("wait.recovered")
            return task if isinstance(task, dict) else None
        except TimeoutError as exc:
            # Distinguish queueing contention from normal long-poll timeouts:
            # dial-slot timeout should retry, while wait long-poll timeout
            # still maps to "task not ready yet" -> None.
            msg = str(exc or "").lower()
            if "dial slot timeout" in msg:
                if attempt >= retries:
                    _retry_metric_inc("wait.failed")
                    raise
                delay_s = _retry_delay_s(attempt=attempt, base_ms=base_ms)
                _retry_metric_inc("wait.retry")
                _dial_debug(
                    f"wait retry after dial slot timeout attempt={attempt + 1}/{retries + 1} delay_s={delay_s:.3f}"
                )
                await anyio.sleep(delay_s)
                continue
            return None
        except BaseExceptionGroup as eg:
            # trio/anyio may wrap cancellations/timeouts in an ExceptionGroup.
            if _exception_group_contains_dial_slot_timeout(eg):
                if attempt >= retries:
                    _retry_metric_inc("wait.failed")
                    raise
                delay_s = _retry_delay_s(attempt=attempt, base_ms=base_ms)
                _retry_metric_inc("wait.retry")
                _dial_debug(
                    f"wait retry after grouped dial slot timeout attempt={attempt + 1}/{retries + 1} delay_s={delay_s:.3f}"
                )
                await anyio.sleep(delay_s)
                continue
            if _exception_group_contains_timeout(eg):
                return None
            _remote_cooldown_mark_failure(remote)
            if attempt >= retries:
                _retry_metric_inc("wait.failed")
                raise
            delay_s = _retry_delay_s(attempt=attempt, base_ms=base_ms)
            _retry_metric_inc("wait.retry")
            _dial_debug(f"wait retry after BaseExceptionGroup attempt={attempt + 1}/{retries + 1} delay_s={delay_s:.3f}")
            await anyio.sleep(delay_s)
        except Exception as exc:
            retryable = _is_retryable_transport_error(exc)
            if retryable:
                _remote_cooldown_mark_failure(remote)
            if attempt >= retries or not retryable:
                if attempt >= retries:
                    _retry_metric_inc("wait.failed")
                raise
            delay_s = _retry_delay_s(attempt=attempt, base_ms=base_ms)
            _retry_metric_inc("wait.retry")
            _dial_debug(f"wait retry after {type(exc).__name__} attempt={attempt + 1}/{retries + 1} delay_s={delay_s:.3f}")
            await anyio.sleep(delay_s)
    return None


async def get_capabilities(*, remote: RemoteQueue, timeout_s: float = 10.0, detail: bool = False) -> Dict[str, Any]:
    resp = await request_status(remote=remote, timeout_s=timeout_s, detail=detail)
    if not resp.get("ok"):
        raise RuntimeError(f"status failed: {resp}")
    caps = resp.get("capabilities")
    return caps if isinstance(caps, dict) else {}


async def request_status(*, remote: RemoteQueue, timeout_s: float = 10.0, detail: bool = False) -> Dict[str, Any]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={"op": "status", "timeout_s": float(timeout_s), "detail": bool(detail)},
        retries=_status_retry_attempts(),
        retry_base_ms=_status_retry_base_ms(),
        dial_timeout_s=_status_dial_timeout_s(),
        op_label="status",
    )
    return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}

async def cancel_task(*, remote: RemoteQueue, task_id: str, reason: str | None = None) -> Dict[str, Any]:
    message: Dict[str, Any] = {"op": "cancel", "task_id": str(task_id)}
    if isinstance(reason, str) and reason.strip():
        message["reason"] = reason.strip()
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message=message,
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=_rpc_dial_timeout_s(),
        op_label="cancel",
    )
    return resp if isinstance(resp, dict) else {"ok": False, "error": "invalid_response"}


def cancel_task_sync(*, remote: RemoteQueue, task_id: str, reason: str | None = None) -> Dict[str, Any]:
    import anyio

    async def _do() -> Dict[str, Any]:
        return await cancel_task(remote=remote, task_id=task_id, reason=reason)

    return anyio.run(_do, backend="trio")


def request_status_sync(*, remote: RemoteQueue, timeout_s: float = 10.0, detail: bool = False) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await request_status(remote=remote, timeout_s=timeout_s, detail=detail)

    trio.run(_main)
    return result


def get_capabilities_sync(*, remote: RemoteQueue, timeout_s: float = 10.0, detail: bool = False) -> Dict[str, Any]:
    """Synchronous wrapper around `get_capabilities`.

    Note: libp2p uses Trio internally; this wrapper runs a Trio event loop.
    """

    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await get_capabilities(remote=remote, timeout_s=timeout_s, detail=detail)

    trio.run(_main)
    return result


async def call_tool(
    *,
    remote: RemoteQueue,
    tool_name: str,
    args: Dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={
            "op": "call_tool",
            "tool_name": str(tool_name),
            "args": (args if isinstance(args, dict) else {}),
            "timeout_s": float(timeout_s),
        },
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=max(_rpc_dial_timeout_s(), float(timeout_s) + 5.0),
        op_label="call_tool",
    )
    if not isinstance(resp, dict):
        return {"ok": False, "tool": str(tool_name), "error": "invalid_response"}
    return resp


def call_tool_sync(
    *,
    remote: RemoteQueue,
    tool_name: str,
    args: Dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await call_tool(remote=remote, tool_name=tool_name, args=args, timeout_s=timeout_s)

    trio.run(_main)
    return result


async def cache_get(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={"op": "cache.get", "key": str(key), "timeout_s": float(timeout_s)},
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=max(_rpc_dial_timeout_s(), float(timeout_s) + 2.0),
        op_label="cache_get",
    )
    if not isinstance(resp, dict):
        return {"ok": False, "error": "invalid_response"}
    return resp


def cache_get_sync(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await cache_get(remote=remote, key=str(key), timeout_s=float(timeout_s))

    trio.run(_main)
    return result


async def cache_has(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={"op": "cache.has", "key": str(key), "timeout_s": float(timeout_s)},
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=max(_rpc_dial_timeout_s(), float(timeout_s) + 2.0),
        op_label="cache_has",
    )
    if not isinstance(resp, dict):
        return {"ok": False, "error": "invalid_response"}
    return resp


def cache_has_sync(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await cache_has(remote=remote, key=str(key), timeout_s=float(timeout_s))

    trio.run(_main)
    return result


async def cache_set(
    *,
    remote: RemoteQueue,
    key: str,
    value: Any,
    ttl_s: float | None = None,
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    message: Dict[str, Any] = {
        "op": "cache.set",
        "key": str(key),
        "value": value,
        "timeout_s": float(timeout_s),
    }
    if ttl_s is not None:
        try:
            message["ttl_s"] = float(ttl_s)
        except Exception:
            pass

    resp = await _dial_and_request_with_retries(
        remote=remote,
        message=message,
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=max(_rpc_dial_timeout_s(), float(timeout_s) + 2.0),
        op_label="cache_set",
    )
    if not isinstance(resp, dict):
        return {"ok": False, "error": "invalid_response"}
    return resp


def cache_set_sync(
    *,
    remote: RemoteQueue,
    key: str,
    value: Any,
    ttl_s: float | None = None,
    timeout_s: float = 10.0,
) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await cache_set(
            remote=remote,
            key=str(key),
            value=value,
            ttl_s=ttl_s,
            timeout_s=float(timeout_s),
        )

    trio.run(_main)
    return result


async def cache_delete(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    resp = await _dial_and_request_with_retries(
        remote=remote,
        message={"op": "cache.delete", "key": str(key), "timeout_s": float(timeout_s)},
        retries=_rpc_retry_attempts(),
        retry_base_ms=_rpc_retry_base_ms(),
        dial_timeout_s=max(_rpc_dial_timeout_s(), float(timeout_s) + 2.0),
        op_label="cache_delete",
    )
    if not isinstance(resp, dict):
        return {"ok": False, "error": "invalid_response"}
    return resp


def cache_delete_sync(*, remote: RemoteQueue, key: str, timeout_s: float = 10.0) -> Dict[str, Any]:
    import trio

    result: Dict[str, Any] = {}

    async def _main() -> None:
        nonlocal result
        result = await cache_delete(remote=remote, key=str(key), timeout_s=float(timeout_s))

    trio.run(_main)
    return result
