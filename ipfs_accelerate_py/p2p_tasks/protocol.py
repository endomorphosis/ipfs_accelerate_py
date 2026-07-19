"""Shared constants/helpers for the libp2p task queue protocol.

This module is the canonical definition of the TaskQueue RPC protocol.

Compatibility:
- Accepts token env vars from both ipfs_datasets_py and ipfs_accelerate_py.
"""

from __future__ import annotations

import os
import base64
import hashlib
import json
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken


# Keep the protocol id stable for interop with existing nodes.
PROTOCOL_V1 = "/ipfs-datasets/task-queue/1.0.0"

# MCP++ tool name used to carry TaskQueue RPC messages over `/mcp+p2p/1.0.0`.
TASK_QUEUE_MCP_RPC_TOOL = "task_queue.rpc"


def _truthy(text: str | None, *, default: bool = False) -> bool:
    if text is None:
        return bool(default)
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def _protocol_env_value() -> str:
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_PROTOCOL")
    if raw is None:
        raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_PROTOCOL")
    return str(raw or "").strip()


def task_p2p_protocol_order() -> list[str]:
    """Return the TaskQueue transport preference order.

    Defaults to MCP++ only. Operators can set
    `IPFS_ACCELERATE_PY_TASK_P2P_PROTOCOL=auto` for MCP++ then legacy fallback,
    or `legacy` for old NDJSON-only interoperability.
    """

    raw = _protocol_env_value() or "mcp"
    tokens = [
        p.strip().lower()
        for p in raw.replace("|", ",").replace(";", ",").replace(" ", ",").split(",")
        if p.strip()
    ]

    out: list[str] = []

    def _append(value: str) -> None:
        if value not in out:
            out.append(value)

    for token in tokens:
        normalized = token.replace("_", "-")
        if normalized in {"auto", "mcp-then-legacy", "mcp+legacy", "mcp/legacy"}:
            _append("mcp")
            _append("legacy")
        elif normalized in {"mcp", "mcp++", "mcp+p2p", "mcpp", "mcpp2p"}:
            _append("mcp")
        elif normalized in {"legacy", "ndjson", "v1", "task-queue", "taskqueue"}:
            _append("legacy")

    if not out:
        out.append("mcp")

    allow_legacy_fallback = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ALLOW_LEGACY_FALLBACK")
    if allow_legacy_fallback is None:
        allow_legacy_fallback = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_ALLOW_LEGACY_FALLBACK")
    if _truthy(allow_legacy_fallback, default=False):
        _append("legacy")

    return out


def legacy_task_protocol_enabled(*, default: bool = False) -> bool:
    """Return whether the service should register the legacy NDJSON handler."""

    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ENABLE_LEGACY_PROTOCOL")
    if raw is None:
        raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_ENABLE_LEGACY_PROTOCOL")
    if raw is not None:
        return _truthy(raw, default=default)

    # Treat an explicit server/client protocol choice that includes legacy as
    # operator intent to expose the old handler.
    if _protocol_env_value():
        return "legacy" in task_p2p_protocol_order()

    return bool(default)


async def read_ndjson_message(
    stream: Any,
    *,
    max_message_bytes: int = 1024 * 1024,
    chunk_size: int = 1024,
) -> tuple[dict[str, Any] | None, str | None]:
    """Read a single newline-delimited JSON message from a libp2p stream.

    Returns (message, error_code).

    Error codes are stable strings intended to be forwarded to callers.
    """

    raw = bytearray()
    saw_newline = False

    try:
        max_b = int(max_message_bytes)
    except Exception:
        max_b = 1024 * 1024
    max_b = max(1, max_b)

    try:
        chunk_b = int(chunk_size)
    except Exception:
        chunk_b = 1024
    chunk_b = max(1, min(1024 * 1024, chunk_b))

    while len(raw) < max_b:
        chunk = await stream.read(chunk_b)
        if not chunk:
            break
        raw.extend(chunk)
        if b"\n" in chunk:
            saw_newline = True
            break

    if not raw:
        return None, "empty"

    # If we hit the max byte budget without reaching the frame delimiter,
    # treat this as a deterministic abuse/oversize violation.
    if len(raw) >= max_b and not saw_newline:
        return None, "frame_too_large"

    try:
        msg = json.loads(bytes(raw).rstrip(b"\n").decode("utf-8"))
    except Exception:
        return None, "invalid_json"

    if not isinstance(msg, dict):
        return None, "invalid_message"

    # Normalize to a plain dict[str, Any] for downstream callers.
    try:
        return dict(msg), None
    except Exception:
        return None, "invalid_message"


def get_shared_token() -> Optional[str]:
    for name in (
        "IPFS_ACCELERATE_PY_TASK_P2P_TOKEN",
        "IPFS_DATASETS_PY_TASK_P2P_TOKEN",
    ):
        token = os.environ.get(name, "").strip()
        if token:
            return token
    return None


def auth_ok(message: Dict[str, Any]) -> bool:
    expected = get_shared_token()
    if not expected:
        return True
    provided = (message.get("token") or "")
    return isinstance(provided, str) and provided == expected


def encryption_enabled() -> bool:
    """Return True if application-layer encryption is enabled.

    Note: libp2p connections are already encrypted at the transport layer.
    This flag adds an extra layer to avoid transmitting/storing plaintext
    prompts in task payloads.
    """

    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ENCRYPT")
    if raw is None:
        raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_ENCRYPT")
    return str(raw or "").strip().lower() in {"1", "true", "yes", "on"}


def _encryption_secret() -> str:
    # Prefer an explicit secret if provided; otherwise reuse the shared token.
    raw = os.environ.get("IPFS_ACCELERATE_PY_TASK_P2P_ENCRYPTION_KEY")
    if raw is None:
        raw = os.environ.get("IPFS_DATASETS_PY_TASK_P2P_ENCRYPTION_KEY")
    secret = str(raw or "").strip()
    if secret:
        return secret
    token = get_shared_token() or ""
    return str(token).strip()


def _fernet() -> Fernet | None:
    secret = _encryption_secret()
    if not secret:
        return None
    # Deterministic derivation: 32 bytes -> urlsafe base64.
    salt = b"ipfs-accelerate-task-p2p-fernet-v1"
    key_bytes = hashlib.sha256(salt + secret.encode("utf-8")).digest()
    key = base64.urlsafe_b64encode(key_bytes)
    return Fernet(key)


def encrypt_text(value: str) -> Dict[str, str] | None:
    if not encryption_enabled():
        return None
    f = _fernet()
    if f is None:
        return None
    ct = f.encrypt(str(value).encode("utf-8"))
    return {"enc": "fernet-v1", "ct": ct.decode("ascii")}


def decrypt_text(wrapped: Any) -> str | None:
    if not isinstance(wrapped, dict):
        return None
    if wrapped.get("enc") != "fernet-v1":
        return None
    ct = wrapped.get("ct")
    if not isinstance(ct, str) or not ct:
        return None
    f = _fernet()
    if f is None:
        return None
    try:
        pt = f.decrypt(ct.encode("ascii"))
        return pt.decode("utf-8")
    except InvalidToken:
        return None
    except Exception:
        return None
