"""Shared constants/helpers for the libp2p task queue protocol.

This module is the canonical definition of the TaskQueue RPC protocol.

Compatibility:
- Accepts token env vars from both ipfs_datasets_py and ipfs_accelerate_py.
"""

from __future__ import annotations

import os
import base64
import hashlib
from typing import Any, Dict, Optional

from cryptography.fernet import Fernet, InvalidToken


# Keep the protocol id stable for interop with existing nodes.
PROTOCOL_V1 = "/ipfs-datasets/task-queue/1.0.0"


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
