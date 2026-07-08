"""Kubo-compatible CIDv1 helpers for cross-server integration tests."""

from __future__ import annotations

import base64
import hashlib


def cid_for_bytes(data: bytes) -> str:
    """Return CIDv1 raw/sha2-256/base32 for bytes."""

    digest = hashlib.sha256(bytes(data)).digest()
    cid_bytes = bytes([0x01, 0x55, 0x12, 0x20]) + digest
    return "b" + base64.b32encode(cid_bytes).decode("ascii").rstrip("=").lower()
