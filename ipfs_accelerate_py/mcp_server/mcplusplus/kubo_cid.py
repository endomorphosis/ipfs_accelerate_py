"""Kubo-conformant CIDv1 content addressing for MCP++ artifacts.

Canonical IPFS profile: CIDv1, raw codec (0x55), sha2-256 (0x12), base32 lower
multibase ('b') -> bafkrei…. Dependency-free; byte-identical to multiformats
``cid_for_obj(codec="raw", base="base32")`` and ipfs_datasets_py cid_utils.
"""

from __future__ import annotations

import base64
import hashlib


def base32_lower_nopad(data: bytes) -> str:
    return base64.b32encode(data).decode("ascii").rstrip("=").lower()


def cid_for_bytes(body: bytes) -> str:
    """Return a Kubo CIDv1 (raw/sha2-256/base32) for raw content bytes."""
    digest = hashlib.sha256(body).digest()
    cid_bytes = bytes([0x01, 0x55, 0x12, 0x20]) + digest
    return "b" + base32_lower_nopad(cid_bytes)
