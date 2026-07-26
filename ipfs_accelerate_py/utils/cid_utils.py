"""Deterministic CID helpers used by router response caches."""

from __future__ import annotations

import json
from typing import Any


def canonical_json_bytes(value: Any) -> bytes:
    """Serialize a value into stable JSON bytes."""

    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    ).encode("utf-8")


def cid_for_bytes(
    data: bytes,
    *,
    base: str = "base32",
    codec: str = "raw",
    mh_type: str = "sha2-256",
    version: int = 1,
) -> str:
    """Compute a CID for bytes using the multiformats package."""

    from multiformats import CID, multihash

    digest = multihash.digest(data, mh_type)
    return str(CID(base, version, codec, digest))


def cid_for_obj(
    value: Any,
    *,
    base: str = "base32",
    codec: str = "raw",
    mh_type: str = "sha2-256",
    version: int = 1,
) -> str:
    """Compute a CID for a deterministically serialized value."""

    return cid_for_bytes(
        canonical_json_bytes(value),
        base=base,
        codec=codec,
        mh_type=mh_type,
        version=version,
    )
