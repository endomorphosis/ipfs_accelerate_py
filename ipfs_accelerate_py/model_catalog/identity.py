"""Deterministic, dependency-free identities for AI catalog records.

Only canonical, non-secret JSON is accepted.  The helpers in this module are
pure: importing or calling them never performs I/O, provider discovery, or
credential lookup.
"""

from __future__ import annotations

import base64
import dataclasses
import hashlib
import json
import math
import re
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Mapping

MAX_CANONICAL_BYTES = 1_048_576
MAX_CANONICAL_DEPTH = 24
MAX_CONTAINER_ITEMS = 4096
MAX_STRING_BYTES = 65_536
MAX_ABS_INTEGER = (1 << 63) - 1
IDENTITY_VERSION = "ai.catalog.identity.v1"

_SECRET_KEY = re.compile(
    r"(?:^|[_-])(?:api[_-]?key|access[_-]?key|secret|password|passwd|token|"
    r"credential|private[_-]?key|auth|auth[_-]?header|authorization|bearer)(?:$|[_-])",
    re.IGNORECASE,
)
_SECRET_VALUE = re.compile(
    r"(?:bearer\s+\S{12,}|sk-[A-Za-z0-9_-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|"
    r"hf_[A-Za-z0-9]{24,}|xox[baprs]-[A-Za-z0-9-]{20,}|"
    r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----)",
    re.IGNORECASE,
)
_ID_KIND = re.compile(r"^[a-z][a-z0-9.-]{0,31}$")
REDACTED = "[REDACTED]"


class CanonicalizationError(ValueError):
    """Raised when a value cannot safely be represented as canonical JSON."""


def is_secret_key(key: str) -> bool:
    """Return whether *key* conventionally contains credential material."""

    return bool(_SECRET_KEY.search(str(key)))


def is_secret_value(value: str) -> bool:
    """Conservatively recognize common credential value formats."""

    return bool(_SECRET_VALUE.search(str(value).strip()))


def redact_secrets(value: Any) -> Any:
    """Return a recursively redacted, JSON-compatible copy of *value*.

    Secret-shaped mapping values and recognizable credential strings are
    replaced.  Input objects are never mutated.
    """

    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = value.to_dict() if hasattr(value, "to_dict") else dataclasses.asdict(value)
    if isinstance(value, Mapping):
        return {
            str(key): REDACTED if is_secret_key(str(key)) else redact_secrets(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [redact_secrets(item) for item in value]
    if isinstance(value, (set, frozenset)):
        redacted = [redact_secrets(item) for item in value]
        return sorted(redacted, key=lambda item: _json_bytes(item))
    if isinstance(value, str) and is_secret_value(value):
        return REDACTED
    if isinstance(value, Enum):
        return redact_secrets(value.value)
    return value


def _json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _canonical_value(value: Any, depth: int, reject_secrets: bool) -> Any:
    if depth > MAX_CANONICAL_DEPTH:
        raise CanonicalizationError("canonical value exceeds maximum nesting depth")
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        value = value.to_dict() if hasattr(value, "to_dict") else dataclasses.asdict(value)
    if isinstance(value, Enum):
        value = value.value
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, int):
        if abs(value) > MAX_ABS_INTEGER:
            raise CanonicalizationError("integer exceeds canonical 64-bit bound")
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise CanonicalizationError("non-finite floats are not canonical JSON")
        return value
    if isinstance(value, datetime):
        if value.tzinfo is None or value.utcoffset() is None:
            raise CanonicalizationError("timestamps must be timezone-aware")
        return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    if isinstance(value, str):
        if len(value.encode("utf-8")) > MAX_STRING_BYTES:
            raise CanonicalizationError("string exceeds canonical size bound")
        if reject_secrets and is_secret_value(value):
            raise CanonicalizationError("credential-shaped string is not canonical catalog data")
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise CanonicalizationError("mapping exceeds canonical item bound")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("canonical JSON mapping keys must be strings")
            if reject_secrets and is_secret_key(key) and item != REDACTED:
                raise CanonicalizationError("credential-bearing field is forbidden: %s" % key)
            if key in result:
                raise CanonicalizationError("duplicate canonical mapping key: %s" % key)
            result[key] = _canonical_value(item, depth + 1, reject_secrets)
        return result
    if isinstance(value, (set, frozenset)):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise CanonicalizationError("set exceeds canonical item bound")
        items = [_canonical_value(item, depth + 1, reject_secrets) for item in value]
        return sorted(items, key=_json_bytes)
    if isinstance(value, (list, tuple)):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise CanonicalizationError("sequence exceeds canonical item bound")
        return [_canonical_value(item, depth + 1, reject_secrets) for item in value]
    raise CanonicalizationError("unsupported canonical value type: %s" % type(value).__name__)


def canonical_data(value: Any, *, reject_secrets: bool = True) -> Any:
    """Return the bounded JSON-compatible canonical form of *value*."""

    canonical = _canonical_value(value, 0, reject_secrets)
    payload = _json_bytes(canonical)
    if len(payload) > MAX_CANONICAL_BYTES:
        raise CanonicalizationError("canonical document exceeds byte bound")
    return canonical


def canonical_json_bytes(value: Any, *, reject_secrets: bool = True) -> bytes:
    """Serialize *value* to deterministic UTF-8 canonical JSON bytes."""

    return _json_bytes(canonical_data(value, reject_secrets=reject_secrets))


def canonical_json(value: Any, *, reject_secrets: bool = True) -> str:
    """Serialize *value* to deterministic canonical JSON text."""

    return canonical_json_bytes(value, reject_secrets=reject_secrets).decode("utf-8")


def content_cid(value: Any) -> str:
    """Return a CIDv1/raw/sha2-256 for the exact canonical JSON bytes."""

    digest = hashlib.sha256(canonical_json_bytes(value)).digest()
    # CIDv1 + raw codec + sha2-256 multihash.  All involved varints fit a byte.
    binary_cid = b"\x01\x55\x12\x20" + digest
    return "b" + base64.b32encode(binary_cid).decode("ascii").lower().rstrip("=")


def stable_id(kind: str, *components: Any) -> str:
    """Build a collision-resistant stable ID from framed identity components.

    Components remain ordered, while mappings and sets inside a component are
    order independent under canonical JSON rules.  Framing as JSON prevents
    concatenation ambiguities such as ``("ab", "c")`` versus ``("a", "bc")``.
    """

    if not isinstance(kind, str) or not _ID_KIND.fullmatch(kind):
        raise ValueError("identity kind must match %s" % _ID_KIND.pattern)
    material = {
        "identity_version": IDENTITY_VERSION,
        "kind": kind,
        "components": list(components),
    }
    digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    return "%s_%s" % (kind, digest)


def provider_identity(name: str) -> str:
    return stable_id("provider", name)


def model_identity(provider_id: str, name: str) -> str:
    return stable_id("model", provider_id, name)


def deployment_identity(provider_id: str, model_id: str, name: str, endpoint_uri: str) -> str:
    return stable_id("deployment", provider_id, model_id, name, endpoint_uri)


def router_binding_identity(
    router: str, provider_id: str, model_id: str, deployment_id: str
) -> str:
    return stable_id("binding", router, provider_id, model_id, deployment_id)


# Friendly aliases for consumers that use "serialize" or "CID" terminology.
canonical_serialize = canonical_json_bytes
canonical_cid = content_cid


__all__ = [
    "CanonicalizationError",
    "IDENTITY_VERSION",
    "MAX_CANONICAL_BYTES",
    "MAX_CANONICAL_DEPTH",
    "MAX_CONTAINER_ITEMS",
    "MAX_ABS_INTEGER",
    "MAX_STRING_BYTES",
    "REDACTED",
    "canonical_cid",
    "canonical_data",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_serialize",
    "content_cid",
    "deployment_identity",
    "is_secret_key",
    "is_secret_value",
    "model_identity",
    "provider_identity",
    "redact_secrets",
    "router_binding_identity",
    "stable_id",
]
