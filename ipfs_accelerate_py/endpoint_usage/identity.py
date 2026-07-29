"""Deterministic, secret-free identities for endpoint usage contracts.

Importing or calling these helpers never performs I/O, provider discovery,
credential lookup, process spawn, model load, or database access.
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
from typing import Any, Mapping, Optional, Sequence, Tuple
from urllib.parse import urlsplit, urlunsplit

MAX_CANONICAL_BYTES = 1_048_576
MAX_CANONICAL_DEPTH = 24
MAX_CONTAINER_ITEMS = 4096
MAX_STRING_BYTES = 65_536
MAX_ABS_INTEGER = (1 << 63) - 1
MAX_NAME_BYTES = 256
MAX_ENDPOINT_BYTES = 2048
IDENTITY_VERSION = "ai.endpoint_usage.identity.v1"
IDENTITY_POLICY_VERSION = "1.0"

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
_BEARER_URL = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://[^/\s]*@|"
    r"[?&#](?:api[_-]?key|access[_-]?token|token|secret|password|auth)=[^&#\s]+|"
    r"bearer\s+[A-Za-z0-9._\-]{8,})"
)
_RAW_ENDPOINT = re.compile(
    r"(?i)(?:[a-z][a-z0-9+.-]*://|"
    r"(?:^|[.@/])(?:localhost|(?:\d{1,3}\.){3}\d{1,3})(?::\d+)?(?:/|$))"
)
_ID_KIND = re.compile(r"^[a-z][a-z0-9.-]{0,31}$")
_CONFIG_REF = re.compile(
    r"^(?:env|secret_store|config|file|keyring|vault):[A-Za-z0-9._/\-]{1,240}$"
)
_KEY_ID = re.compile(r"^[A-Za-z0-9._\-]{1,128}$")
_CATALOG_ID = re.compile(r"^[a-z]+_[0-9a-f]{64}$")
_PSEUDONYM = re.compile(r"^(?:cred|ep|scope|acct|proj|org)_[0-9a-f]{64}$")

REDACTED = "[REDACTED]"
UNKNOWN_SCOPE_SENTINEL = "unknown_scope"

_FORBIDDEN_PAYLOAD_KEYS = frozenset(
    {
        "prompt",
        "prompts",
        "message",
        "messages",
        "input_text",
        "output_text",
        "completion",
        "payload",
        "media",
        "image_data",
        "audio_data",
        "video_data",
        "raw_headers",
        "raw_body",
        "response_body",
    }
)


class CanonicalizationError(ValueError):
    """Raised when a value cannot safely be represented as canonical JSON."""


class UsageIdentityError(ValueError):
    """Raised when an endpoint-usage identity cannot be formed safely."""


def is_secret_key(key: str) -> bool:
    """Return whether *key* conventionally names credential material.

    Stable non-secret identity handles such as ``credential_pseudonym`` and
    ``endpoint_fingerprint`` are intentionally allowed: they name digests, not
    raw secrets.
    """

    text = str(key)
    if re.search(r"(?:pseudonym|fingerprint)s?$", text, re.IGNORECASE):
        return False
    return bool(_SECRET_KEY.search(text))


def is_secret_value(value: str) -> bool:
    """Conservatively recognize common credential value formats."""

    return bool(_SECRET_VALUE.search(str(value).strip()))


def contains_bearer_url(value: str) -> bool:
    """Return whether *value* embeds a bearer URL, userinfo, or query secret."""

    return bool(_BEARER_URL.search(str(value)))


def contains_raw_endpoint(value: str) -> bool:
    """Return whether *value* looks like a raw host or URL (not a fingerprint)."""

    return bool(_RAW_ENDPOINT.search(str(value)))


def redact_secrets(value: Any) -> Any:
    """Return a recursively redacted, JSON-compatible copy of *value*."""

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
            raise CanonicalizationError(
                "credential-shaped string is not canonical catalog data"
            )
        return value
    if isinstance(value, Mapping):
        if len(value) > MAX_CONTAINER_ITEMS:
            raise CanonicalizationError("mapping exceeds canonical item bound")
        result = {}
        for key, item in value.items():
            if not isinstance(key, str):
                raise CanonicalizationError("canonical JSON mapping keys must be strings")
            if reject_secrets and is_secret_key(key) and item != REDACTED:
                raise CanonicalizationError(
                    "credential-bearing field is forbidden: %s" % key
                )
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
    raise CanonicalizationError(
        "unsupported canonical value type: %s" % type(value).__name__
    )


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
    binary_cid = b"\x01\x55\x12\x20" + digest
    return "b" + base64.b32encode(binary_cid).decode("ascii").lower().rstrip("=")


def stable_id(kind: str, *components: Any) -> str:
    """Build a collision-resistant stable ID from framed identity components."""

    if not isinstance(kind, str) or not _ID_KIND.fullmatch(kind):
        raise ValueError("identity kind must match %s" % _ID_KIND.pattern)
    material = {
        "identity_version": IDENTITY_VERSION,
        "kind": kind,
        "components": list(components),
    }
    digest = hashlib.sha256(canonical_json_bytes(material)).hexdigest()
    return "%s_%s" % (kind, digest)


def _require_text(
    value: Any, field: str, *, maximum: int = MAX_NAME_BYTES
) -> str:
    if not isinstance(value, str):
        raise UsageIdentityError("%s must be a string" % field)
    if value != value.strip() and value.strip():
        raise UsageIdentityError("%s must not contain surrounding whitespace" % field)
    text = value.strip()
    if not text:
        raise UsageIdentityError("%s must not be empty" % field)
    if len(text.encode("utf-8")) > maximum:
        raise UsageIdentityError("%s exceeds %d UTF-8 bytes" % (field, maximum))
    if any(ord(char) < 32 or ord(char) == 127 for char in text):
        raise UsageIdentityError("%s contains control characters" % field)
    return text


def require_catalog_id(value: Any, field: str = "id") -> str:
    """Validate a stable catalog-style ``kind_hex`` identifier."""

    text = _require_text(value, field, maximum=128)
    if not _CATALOG_ID.fullmatch(text):
        raise UsageIdentityError("%s is not a stable catalog identifier" % field)
    return text


def normalize_endpoint_uri(value: Any) -> str:
    """Normalize a public endpoint URI for fingerprinting only.

    Userinfo, query strings, fragments, and non-http(s)/unix schemes are
    rejected.  The returned value is intermediate material for hashing and must
    not be persisted on usage records.
    """

    text = _require_text(value, "endpoint_uri", maximum=MAX_ENDPOINT_BYTES)
    if contains_bearer_url(text):
        raise UsageIdentityError("endpoint_uri embeds bearer or credential material")
    try:
        parts = urlsplit(text)
        port = parts.port
    except ValueError as exc:
        raise UsageIdentityError("endpoint_uri is malformed") from exc
    scheme = parts.scheme.casefold()
    if scheme not in ("http", "https", "unix"):
        raise UsageIdentityError("endpoint_uri has an unsupported URI scheme")
    if parts.username is not None or parts.password is not None:
        raise UsageIdentityError("endpoint_uri must not contain user information")
    if parts.query or parts.fragment:
        raise UsageIdentityError("endpoint_uri must not contain query or fragment")
    if scheme == "unix":
        if not parts.path.startswith("/"):
            raise UsageIdentityError("unix endpoint_uri must use an absolute path")
        return urlunsplit((scheme, "", parts.path, "", ""))
    if not parts.hostname:
        raise UsageIdentityError("endpoint_uri must include a host")
    host = parts.hostname.casefold()
    if ":" in host and not host.startswith("["):
        host = "[%s]" % host
    if port is not None and not (
        (scheme == "http" and port == 80) or (scheme == "https" and port == 443)
    ):
        host = "%s:%d" % (host, port)
    path = parts.path or "/"
    return urlunsplit((scheme, host, path, "", ""))


def endpoint_fingerprint(endpoint_uri: Any) -> str:
    """Return a secret-free fingerprint for a normalized endpoint URI."""

    normalized = normalize_endpoint_uri(endpoint_uri)
    return stable_id(
        "ep",
        {
            "identity_policy_version": IDENTITY_POLICY_VERSION,
            "normalized_endpoint": normalized,
        },
    )


def credential_configuration_pseudonym(
    config_reference: Any, *, key_id: Any
) -> str:
    """Derive a keyed local pseudonym for a credential *configuration* reference.

    The reference names where a credential is configured (for example
    ``env:OPENAI_API_KEY`` or ``secret_store:openai/default``).  Raw tokens are
    rejected.  The ledger *key_id* scopes the pseudonym so offline comparison of
    token values across ledgers is not possible from the digest alone.
    """

    reference = _require_text(config_reference, "config_reference", maximum=256)
    if is_secret_value(reference) or contains_bearer_url(reference):
        raise UsageIdentityError(
            "config_reference must not carry credential material"
        )
    if not _CONFIG_REF.fullmatch(reference):
        raise UsageIdentityError(
            "config_reference must be a typed non-secret configuration handle"
        )
    key = _require_text(key_id, "key_id", maximum=128)
    if not _KEY_ID.fullmatch(key):
        raise UsageIdentityError("key_id is not a valid ledger key identifier")
    return stable_id(
        "cred",
        {
            "identity_policy_version": IDENTITY_POLICY_VERSION,
            "key_id": key,
            "config_reference": reference,
        },
    )


def account_pseudonym(account: Any, *, provider_id: Any) -> str:
    """Stable non-secret account pseudonym scoped to a provider identity."""

    provider = require_catalog_id(provider_id, "provider_id")
    text = _require_text(account, "account", maximum=MAX_NAME_BYTES)
    if contains_raw_endpoint(text):
        raise UsageIdentityError("account must not embed a raw endpoint")
    return stable_id(
        "acct",
        {
            "identity_policy_version": IDENTITY_POLICY_VERSION,
            "provider_id": provider,
            "account": text.casefold(),
        },
    )


def project_pseudonym(project: Any, *, provider_id: Any) -> str:
    """Stable non-secret project/tenant pseudonym scoped to a provider identity."""

    provider = require_catalog_id(provider_id, "provider_id")
    text = _require_text(project, "project", maximum=MAX_NAME_BYTES)
    if contains_raw_endpoint(text):
        raise UsageIdentityError("project must not embed a raw endpoint")
    return stable_id(
        "proj",
        {
            "identity_policy_version": IDENTITY_POLICY_VERSION,
            "provider_id": provider,
            "project": text.casefold(),
        },
    )


def organization_pseudonym(organization: Any, *, provider_id: Any) -> str:
    """Stable non-secret organization pseudonym scoped to a provider identity."""

    provider = require_catalog_id(provider_id, "provider_id")
    text = _require_text(organization, "organization", maximum=MAX_NAME_BYTES)
    if contains_raw_endpoint(text):
        raise UsageIdentityError("organization must not embed a raw endpoint")
    return stable_id(
        "org",
        {
            "identity_policy_version": IDENTITY_POLICY_VERSION,
            "provider_id": provider,
            "organization": text.casefold(),
        },
    )


def scope_identity_components(
    *,
    provider_id: str,
    protocol: str,
    operation: str,
    deployment_id: Optional[str] = None,
    endpoint_fingerprint_value: Optional[str] = None,
    model_id: Optional[str] = None,
    account_pseudonym_value: Optional[str] = None,
    project_pseudonym_value: Optional[str] = None,
    organization_pseudonym_value: Optional[str] = None,
    region: Optional[str] = None,
    credential_pseudonym_value: Optional[str] = None,
    unknown_scope: bool = False,
) -> Mapping[str, Any]:
    """Return framed components used by :func:`endpoint_usage_scope_identity`."""

    if unknown_scope:
        return {
            "identity_policy_version": IDENTITY_POLICY_VERSION,
            "unknown_scope": True,
            "isolation": UNKNOWN_SCOPE_SENTINEL,
            "provider_id": provider_id,
            "protocol": protocol,
            "operation": operation,
            "credential_pseudonym": credential_pseudonym_value,
        }
    if not deployment_id and not endpoint_fingerprint_value:
        raise UsageIdentityError(
            "scope requires deployment_id or endpoint_fingerprint"
        )
    return {
        "identity_policy_version": IDENTITY_POLICY_VERSION,
        "unknown_scope": False,
        "provider_id": provider_id,
        "deployment_id": deployment_id,
        "endpoint_fingerprint": endpoint_fingerprint_value,
        "protocol": protocol,
        "operation": operation,
        "model_id": model_id,
        "account_pseudonym": account_pseudonym_value,
        "project_pseudonym": project_pseudonym_value,
        "organization_pseudonym": organization_pseudonym_value,
        "region": region,
        "credential_pseudonym": credential_pseudonym_value,
    }


def endpoint_usage_scope_identity(components: Mapping[str, Any]) -> str:
    """Return the stable scope identity for the framed *components* mapping."""

    if not isinstance(components, Mapping):
        raise UsageIdentityError("scope components must be a mapping")
    if bool(components.get("unknown_scope")):
        framed = scope_identity_components(
            provider_id=str(components.get("provider_id", "")),
            protocol=str(components.get("protocol", "")),
            operation=str(components.get("operation", "")),
            credential_pseudonym_value=components.get("credential_pseudonym"),
            unknown_scope=True,
        )
    else:
        framed = scope_identity_components(
            provider_id=str(components.get("provider_id", "")),
            protocol=str(components.get("protocol", "")),
            operation=str(components.get("operation", "")),
            deployment_id=components.get("deployment_id"),
            endpoint_fingerprint_value=components.get("endpoint_fingerprint"),
            model_id=components.get("model_id"),
            account_pseudonym_value=components.get("account_pseudonym"),
            project_pseudonym_value=components.get("project_pseudonym"),
            organization_pseudonym_value=components.get("organization_pseudonym"),
            region=components.get("region"),
            credential_pseudonym_value=components.get("credential_pseudonym"),
            unknown_scope=False,
        )
    return stable_id("scope", framed)


def event_identity(*components: Any) -> str:
    """Content-addressed usage event identity."""

    return stable_id("uevt", *components)


def reservation_identity(*components: Any) -> str:
    """Stable reservation identity from framed components."""

    return stable_id("ures", *components)


def snapshot_identity(*components: Any) -> str:
    """Stable usage snapshot revision identity."""

    return stable_id("usnap", *components)


def receipt_identity(*components: Any) -> str:
    """Stable usage routing receipt identity."""

    return stable_id("urcpt", *components)


def is_pseudonym(value: Any) -> bool:
    """Return whether *value* matches a usage pseudonym or catalog id form."""

    if not isinstance(value, str):
        return False
    return bool(_PSEUDONYM.fullmatch(value) or _CATALOG_ID.fullmatch(value))


def assert_no_prompt_media_or_output(value: Any, path: str = "$") -> None:
    """Reject nested structures that embed prompts, media, or model output."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            folded = name.casefold()
            if folded in _FORBIDDEN_PAYLOAD_KEYS or is_secret_key(name):
                raise UsageIdentityError(
                    "%s contains forbidden field %r" % (path, name)
                )
            assert_no_prompt_media_or_output(item, "%s.%s" % (path, name))
        return
    if isinstance(value, (list, tuple, set, frozenset)):
        for index, item in enumerate(value):
            assert_no_prompt_media_or_output(item, "%s[%d]" % (path, index))
        return
    if isinstance(value, str):
        if is_secret_value(value) or contains_bearer_url(value):
            raise UsageIdentityError("%s contains credential-shaped data" % path)
        if len(value.encode("utf-8")) > 4096:
            raise UsageIdentityError("%s exceeds safe identity text bound" % path)


# Friendly aliases for consumers that use "serialize" or "CID" terminology.
canonical_serialize = canonical_json_bytes
canonical_cid = content_cid


__all__ = [
    "CanonicalizationError",
    "IDENTITY_POLICY_VERSION",
    "IDENTITY_VERSION",
    "MAX_ABS_INTEGER",
    "MAX_CANONICAL_BYTES",
    "MAX_CANONICAL_DEPTH",
    "MAX_CONTAINER_ITEMS",
    "MAX_STRING_BYTES",
    "REDACTED",
    "UNKNOWN_SCOPE_SENTINEL",
    "UsageIdentityError",
    "account_pseudonym",
    "assert_no_prompt_media_or_output",
    "canonical_cid",
    "canonical_data",
    "canonical_json",
    "canonical_json_bytes",
    "canonical_serialize",
    "contains_bearer_url",
    "contains_raw_endpoint",
    "content_cid",
    "credential_configuration_pseudonym",
    "endpoint_fingerprint",
    "endpoint_usage_scope_identity",
    "event_identity",
    "is_pseudonym",
    "is_secret_key",
    "is_secret_value",
    "normalize_endpoint_uri",
    "organization_pseudonym",
    "project_pseudonym",
    "receipt_identity",
    "redact_secrets",
    "require_catalog_id",
    "reservation_identity",
    "scope_identity_components",
    "snapshot_identity",
    "stable_id",
]
