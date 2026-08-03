"""Transient, capability-protected prompt-body brokering.

Carries exact prompt bytes from CLI/Python/MCP intake to bounded planner use
without placing the body on durable requests, results, events, logs, argv,
environment, or ordinary state.  Callers persist only
:class:`PromptReference` (CID + opaque handle).  Retrieval requires a matching
:class:`PromptCapability` bound to the same run, within the authorized window.

Storage modes:

* **memory** (default) — process-local ``bytearray`` buffers.  Restart loses
  the body; this is reported explicitly via :meth:`PromptBodyBroker.restart_behavior`.
* **encrypted_artifact** — Fernet ciphertext under an optional artifact root.
  Restart recovers the body only when the same master secret, artifact root,
  valid capability, and unexpired window are all present.

Internal buffers are zeroized on release, expiry, use exhaustion, and broker
close.  Capability tokens are high-entropy secrets returned once; only their
digests appear on durable broker surfaces.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import Any, ClassVar, Final

from cryptography.fernet import Fernet, InvalidToken


def _load_cid_bridge() -> tuple[type[Exception], Callable[..., str], Callable[..., str]]:
    """Prefer the supervisor identity bridge; fall back to multiformats.

    Hermetic validation runs with ``PYTHONNOUSERSITE=1`` and empty worktree
    stubs for sibling packages, so the editable ``ipfs_datasets_py`` install
    used by :mod:`multiformats_identity` is often unavailable.  CIDv1/raw/
    sha2-256 is still required for prompt references, so a direct multiformats
    path keeps the broker importable and behaviorally equivalent.
    """

    try:
        from ..multiformats_identity import (  # type: ignore[attr-defined]
            MultiformatsIdentityError as identity_error,
            cid_for_bytes as identity_cid_for_bytes,
            validate_cid as identity_validate_cid,
        )

        return identity_error, identity_cid_for_bytes, identity_validate_cid
    except ModuleNotFoundError:
        pass

    class _IdentityError(ValueError):
        """Local stand-in when the multiformats identity bridge is unavailable."""

    def _cid_for_bytes(
        data: bytes,
        *,
        base: str = "base32",
        codec: str = "raw",
        mh_type: str = "sha2-256",
        version: int = 1,
    ) -> str:
        if not isinstance(data, (bytes, bytearray, memoryview)):
            raise _IdentityError("cid payload must be exact bytes")
        if base != "base32" or mh_type != "sha2-256" or version != 1:
            raise _IdentityError(
                "only CIDv1/base32/sha2-256 is supported by the prompt broker"
            )
        if codec not in {"raw", "dag-json"}:
            raise _IdentityError(f"unsupported codec: {codec}")
        from multiformats import CID, multihash

        digest = multihash.digest(bytes(data), mh_type)
        return str(CID(base, version, codec, digest))

    def _validate_cid(
        value: Any,
        *,
        codecs: Sequence[str] = ("raw", "dag-json"),
        mh_type: str = "sha2-256",
        version: int = 1,
        base: str = "base32",
    ) -> str:
        if not isinstance(value, str) or not value or value != value.lower():
            raise _IdentityError("CID must be a nonempty lowercase string")
        from multiformats import CID, multihash

        try:
            parsed = CID.decode(value)
        except Exception as exc:  # noqa: BLE001 - library raises mixed types
            raise _IdentityError("CID is not decodable") from exc
        allowed = frozenset(codecs)
        expected_size = multihash.get(mh_type).max_digest_size
        if (
            parsed.version != version
            or parsed.codec.name not in allowed
            or parsed.hashfun.name != mh_type
            or (
                expected_size is not None
                and len(parsed.raw_digest) != expected_size
            )
            or parsed.base.name != base
            or str(parsed) != value
        ):
            raise _IdentityError(
                "CID must use the requested canonical version/base/codec/multihash"
            )
        return value

    return _IdentityError, _cid_for_bytes, _validate_cid


MultiformatsIdentityError, cid_for_bytes, validate_cid = _load_cid_bridge()

PROMPT_BROKER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/prompt-broker@1"
)
PROMPT_REFERENCE_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/prompt-reference@1"
)
PROMPT_CAPABILITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/entrypoints/prompt-capability@1"
)
PROMPT_BROKER_REQUIREMENT_ID: Final = (
    "prompt_broker.TRANSIENT_CAPABILITY_PROTECTED_BODY_REQUIREMENT_ID"
)

DEFAULT_TTL_MS: Final = 15 * 60 * 1000
MAX_TTL_MS: Final = 24 * 60 * 60 * 1000
DEFAULT_MAX_PROMPT_BYTES: Final = 1024 * 1024
DEFAULT_MAX_USES: Final = 1
MAX_USES_BOUND: Final = 64
MAX_RUN_ID_BYTES: Final = 512
MAX_PURPOSE_BYTES: Final = 128
MAX_REFERENCE_BYTES: Final = 2_048
ARTIFACT_INDEX_NAME: Final = "prompt_broker_index.json"
ARTIFACT_BLOB_DIR: Final = "bodies"
MASTER_SECRET_ENV: Final = "IPFS_ACCELERATE_PROMPT_BROKER_MASTER_SECRET"

_REFERENCE_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$")
_PURPOSE_RE = re.compile(r"^[a-z][a-z0-9_:-]*$")
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{32,256}$")


class PromptBrokerError(ValueError):
    """Base error for the prompt body broker."""

    def __init__(self, message: str, *, reason_code: str = "broker_error") -> None:
        super().__init__(message)
        self.reason_code = reason_code


class PromptCapabilityError(PromptBrokerError):
    """Capability missing, forged, exhausted, or mismatched."""

    def __init__(
        self, message: str, *, reason_code: str = "capability_denied"
    ) -> None:
        super().__init__(message, reason_code=reason_code)


class PromptExpiredError(PromptBrokerError):
    """The authorized window for a prompt body has closed."""

    def __init__(
        self, message: str = "prompt body authorization window expired"
    ) -> None:
        super().__init__(message, reason_code="expired")


class PromptNotFoundError(PromptBrokerError):
    """No active body is bound to the requested reference."""

    def __init__(self, message: str = "prompt body not found") -> None:
        super().__init__(message, reason_code="not_found")


class PromptCrossRunError(PromptCapabilityError):
    """Capability or reference is bound to a different run."""

    def __init__(
        self, message: str = "prompt body is not authorized for this run"
    ) -> None:
        super().__init__(message, reason_code="cross_run_denied")


class PromptBrokerBoundsError(PromptBrokerError):
    """A deposit exceeds a frozen size or TTL bound."""

    def __init__(self, message: str, *, reason_code: str = "bounds") -> None:
        super().__init__(message, reason_code=reason_code)


class PromptStorageKind(str, Enum):
    MEMORY = "memory"
    ENCRYPTED_ARTIFACT = "encrypted_artifact"


class PromptBodyStatus(str, Enum):
    ACTIVE = "active"
    RELEASED = "released"
    EXPIRED = "expired"
    EXHAUSTED = "exhausted"
    MISSING_AFTER_RESTART = "missing_after_restart"


def _now_ms() -> int:
    return int(time.time() * 1000)


def _digest_hex(value: bytes | str) -> str:
    if isinstance(value, str):
        payload = value.encode("utf-8")
    else:
        payload = value
    return hashlib.sha256(payload).hexdigest()


def _token_digest(token: str) -> str:
    return f"sha256:{_digest_hex(token)}"


def _zeroize_bytearray(buffer: bytearray | None) -> None:
    if buffer is None:
        return
    for index in range(len(buffer)):
        buffer[index] = 0
    buffer.clear()


def _require_text(
    value: Any,
    name: str,
    *,
    maximum: int,
    pattern: re.Pattern[str] | None = None,
) -> str:
    if not isinstance(value, str) or not value:
        raise PromptBrokerError(f"{name} must be a nonempty string")
    if len(value.encode("utf-8")) > maximum:
        raise PromptBrokerBoundsError(f"{name} exceeds {maximum} bytes")
    if pattern is not None and not pattern.fullmatch(value):
        raise PromptBrokerError(f"{name} has an invalid format")
    if "=" in value:
        raise PromptBrokerError(f"{name} must not contain assignment material")
    return value


def _require_prompt_cid(value: Any) -> str:
    text = _require_text(value, "prompt_cid", maximum=256)
    try:
        return validate_cid(text, codecs=("raw",))
    except (MultiformatsIdentityError, TypeError, ValueError) as exc:
        raise PromptBrokerError("prompt_cid must be a canonical raw CIDv1") from exc


def _coerce_body(body: str | bytes | bytearray | memoryview) -> bytes:
    if isinstance(body, str):
        if not body:
            raise PromptBrokerBoundsError("prompt body must not be empty")
        return body.encode("utf-8")
    if isinstance(body, (bytes, bytearray, memoryview)):
        payload = bytes(body)
        if not payload:
            raise PromptBrokerBoundsError("prompt body must not be empty")
        return payload
    raise PromptBrokerError("prompt body must be text or exact bytes")


def _capability_token() -> str:
    # urlsafe without padding so tokens stay reference-safe and unpadded.
    return secrets.token_urlsafe(32).rstrip("=")


def _opaque_prompt_ref() -> str:
    return f"prompt-broker:{secrets.token_hex(16)}"


def _master_secret_bytes(value: bytes | str | None) -> bytes | None:
    if value is None:
        return None
    if isinstance(value, str):
        if not value:
            return None
        return value.encode("utf-8")
    if isinstance(value, (bytes, bytearray, memoryview)):
        payload = bytes(value)
        return payload or None
    raise PromptBrokerError("master_secret must be bytes or text")


def _fernet_from_master(master_secret: bytes, prompt_ref: str) -> Fernet:
    material = hmac.new(
        master_secret,
        f"prompt-broker-body-v1:{prompt_ref}".encode("utf-8"),
        hashlib.sha256,
    ).digest()
    # Fernet keys are 32 url-safe base64-encoded bytes.
    return Fernet(base64.urlsafe_b64encode(material))


@dataclass(frozen=True)
class PromptReference:
    """Body-free durable handle for a brokered prompt.

    Safe for requests, results, events, logs, argv, environment, and state.
    Never carries raw prompt text or capability secrets.
    """

    SCHEMA: ClassVar[str] = PROMPT_REFERENCE_SCHEMA

    prompt_cid: str
    prompt_ref: str
    run_id: str
    byte_count: int
    issued_at_ms: int
    expires_at_ms: int
    storage: PromptStorageKind
    purpose: str = "planner"
    artifact_ref: str = ""
    status: PromptBodyStatus = PromptBodyStatus.ACTIVE

    def __post_init__(self) -> None:
        object.__setattr__(self, "prompt_cid", _require_prompt_cid(self.prompt_cid))
        object.__setattr__(
            self,
            "prompt_ref",
            _require_text(
                self.prompt_ref,
                "prompt_ref",
                maximum=MAX_REFERENCE_BYTES,
                pattern=_REFERENCE_RE,
            ),
        )
        object.__setattr__(
            self,
            "run_id",
            _require_text(
                self.run_id,
                "run_id",
                maximum=MAX_RUN_ID_BYTES,
                pattern=_RUN_ID_RE,
            ),
        )
        if not isinstance(self.byte_count, int) or isinstance(self.byte_count, bool):
            raise PromptBrokerError("byte_count must be an integer")
        if self.byte_count < 1 or self.byte_count > DEFAULT_MAX_PROMPT_BYTES * 4:
            raise PromptBrokerBoundsError("byte_count is out of bounds")
        if not isinstance(self.issued_at_ms, int) or isinstance(
            self.issued_at_ms, bool
        ):
            raise PromptBrokerError("issued_at_ms must be an integer")
        if not isinstance(self.expires_at_ms, int) or isinstance(
            self.expires_at_ms, bool
        ):
            raise PromptBrokerError("expires_at_ms must be an integer")
        if self.expires_at_ms <= self.issued_at_ms:
            raise PromptBrokerError("expires_at_ms must follow issued_at_ms")
        storage = self.storage
        if isinstance(storage, str):
            storage = PromptStorageKind(storage)
        if not isinstance(storage, PromptStorageKind):
            raise PromptBrokerError("storage must be a PromptStorageKind")
        object.__setattr__(self, "storage", storage)
        object.__setattr__(
            self,
            "purpose",
            _require_text(
                self.purpose,
                "purpose",
                maximum=MAX_PURPOSE_BYTES,
                pattern=_PURPOSE_RE,
            ),
        )
        artifact = self.artifact_ref or ""
        if artifact:
            artifact = _require_text(
                artifact,
                "artifact_ref",
                maximum=MAX_REFERENCE_BYTES,
                pattern=_REFERENCE_RE,
            )
            if artifact.startswith("/") or ".." in artifact.split("/"):
                raise PromptBrokerError("artifact_ref escapes broker roots")
        object.__setattr__(self, "artifact_ref", artifact)
        status = self.status
        if isinstance(status, str):
            status = PromptBodyStatus(status)
        if not isinstance(status, PromptBodyStatus):
            raise PromptBrokerError("status must be a PromptBodyStatus")
        object.__setattr__(self, "status", status)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "prompt_cid": self.prompt_cid,
            "prompt_ref": self.prompt_ref,
            "run_id": self.run_id,
            "byte_count": self.byte_count,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "storage": self.storage.value,
            "purpose": self.purpose,
            "artifact_ref": self.artifact_ref,
            "status": self.status.value,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), separators=(",", ":"), sort_keys=True)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> PromptReference:
        if not isinstance(value, Mapping):
            raise PromptBrokerError("PromptReference payload must be an object")
        allowed = {
            "schema",
            "prompt_cid",
            "prompt_ref",
            "run_id",
            "byte_count",
            "issued_at_ms",
            "expires_at_ms",
            "storage",
            "purpose",
            "artifact_ref",
            "status",
        }
        unknown = set(value) - allowed
        if unknown:
            raise PromptBrokerError(
                f"unknown PromptReference fields: {sorted(unknown)}"
            )
        schema = value.get("schema", cls.SCHEMA)
        if schema != cls.SCHEMA:
            raise PromptBrokerError("unsupported PromptReference schema")
        return cls(
            prompt_cid=value["prompt_cid"],
            prompt_ref=value["prompt_ref"],
            run_id=value["run_id"],
            byte_count=value["byte_count"],
            issued_at_ms=value["issued_at_ms"],
            expires_at_ms=value["expires_at_ms"],
            storage=value["storage"],
            purpose=value.get("purpose", "planner"),
            artifact_ref=value.get("artifact_ref", ""),
            status=value.get("status", PromptBodyStatus.ACTIVE),
        )


@dataclass(frozen=True)
class PromptCapability:
    """Single-process capability authorizing exact-byte retrieval.

    The ``token`` is a high-entropy secret.  It is returned once at deposit and
    must not be written to durable surfaces.  :meth:`redacted_dict` is the only
    log/event-safe projection.
    """

    SCHEMA: ClassVar[str] = PROMPT_CAPABILITY_SCHEMA

    token: str = field(repr=False)
    prompt_ref: str
    run_id: str
    prompt_cid: str
    issued_at_ms: int
    expires_at_ms: int
    max_uses: int = DEFAULT_MAX_USES
    purpose: str = "planner"
    capability_digest: str = ""

    def __post_init__(self) -> None:
        token = self.token
        if not isinstance(token, str) or not _TOKEN_RE.fullmatch(token):
            raise PromptCapabilityError("capability token is malformed")
        object.__setattr__(self, "token", token)
        object.__setattr__(
            self,
            "prompt_ref",
            _require_text(
                self.prompt_ref,
                "prompt_ref",
                maximum=MAX_REFERENCE_BYTES,
                pattern=_REFERENCE_RE,
            ),
        )
        object.__setattr__(
            self,
            "run_id",
            _require_text(
                self.run_id,
                "run_id",
                maximum=MAX_RUN_ID_BYTES,
                pattern=_RUN_ID_RE,
            ),
        )
        object.__setattr__(self, "prompt_cid", _require_prompt_cid(self.prompt_cid))
        if not isinstance(self.issued_at_ms, int) or isinstance(
            self.issued_at_ms, bool
        ):
            raise PromptBrokerError("issued_at_ms must be an integer")
        if not isinstance(self.expires_at_ms, int) or isinstance(
            self.expires_at_ms, bool
        ):
            raise PromptBrokerError("expires_at_ms must be an integer")
        if self.expires_at_ms <= self.issued_at_ms:
            raise PromptBrokerError("expires_at_ms must follow issued_at_ms")
        if (
            not isinstance(self.max_uses, int)
            or isinstance(self.max_uses, bool)
            or self.max_uses < 1
            or self.max_uses > MAX_USES_BOUND
        ):
            raise PromptBrokerBoundsError("max_uses is out of bounds")
        object.__setattr__(
            self,
            "purpose",
            _require_text(
                self.purpose,
                "purpose",
                maximum=MAX_PURPOSE_BYTES,
                pattern=_PURPOSE_RE,
            ),
        )
        digest = self.capability_digest or _token_digest(token)
        if digest != _token_digest(token):
            raise PromptCapabilityError("capability_digest does not match token")
        object.__setattr__(self, "capability_digest", digest)

    def __repr__(self) -> str:
        return (
            "PromptCapability("
            f"prompt_ref={self.prompt_ref!r}, run_id={self.run_id!r}, "
            f"prompt_cid={self.prompt_cid!r}, purpose={self.purpose!r}, "
            "token=<redacted>)"
        )

    __str__ = __repr__

    def redacted_dict(self) -> dict[str, Any]:
        """Safe projection for logs/events — never includes the token."""

        return {
            "schema": self.SCHEMA,
            "prompt_ref": self.prompt_ref,
            "run_id": self.run_id,
            "prompt_cid": self.prompt_cid,
            "issued_at_ms": self.issued_at_ms,
            "expires_at_ms": self.expires_at_ms,
            "max_uses": self.max_uses,
            "purpose": self.purpose,
            "capability_digest": self.capability_digest,
            "token_present": True,
            "token_redacted": True,
        }

    def to_dict(self) -> dict[str, Any]:
        """Alias of :meth:`redacted_dict` so accidental serialization stays safe."""

        return self.redacted_dict()


@dataclass
class _LiveEntry:
    reference: PromptReference
    capability_digest: str
    max_uses: int
    uses: int
    body: bytearray | None
    status: PromptBodyStatus
    purpose: str

    def remaining_uses(self) -> int:
        return max(0, self.max_uses - self.uses)


class PromptBodyBroker:
    """Process-local prompt body channel with capability-gated retrieval.

    Parameters
    ----------
    artifact_dir:
        Optional directory for encrypted continuation artifacts and the durable
        index.  When omitted, only in-memory storage is available.
    master_secret:
        Secret material used to derive per-body Fernet keys.  Required for
        encrypted artifacts and for restart recovery.  When omitted, a random
        process-local secret is generated and restart cannot decrypt artifacts.
    clock_ms:
        Injectable millisecond clock for deterministic expiry tests.
    default_ttl_ms / max_prompt_bytes:
        Bounded defaults for deposit windows and body size.
    """

    def __init__(
        self,
        *,
        artifact_dir: str | Path | None = None,
        master_secret: bytes | str | None = None,
        clock_ms: Callable[[], int] | None = None,
        default_ttl_ms: int = DEFAULT_TTL_MS,
        max_prompt_bytes: int = DEFAULT_MAX_PROMPT_BYTES,
        default_max_uses: int = DEFAULT_MAX_USES,
    ) -> None:
        if (
            not isinstance(default_ttl_ms, int)
            or isinstance(default_ttl_ms, bool)
            or default_ttl_ms < 1
            or default_ttl_ms > MAX_TTL_MS
        ):
            raise PromptBrokerBoundsError("default_ttl_ms is out of bounds")
        if (
            not isinstance(max_prompt_bytes, int)
            or isinstance(max_prompt_bytes, bool)
            or max_prompt_bytes < 1
            or max_prompt_bytes > DEFAULT_MAX_PROMPT_BYTES
        ):
            raise PromptBrokerBoundsError("max_prompt_bytes is out of bounds")
        if (
            not isinstance(default_max_uses, int)
            or isinstance(default_max_uses, bool)
            or default_max_uses < 1
            or default_max_uses > MAX_USES_BOUND
        ):
            raise PromptBrokerBoundsError("default_max_uses is out of bounds")

        env_secret = os.environ.get(MASTER_SECRET_ENV)
        resolved_secret = _master_secret_bytes(master_secret)
        if resolved_secret is None and env_secret:
            resolved_secret = _master_secret_bytes(env_secret)
        self._ephemeral_master = resolved_secret is None
        if resolved_secret is None:
            resolved_secret = secrets.token_bytes(32)

        self._master_secret = resolved_secret
        self._clock_ms = clock_ms or _now_ms
        self._default_ttl_ms = default_ttl_ms
        self._max_prompt_bytes = max_prompt_bytes
        self._default_max_uses = default_max_uses
        self._lock = threading.RLock()
        self._entries: dict[str, _LiveEntry] = {}
        self._closed = False

        if artifact_dir is None:
            self._artifact_dir: Path | None = None
        else:
            path = Path(artifact_dir)
            path.mkdir(parents=True, exist_ok=True)
            (path / ARTIFACT_BLOB_DIR).mkdir(parents=True, exist_ok=True)
            self._artifact_dir = path.resolve()
            self._load_index()

    # ------------------------------------------------------------------ public

    @property
    def artifact_dir(self) -> Path | None:
        return self._artifact_dir

    @property
    def max_prompt_bytes(self) -> int:
        return self._max_prompt_bytes

    def restart_behavior(self) -> dict[str, Any]:
        """Explicit description of restart/expiry recovery semantics."""

        encrypted_possible = self._artifact_dir is not None
        recoverable = encrypted_possible and not self._ephemeral_master
        return {
            "schema": PROMPT_BROKER_SCHEMA,
            "requirement_id": PROMPT_BROKER_REQUIREMENT_ID,
            "memory_bodies_survive_restart": False,
            "encrypted_artifacts_enabled": encrypted_possible,
            "encrypted_artifacts_recoverable_after_restart": recoverable,
            "master_secret_source": (
                "caller_or_environment"
                if not self._ephemeral_master
                else "ephemeral_process_local"
            ),
            "expired_bodies_recoverable": False,
            "capability_required_after_restart": True,
            "cross_run_access": "denied",
            "routine_surfaces": "cid_and_reference_only",
        }

    def deposit(
        self,
        body: str | bytes | bytearray | memoryview,
        *,
        run_id: str,
        ttl_ms: int | None = None,
        max_uses: int | None = None,
        purpose: str = "planner",
        enable_encrypted_artifact: bool = False,
        now_ms: int | None = None,
    ) -> tuple[PromptReference, PromptCapability]:
        """Accept exact prompt bytes and return a reference plus capability."""

        with self._lock:
            self._ensure_open()
            payload = _coerce_body(body)
            if len(payload) > self._max_prompt_bytes:
                raise PromptBrokerBoundsError(
                    "prompt body exceeds broker max_prompt_bytes"
                )
            run = _require_text(
                run_id, "run_id", maximum=MAX_RUN_ID_BYTES, pattern=_RUN_ID_RE
            )
            purpose_text = _require_text(
                purpose, "purpose", maximum=MAX_PURPOSE_BYTES, pattern=_PURPOSE_RE
            )
            ttl = self._default_ttl_ms if ttl_ms is None else ttl_ms
            if (
                not isinstance(ttl, int)
                or isinstance(ttl, bool)
                or ttl < 1
                or ttl > MAX_TTL_MS
            ):
                raise PromptBrokerBoundsError("ttl_ms is out of bounds")
            uses = self._default_max_uses if max_uses is None else max_uses
            if (
                not isinstance(uses, int)
                or isinstance(uses, bool)
                or uses < 1
                or uses > MAX_USES_BOUND
            ):
                raise PromptBrokerBoundsError("max_uses is out of bounds")
            if enable_encrypted_artifact and self._artifact_dir is None:
                raise PromptBrokerError(
                    "enable_encrypted_artifact requires artifact_dir",
                    reason_code="artifact_unavailable",
                )

            issued = self._clock_ms() if now_ms is None else int(now_ms)
            expires = issued + ttl
            prompt_cid = cid_for_bytes(payload, codec="raw")
            prompt_ref = _opaque_prompt_ref()
            token = _capability_token()
            capability_digest = _token_digest(token)
            storage = (
                PromptStorageKind.ENCRYPTED_ARTIFACT
                if enable_encrypted_artifact
                else PromptStorageKind.MEMORY
            )
            artifact_ref = ""
            if storage is PromptStorageKind.ENCRYPTED_ARTIFACT:
                artifact_ref = f"{ARTIFACT_BLOB_DIR}/{prompt_ref.split(':', 1)[1]}.bin"
                self._write_encrypted_artifact(prompt_ref, artifact_ref, payload)

            reference = PromptReference(
                prompt_cid=prompt_cid,
                prompt_ref=prompt_ref,
                run_id=run,
                byte_count=len(payload),
                issued_at_ms=issued,
                expires_at_ms=expires,
                storage=storage,
                purpose=purpose_text,
                artifact_ref=artifact_ref,
                status=PromptBodyStatus.ACTIVE,
            )
            capability = PromptCapability(
                token=token,
                prompt_ref=prompt_ref,
                run_id=run,
                prompt_cid=prompt_cid,
                issued_at_ms=issued,
                expires_at_ms=expires,
                max_uses=uses,
                purpose=purpose_text,
                capability_digest=capability_digest,
            )
            entry = _LiveEntry(
                reference=reference,
                capability_digest=capability_digest,
                max_uses=uses,
                uses=0,
                body=bytearray(payload),
                status=PromptBodyStatus.ACTIVE,
                purpose=purpose_text,
            )
            self._entries[prompt_ref] = entry
            self._persist_index()
            return reference, capability

    def resolve(
        self,
        reference: PromptReference | str,
        capability: PromptCapability | str,
        *,
        run_id: str,
        now_ms: int | None = None,
        consume: bool = True,
    ) -> bytes:
        """Return exact prompt bytes during the authorized window.

        Validates capability digest, run binding, expiry, and remaining uses.
        By default one use is consumed; exhausted entries are released.
        """

        with self._lock:
            self._ensure_open()
            prompt_ref, expected_cid = self._reference_parts(reference)
            token = self._capability_token_value(capability)
            run = _require_text(
                run_id, "run_id", maximum=MAX_RUN_ID_BYTES, pattern=_RUN_ID_RE
            )
            now = self._clock_ms() if now_ms is None else int(now_ms)
            entry = self._entries.get(prompt_ref)
            if entry is None:
                raise PromptNotFoundError()
            self._validate_access(entry, token=token, run_id=run, now_ms=now)
            if expected_cid and expected_cid != entry.reference.prompt_cid:
                raise PromptCapabilityError(
                    "prompt_cid does not match brokered body",
                    reason_code="cid_mismatch",
                )
            body = self._materialize_body(entry)
            if cid_for_bytes(body, codec="raw") != entry.reference.prompt_cid:
                self._force_release(entry, PromptBodyStatus.RELEASED)
                raise PromptBrokerError(
                    "brokered body integrity check failed",
                    reason_code="integrity_failure",
                )
            if consume:
                entry.uses += 1
                if entry.uses >= entry.max_uses:
                    self._force_release(entry, PromptBodyStatus.EXHAUSTED)
                else:
                    self._persist_index()
            # Return an independent immutable copy for the planner.
            return bytes(body)

    @contextmanager
    def open_for_planner(
        self,
        reference: PromptReference | str,
        capability: PromptCapability | str,
        *,
        run_id: str,
        now_ms: int | None = None,
    ) -> Iterator[bytes]:
        """Yield exact bytes then rely on resolve's use accounting/zeroization."""

        body = self.resolve(
            reference,
            capability,
            run_id=run_id,
            now_ms=now_ms,
            consume=True,
        )
        try:
            yield body
        finally:
            # Caller-owned copy; best-effort scrub of the local name.
            del body

    def release(
        self,
        reference: PromptReference | str,
        *,
        run_id: str,
        capability: PromptCapability | str | None = None,
        now_ms: int | None = None,
    ) -> PromptReference:
        """Zeroize and delete a body before natural expiry."""

        with self._lock:
            self._ensure_open()
            prompt_ref, _ = self._reference_parts(reference)
            run = _require_text(
                run_id, "run_id", maximum=MAX_RUN_ID_BYTES, pattern=_RUN_ID_RE
            )
            now = self._clock_ms() if now_ms is None else int(now_ms)
            entry = self._entries.get(prompt_ref)
            if entry is None:
                raise PromptNotFoundError()
            if entry.reference.run_id != run:
                raise PromptCrossRunError()
            if capability is not None:
                token = self._capability_token_value(capability)
                if not hmac.compare_digest(
                    _token_digest(token), entry.capability_digest
                ):
                    raise PromptCapabilityError("capability token rejected")
            if now >= entry.reference.expires_at_ms:
                self._force_release(entry, PromptBodyStatus.EXPIRED)
            else:
                self._force_release(entry, PromptBodyStatus.RELEASED)
            return entry.reference

    def expire_due(self, *, now_ms: int | None = None) -> tuple[str, ...]:
        """Release every body whose authorized window has closed."""

        with self._lock:
            self._ensure_open()
            now = self._clock_ms() if now_ms is None else int(now_ms)
            expired: list[str] = []
            for prompt_ref, entry in list(self._entries.items()):
                if (
                    entry.status is PromptBodyStatus.ACTIVE
                    and now >= entry.reference.expires_at_ms
                ):
                    self._force_release(entry, PromptBodyStatus.EXPIRED)
                    expired.append(prompt_ref)
            return tuple(expired)

    def describe(self, reference: PromptReference | str) -> PromptReference:
        """Return the current body-free reference (may reflect terminal status)."""

        with self._lock:
            self._ensure_open()
            prompt_ref, _ = self._reference_parts(reference)
            entry = self._entries.get(prompt_ref)
            if entry is None:
                raise PromptNotFoundError()
            return entry.reference

    def inspect_durable_surfaces(self) -> tuple[Mapping[str, Any], ...]:
        """Return every durable index/artifact projection for leak scanning.

        Projections contain only CID/reference metadata.  Capability tokens and
        plaintext bodies are never included.
        """

        with self._lock:
            surfaces: list[Mapping[str, Any]] = []
            for entry in self._entries.values():
                surfaces.append(
                    MappingProxyType(
                        {
                            "kind": "broker_entry",
                            "reference": entry.reference.to_dict(),
                            "capability_digest": entry.capability_digest,
                            "max_uses": entry.max_uses,
                            "uses": entry.uses,
                            "status": entry.status.value,
                            "body_resident": entry.body is not None
                            and len(entry.body) > 0,
                            "token_present": False,
                        }
                    )
                )
            if self._artifact_dir is not None:
                index_path = self._artifact_dir / ARTIFACT_INDEX_NAME
                if index_path.is_file():
                    try:
                        raw = index_path.read_text(encoding="utf-8")
                        payload = json.loads(raw)
                    except (OSError, UnicodeError, json.JSONDecodeError):
                        payload = {"unreadable": True, "path": str(index_path)}
                    surfaces.append(
                        MappingProxyType(
                            {
                                "kind": "artifact_index",
                                "path": str(index_path),
                                "payload": payload,
                            }
                        )
                    )
                blob_root = self._artifact_dir / ARTIFACT_BLOB_DIR
                if blob_root.is_dir():
                    for path in sorted(blob_root.iterdir()):
                        if not path.is_file():
                            continue
                        surfaces.append(
                            MappingProxyType(
                                {
                                    "kind": "encrypted_blob",
                                    "path": str(path),
                                    "byte_count": path.stat().st_size,
                                    "plaintext_absent_by_policy": True,
                                }
                            )
                        )
            return tuple(surfaces)

    def scan_for_secrets(
        self,
        canaries: Sequence[str],
        *,
        extra_surfaces: Sequence[str | bytes | Mapping[str, Any]] = (),
    ) -> tuple[str, ...]:
        """Return canary strings found on inspected durable surfaces."""

        if not canaries:
            return ()
        haystacks: list[str] = []
        for surface in self.inspect_durable_surfaces():
            haystacks.append(
                json.dumps(surface, sort_keys=True, default=str, separators=(",", ":"))
            )
            # Encrypted blob files must not contain plaintext canaries.
            if surface.get("kind") == "encrypted_blob":
                path = Path(str(surface["path"]))
                try:
                    blob = path.read_bytes()
                except OSError:
                    continue
                # Decode as latin-1 so arbitrary ciphertext is searchable as text.
                haystacks.append(blob.decode("latin-1", errors="ignore"))
        for extra in extra_surfaces:
            if isinstance(extra, Mapping):
                haystacks.append(
                    json.dumps(extra, sort_keys=True, default=str, separators=(",", ":"))
                )
            elif isinstance(extra, bytes):
                haystacks.append(extra.decode("utf-8", errors="ignore"))
            else:
                haystacks.append(str(extra))
        found: list[str] = []
        for canary in canaries:
            if not canary:
                continue
            if any(canary in hay for hay in haystacks):
                found.append(canary)
        return tuple(found)

    def close(self) -> None:
        """Zeroize resident plaintext and mark the broker closed.

        Active encrypted artifacts remain indexed so a later broker with the
        same master secret can recover them until natural expiry or release.
        Memory-only bodies are irrecoverable after close/restart; the durable
        index keeps their identity so :meth:`_load_index` reports
        ``missing_after_restart`` explicitly.
        """

        with self._lock:
            if self._closed:
                return
            for entry in list(self._entries.values()):
                _zeroize_bytearray(entry.body)
                entry.body = None
            self._persist_index()
            self._closed = True

    def __enter__(self) -> PromptBodyBroker:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        del exc_type, exc, tb
        self.close()

    # ---------------------------------------------------------------- private

    def _ensure_open(self) -> None:
        if self._closed:
            raise PromptBrokerError(
                "prompt broker is closed", reason_code="closed"
            )

    def _reference_parts(
        self, reference: PromptReference | str
    ) -> tuple[str, str]:
        if isinstance(reference, PromptReference):
            return reference.prompt_ref, reference.prompt_cid
        return (
            _require_text(
                reference,
                "prompt_ref",
                maximum=MAX_REFERENCE_BYTES,
                pattern=_REFERENCE_RE,
            ),
            "",
        )

    def _capability_token_value(self, capability: PromptCapability | str) -> str:
        if isinstance(capability, PromptCapability):
            return capability.token
        if not isinstance(capability, str) or not _TOKEN_RE.fullmatch(capability):
            raise PromptCapabilityError("capability token is malformed")
        return capability

    def _validate_access(
        self,
        entry: _LiveEntry,
        *,
        token: str,
        run_id: str,
        now_ms: int,
    ) -> None:
        if entry.status is not PromptBodyStatus.ACTIVE:
            if entry.status is PromptBodyStatus.EXPIRED:
                raise PromptExpiredError()
            if entry.status is PromptBodyStatus.MISSING_AFTER_RESTART:
                raise PromptNotFoundError(
                    "prompt body was not recoverable after restart "
                    f"({self.restart_behavior()['master_secret_source']})"
                )
            raise PromptNotFoundError(
                f"prompt body is {entry.status.value}"
            )
        if entry.reference.run_id != run_id:
            raise PromptCrossRunError()
        if not hmac.compare_digest(_token_digest(token), entry.capability_digest):
            # Wrong capability for this reference is a capability denial.  When
            # the token belongs to another live entry, surface cross-run denial
            # so callers can distinguish forged tokens from run isolation.
            for other in self._entries.values():
                if other is entry:
                    continue
                if hmac.compare_digest(
                    _token_digest(token), other.capability_digest
                ):
                    raise PromptCrossRunError(
                        "prompt capability is bound to a different run"
                    )
            raise PromptCapabilityError("capability token rejected")
        if now_ms >= entry.reference.expires_at_ms:
            self._force_release(entry, PromptBodyStatus.EXPIRED)
            raise PromptExpiredError()
        if entry.remaining_uses() < 1:
            self._force_release(entry, PromptBodyStatus.EXHAUSTED)
            raise PromptCapabilityError(
                "capability uses exhausted", reason_code="uses_exhausted"
            )

    def _materialize_body(self, entry: _LiveEntry) -> bytes:
        if entry.body is not None and len(entry.body) > 0:
            return bytes(entry.body)
        if (
            entry.reference.storage is PromptStorageKind.ENCRYPTED_ARTIFACT
            and entry.reference.artifact_ref
            and self._artifact_dir is not None
        ):
            payload = self._read_encrypted_artifact(
                entry.reference.prompt_ref, entry.reference.artifact_ref
            )
            entry.body = bytearray(payload)
            return payload
        raise PromptNotFoundError(
            "prompt body is not resident and no encrypted artifact is available"
        )

    def _force_release(self, entry: _LiveEntry, status: PromptBodyStatus) -> None:
        _zeroize_bytearray(entry.body)
        entry.body = None
        entry.status = status
        entry.reference = PromptReference(
            prompt_cid=entry.reference.prompt_cid,
            prompt_ref=entry.reference.prompt_ref,
            run_id=entry.reference.run_id,
            byte_count=entry.reference.byte_count,
            issued_at_ms=entry.reference.issued_at_ms,
            expires_at_ms=entry.reference.expires_at_ms,
            storage=entry.reference.storage,
            purpose=entry.reference.purpose,
            artifact_ref=entry.reference.artifact_ref,
            status=status,
        )
        if (
            entry.reference.artifact_ref
            and self._artifact_dir is not None
            and status
            in {
                PromptBodyStatus.RELEASED,
                PromptBodyStatus.EXPIRED,
                PromptBodyStatus.EXHAUSTED,
            }
        ):
            path = self._artifact_dir / entry.reference.artifact_ref
            try:
                if path.is_file():
                    # Best-effort overwrite then unlink.
                    size = path.stat().st_size
                    with path.open("r+b") as handle:
                        handle.write(b"\x00" * size)
                        handle.flush()
                        os.fsync(handle.fileno())
                    path.unlink()
            except OSError:
                pass
        self._persist_index()

    def _write_encrypted_artifact(
        self, prompt_ref: str, artifact_ref: str, payload: bytes
    ) -> None:
        assert self._artifact_dir is not None
        path = self._artifact_dir / artifact_ref
        path.parent.mkdir(parents=True, exist_ok=True)
        token = _fernet_from_master(self._master_secret, prompt_ref).encrypt(payload)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_bytes(token)
        os.replace(tmp, path)

    def _read_encrypted_artifact(self, prompt_ref: str, artifact_ref: str) -> bytes:
        assert self._artifact_dir is not None
        path = self._artifact_dir / artifact_ref
        try:
            token = path.read_bytes()
        except OSError as exc:
            raise PromptNotFoundError(
                "encrypted prompt artifact is missing"
            ) from exc
        try:
            return _fernet_from_master(self._master_secret, prompt_ref).decrypt(token)
        except InvalidToken as exc:
            raise PromptBrokerError(
                "encrypted prompt artifact could not be decrypted",
                reason_code="decrypt_failed",
            ) from exc

    def _index_payload(self) -> dict[str, Any]:
        entries: list[dict[str, Any]] = []
        for entry in self._entries.values():
            entries.append(
                {
                    "reference": entry.reference.to_dict(),
                    "capability_digest": entry.capability_digest,
                    "max_uses": entry.max_uses,
                    "uses": entry.uses,
                    "status": entry.status.value,
                    "purpose": entry.purpose,
                }
            )
        entries.sort(key=lambda item: item["reference"]["prompt_ref"])
        return {
            "schema": PROMPT_BROKER_SCHEMA,
            "requirement_id": PROMPT_BROKER_REQUIREMENT_ID,
            "entries": entries,
        }

    def _persist_index(self) -> None:
        if self._artifact_dir is None:
            return
        path = self._artifact_dir / ARTIFACT_INDEX_NAME
        payload = json.dumps(
            self._index_payload(), separators=(",", ":"), sort_keys=True
        )
        tmp = path.with_suffix(".tmp")
        tmp.write_text(payload + "\n", encoding="utf-8")
        os.replace(tmp, path)

    def _load_index(self) -> None:
        if self._artifact_dir is None:
            return
        path = self._artifact_dir / ARTIFACT_INDEX_NAME
        if not path.is_file():
            return
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            return
        if not isinstance(payload, Mapping):
            return
        raw_entries = payload.get("entries")
        if not isinstance(raw_entries, list):
            return
        for item in raw_entries:
            if not isinstance(item, Mapping):
                continue
            try:
                reference = PromptReference.from_dict(item["reference"])
                capability_digest = str(item["capability_digest"])
                max_uses = int(item["max_uses"])
                uses = int(item.get("uses", 0))
                status = PromptBodyStatus(str(item.get("status", "active")))
                purpose = str(item.get("purpose", reference.purpose))
            except (KeyError, TypeError, ValueError, PromptBrokerError):
                continue
            body: bytearray | None = None
            if status is PromptBodyStatus.ACTIVE:
                if reference.storage is PromptStorageKind.ENCRYPTED_ARTIFACT:
                    # Body is loaded lazily from ciphertext on resolve.
                    body = None
                else:
                    # Memory-only deposits cannot survive restart.
                    status = PromptBodyStatus.MISSING_AFTER_RESTART
                    reference = PromptReference(
                        prompt_cid=reference.prompt_cid,
                        prompt_ref=reference.prompt_ref,
                        run_id=reference.run_id,
                        byte_count=reference.byte_count,
                        issued_at_ms=reference.issued_at_ms,
                        expires_at_ms=reference.expires_at_ms,
                        storage=reference.storage,
                        purpose=reference.purpose,
                        artifact_ref=reference.artifact_ref,
                        status=status,
                    )
            self._entries[reference.prompt_ref] = _LiveEntry(
                reference=reference,
                capability_digest=capability_digest,
                max_uses=max_uses,
                uses=uses,
                body=body,
                status=status,
                purpose=purpose,
            )


__all__ = (
    "ARTIFACT_BLOB_DIR",
    "ARTIFACT_INDEX_NAME",
    "DEFAULT_MAX_PROMPT_BYTES",
    "DEFAULT_MAX_USES",
    "DEFAULT_TTL_MS",
    "MASTER_SECRET_ENV",
    "MAX_TTL_MS",
    "PROMPT_BROKER_REQUIREMENT_ID",
    "PROMPT_BROKER_SCHEMA",
    "PROMPT_CAPABILITY_SCHEMA",
    "PROMPT_REFERENCE_SCHEMA",
    "PromptBodyBroker",
    "PromptBodyStatus",
    "PromptBrokerBoundsError",
    "PromptBrokerError",
    "PromptCapability",
    "PromptCapabilityError",
    "PromptCrossRunError",
    "PromptExpiredError",
    "PromptNotFoundError",
    "PromptReference",
    "PromptStorageKind",
)
