"""Strict DAG-JSON / CIDv1 / multihash identity bridge for the agent supervisor.

This module wraps :mod:`ipfs_datasets_py.utils.cid_utils` so supervisor code can
mint and validate content addresses under a single frozen profile:

* CIDv1
* lowercase base32 multibase
* ``dag-json`` for structured objects and ``raw`` for exact bytes
* ``sha2-256`` multihashes with a full 32-byte digest

Existing supervisor identities (``content_identity`` CIDs and
``runtime-artifact:sha256:…`` / ``sha256:…`` digests) are preserved.  Typed
:class:`IdentityLink` records connect those local IDs to multiformats CIDs
without silently replacing either side.  Cross-package drift is fail-closed:
independent construction via ``cid_utils`` and the ``multiformats`` library
must agree, and double-hashing of an already-computed digest is rejected.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import date, datetime, time, timedelta
from enum import Enum
from pathlib import Path
from typing import Any, Final

try:
    # Preferred shared encoder when the datasets package is present.
    from ipfs_datasets_py.utils import cid_utils as _cid_utils
except ImportError:  # pragma: no cover - exercised under hermetic validation
    # Hermetic validation uses PYTHONNOUSERSITE and worktree submodule
    # placeholders (empty ``ipfs_datasets_py/`` dirs) that shadow the real
    # optional package.  Fall back to the in-tree helpers so entrypoint
    # contracts remain importable without user-site editable installs.
    from ipfs_accelerate_py.utils import cid_utils as _cid_utils


MULTIFORMATS_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/multiformats-identity@1"
)
IDENTITY_LINK_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/multiformats-identity-link@1"
)

CID_VERSION: Final = 1
CID_BASE: Final = "base32"
MH_TYPE: Final = "sha2-256"
DIGEST_SIZE: Final = 32
ALLOWED_CODECS: Final = frozenset({"raw", "dag-json"})

RUNTIME_ARTIFACT_PREFIX: Final = "runtime-artifact:sha256:"
PAYLOAD_DIGEST_PREFIX: Final = "sha256:"

# Wall-clock / expiry material must not participate in identity digests.
_TEMPORAL_IDENTITY_KEYS: Final = frozenset(
    {
        "timestamp",
        "timestamps",
        "created_at",
        "updated_at",
        "modified_at",
        "expires_at",
        "expired_at",
        "created_at_ms",
        "updated_at_ms",
        "expires_at_ms",
        "wall_time",
        "wall_clock",
        "now",
        "as_of",
        "observed_at",
        "issued_at",
    }
)

_SHA256_HEX_RE = re.compile(r"^[0-9a-f]{64}$")
_TEMPORAL_TYPES = (datetime, date, time, timedelta)


class MultiformatsIdentityError(ValueError):
    """A multiformats identity or link violates the frozen profile."""


class IdentityKind(str, Enum):
    """Closed vocabulary for typed dual-identity links."""

    CONTENT_IDENTITY = "content_identity"
    RUNTIME_ARTIFACT = "runtime_artifact"
    PAYLOAD_DIGEST = "payload_digest"
    RAW_BYTES = "raw_bytes"
    DAG_JSON = "dag_json"


@dataclass(frozen=True)
class IdentityLink:
    """Explicit dual mapping between a persisted local ID and a multiformats CID.

    ``local_id`` is never rewritten to the CID (or vice versa).  Callers that
    need either form must read the matching field; replacing persisted
    identities silently is out of contract for this type.
    """

    kind: str
    local_id: str
    cid: str
    codec: str
    version: int = CID_VERSION
    base: str = CID_BASE
    multihash_type: str = MH_TYPE
    digest_hex: str = ""
    digest_size: int = DIGEST_SIZE
    schema: str = IDENTITY_LINK_SCHEMA

    def __post_init__(self) -> None:
        if self.schema != IDENTITY_LINK_SCHEMA:
            raise MultiformatsIdentityError(
                "unsupported multiformats identity link schema"
            )
        try:
            IdentityKind(self.kind)
        except ValueError as exc:
            raise MultiformatsIdentityError(
                f"unsupported identity link kind: {self.kind!r}"
            ) from exc
        if not isinstance(self.local_id, str) or not self.local_id:
            raise MultiformatsIdentityError("local_id must be a nonempty string")
        if self.codec not in ALLOWED_CODECS:
            raise MultiformatsIdentityError(
                f"codec must be one of {sorted(ALLOWED_CODECS)}"
            )
        if (
            self.version != CID_VERSION
            or self.base != CID_BASE
            or self.multihash_type != MH_TYPE
            or self.digest_size != DIGEST_SIZE
        ):
            raise MultiformatsIdentityError(
                "identity link must use CIDv1/base32/sha2-256/32-byte digests"
            )
        if not _SHA256_HEX_RE.fullmatch(self.digest_hex):
            raise MultiformatsIdentityError(
                "digest_hex must be a 64-character lowercase sha2-256 hex string"
            )
        validated = validate_cid(self.cid, codecs=(self.codec,))
        if validated != self.cid:
            raise MultiformatsIdentityError(
                "identity link CID is not the validated canonical form"
            )
        actual_digest = digest_hex_from_cid(self.cid, codecs=(self.codec,))
        if actual_digest != self.digest_hex:
            raise MultiformatsIdentityError(
                "identity link digest_hex does not match the CID multihash"
            )
        # Local ID is an independent persisted handle; it must remain distinct
        # as a stored field even when it happens to equal the CID string
        # (content_identity already is a CIDv1).
        if self.local_id != self.cid and self.kind == IdentityKind.CONTENT_IDENTITY.value:
            # content_identity local IDs that are themselves CIDs must match.
            if self.local_id.startswith("b") and len(self.local_id) > 8:
                raise MultiformatsIdentityError(
                    "content_identity local_id disagrees with linked CID"
                )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "IdentityLink":
        if not isinstance(value, Mapping):
            raise MultiformatsIdentityError("identity link must be an object")
        allowed = {
            "schema",
            "kind",
            "local_id",
            "cid",
            "codec",
            "version",
            "base",
            "multihash_type",
            "digest_hex",
            "digest_size",
        }
        unknown = set(value).difference(allowed)
        if unknown:
            raise MultiformatsIdentityError(
                f"identity link contains unknown fields: {sorted(unknown)}"
            )
        return cls(
            schema=str(value.get("schema") or IDENTITY_LINK_SCHEMA),
            kind=str(value.get("kind") or ""),
            local_id=str(value.get("local_id") or ""),
            cid=str(value.get("cid") or ""),
            codec=str(value.get("codec") or ""),
            version=int(value.get("version", CID_VERSION)),
            base=str(value.get("base") or CID_BASE),
            multihash_type=str(value.get("multihash_type") or MH_TYPE),
            digest_hex=str(value.get("digest_hex") or ""),
            digest_size=int(value.get("digest_size", DIGEST_SIZE)),
        )


def _require_codec(codec: str) -> str:
    if codec not in ALLOWED_CODECS:
        raise MultiformatsIdentityError(
            f"codec must be one of {sorted(ALLOWED_CODECS)}; got {codec!r}"
        )
    return codec


def _is_temporal(value: Any) -> bool:
    return isinstance(value, _TEMPORAL_TYPES)


def _validate_dag_json_value(
    value: Any,
    *,
    path: str = "$",
    reject_temporal_keys: bool = False,
) -> None:
    """Require one unambiguous finite JSON/IPLD data-model value."""

    if _is_temporal(value):
        raise MultiformatsIdentityError(
            f"{path} must not contain timestamps in identity material: "
            f"{type(value).__name__}"
        )
    if value is None or type(value) in {str, bool, int}:
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise MultiformatsIdentityError(
                f"{path} is not JSON compliant: non-finite number"
            )
        return
    if type(value) is list:
        for index, item in enumerate(value):
            _validate_dag_json_value(
                item,
                path=f"{path}[{index}]",
                reject_temporal_keys=reject_temporal_keys,
            )
        return
    if type(value) is dict:
        for key, item in value.items():
            if type(key) is not str:
                raise MultiformatsIdentityError(
                    f"{path} contains a non-string DAG-JSON map key"
                )
            if reject_temporal_keys and key.casefold() in _TEMPORAL_IDENTITY_KEYS:
                raise MultiformatsIdentityError(
                    f"{path}.{key} introduces a timestamp into identity material"
                )
            _validate_dag_json_value(
                item,
                path=f"{path}.{key}",
                reject_temporal_keys=reject_temporal_keys,
            )
        return
    # Refuse default=repr / ad-hoc stringification of arbitrary objects.
    raise MultiformatsIdentityError(
        f"{path} is not JSON serializable as DAG-JSON: {type(value).__name__} "
        "(default=repr and non-JSON types are rejected)"
    )


def canonical_dag_json_bytes(
    obj: Any,
    *,
    for_identity: bool = False,
) -> bytes:
    """Serialize strict, deterministic, finite DAG-JSON bytes.

    Unlike legacy serializers that pass ``default=repr``, unsupported Python
    objects (including timestamps) fail closed.  When ``for_identity`` is true,
    known wall-clock field names are also rejected so identity digests stay
    reproducible.
    """

    _validate_dag_json_value(obj, reject_temporal_keys=for_identity)
    # Prefer the shared package encoder so bytes match cid_utils exactly, then
    # re-validate that its output is the unique sorted/compact form.
    encoded = _cid_utils.canonical_dag_json_bytes(obj)
    return require_canonical_dag_json_bytes(encoded)


def require_canonical_dag_json_bytes(data: bytes) -> bytes:
    """Accept only exact canonical (sorted-key, compact, finite) DAG-JSON bytes."""

    if type(data) is not bytes:
        raise MultiformatsIdentityError(
            "DAG-JSON identity input must be exact bytes"
        )
    try:
        text = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise MultiformatsIdentityError(
            "DAG-JSON bytes must be UTF-8"
        ) from exc
    try:
        parsed = json.loads(text, parse_constant=_reject_json_constant)
    except json.JSONDecodeError as exc:
        raise MultiformatsIdentityError(
            "DAG-JSON bytes are not valid JSON"
        ) from exc
    _validate_dag_json_value(parsed)
    # Reconstruct with the shared encoder — never default=repr.
    expected = _cid_utils.canonical_dag_json_bytes(parsed)
    if data != expected:
        raise MultiformatsIdentityError(
            "DAG-JSON bytes are not canonical (unsorted keys, non-compact "
            "separators, or non-normalized form)"
        )
    return data


def _reject_json_constant(name: str) -> None:
    raise MultiformatsIdentityError(
        f"JSON constant {name!r} is not allowed in DAG-JSON identity material"
    )


def _reject_ambiguous_raw_input(data: Any) -> bytes:
    """Require unambiguous exact bytes; refuse str/path/file-like/memoryview."""

    if type(data) is bytes:
        return data
    if isinstance(data, (str, Path)):
        raise MultiformatsIdentityError(
            "ambiguous raw/string/file input: pass exact bytes, not a path or str"
        )
    if isinstance(data, (bytearray, memoryview)):
        raise MultiformatsIdentityError(
            "ambiguous raw input: coerce to bytes explicitly before hashing"
        )
    # File-like objects
    if hasattr(data, "read") and not isinstance(data, (bytes, bytearray)):
        raise MultiformatsIdentityError(
            "ambiguous raw/file input: read the file to bytes before hashing"
        )
    raise MultiformatsIdentityError(
        f"raw CID input must be exact bytes; got {type(data).__name__}"
    )


def cid_for_bytes(
    data: bytes,
    *,
    base: str = CID_BASE,
    codec: str = "raw",
    mh_type: str = MH_TYPE,
    version: int = CID_VERSION,
) -> str:
    """Return a CIDv1 for exact bytes under the frozen multiformats profile."""

    payload = _reject_ambiguous_raw_input(data)
    _require_codec(codec)
    if base != CID_BASE or mh_type != MH_TYPE or version != CID_VERSION:
        raise MultiformatsIdentityError(
            "only CIDv1/base32/sha2-256 is supported by the identity bridge"
        )
    cid = _cid_utils.cid_for_bytes(
        payload,
        base=base,
        codec=codec,
        mh_type=mh_type,
        version=version,
    )
    return _cross_check_cid_for_bytes(payload, cid, codec=codec)


def cid_for_dag_json(
    obj: Any,
    *,
    base: str = CID_BASE,
    mh_type: str = MH_TYPE,
    version: int = CID_VERSION,
    for_identity: bool = False,
) -> str:
    """Return a CIDv1 for the canonical DAG-JSON encoding of ``obj``."""

    if base != CID_BASE or mh_type != MH_TYPE or version != CID_VERSION:
        raise MultiformatsIdentityError(
            "only CIDv1/base32/sha2-256 is supported by the identity bridge"
        )
    encoded = canonical_dag_json_bytes(obj, for_identity=for_identity)
    cid = _cid_utils.cid_for_bytes(
        encoded,
        base=base,
        codec="dag-json",
        mh_type=mh_type,
        version=version,
    )
    # Cross-check against the package helper and an independent multiformats build.
    package_cid = _cid_utils.cid_for_dag_json(
        obj, base=base, mh_type=mh_type, version=version
    )
    if package_cid != cid:
        raise MultiformatsIdentityError(
            "cross-package codec drift: cid_utils dag-json CID mismatch"
        )
    return _cross_check_cid_for_bytes(encoded, cid, codec="dag-json")


def validate_cid(
    value: Any,
    *,
    codecs: Iterable[str] = ("raw", "dag-json"),
    mh_type: str = MH_TYPE,
    version: int = CID_VERSION,
    base: str = CID_BASE,
) -> str:
    """Validate and return one canonical lowercase CIDv1 string."""

    if not isinstance(value, str) or not value:
        raise MultiformatsIdentityError("CID must be a nonempty lowercase string")
    if value != value.lower():
        raise MultiformatsIdentityError("CID must be canonical lowercase form")
    if mh_type != MH_TYPE or version != CID_VERSION or base != CID_BASE:
        raise MultiformatsIdentityError(
            "only CIDv1/base32/sha2-256 validation is supported"
        )
    allowed = tuple(codecs)
    if not allowed or any(item not in ALLOWED_CODECS for item in allowed):
        raise MultiformatsIdentityError(
            f"codecs must be a nonempty subset of {sorted(ALLOWED_CODECS)}"
        )
    # Reject obvious truncated / pseudo forms before multiformats decode.
    if len(value) < 16 or not value.startswith("b"):
        raise MultiformatsIdentityError(
            "CID is truncated or not canonical lowercase base32 CIDv1"
        )
    try:
        canonical = _cid_utils.validate_cid(
            value,
            codecs=allowed,
            mh_type=mh_type,
            version=version,
            base=base,
        )
    except (TypeError, ValueError) as exc:
        raise MultiformatsIdentityError(str(exc) or "CID validation failed") from exc
    if canonical != value:
        raise MultiformatsIdentityError("CID is not the validated canonical form")
    _validate_cid_with_multiformats(
        value, codecs=allowed, mh_type=mh_type, version=version, base=base
    )
    return value


def _validate_cid_with_multiformats(
    value: str,
    *,
    codecs: Iterable[str],
    mh_type: str,
    version: int,
    base: str,
) -> None:
    """Independent multiformats-library check for cross-package drift."""

    try:
        from multiformats import CID, multihash
    except ImportError as exc:  # pragma: no cover - environment must provide it
        raise MultiformatsIdentityError(
            "multiformats package is required for independent CID validation"
        ) from exc
    try:
        parsed = CID.decode(value)
    except Exception as exc:
        raise MultiformatsIdentityError("CID is not decodable") from exc
    expected_size = multihash.get(mh_type).max_digest_size
    if expected_size is None:
        expected_size = DIGEST_SIZE
    if (
        parsed.version != version
        or parsed.codec.name not in frozenset(codecs)
        or parsed.hashfun.name != mh_type
        or len(parsed.raw_digest) != expected_size
        or parsed.base.name != base
        or str(parsed) != value
    ):
        raise MultiformatsIdentityError(
            "CID must use the requested canonical version/base/codec/multihash"
        )


def _cross_check_cid_for_bytes(data: bytes, cid: str, *, codec: str) -> str:
    """Ensure cid_utils and multiformats agree on the same bytes/codec."""

    validated = validate_cid(cid, codecs=(codec,))
    try:
        from multiformats import CID, multihash
    except ImportError as exc:  # pragma: no cover
        raise MultiformatsIdentityError(
            "multiformats package is required for independent CID construction"
        ) from exc
    independent = str(
        CID(CID_BASE, CID_VERSION, codec, multihash.digest(data, MH_TYPE))
    )
    if independent != validated:
        raise MultiformatsIdentityError(
            "cross-package codec drift: multiformats CID disagrees with cid_utils"
        )
    return validated


def digest_hex_from_cid(
    value: str,
    *,
    codecs: Iterable[str] = ("raw", "dag-json"),
) -> str:
    """Return the lowercase sha2-256 hex digest carried by a validated CID."""

    canonical = validate_cid(value, codecs=codecs)
    from multiformats import CID

    parsed = CID.decode(canonical)
    digest = bytes(parsed.raw_digest)
    if len(digest) != DIGEST_SIZE:
        raise MultiformatsIdentityError("CID multihash digest size is not 32 bytes")
    return digest.hex()


def cid_from_sha256_digest(
    digest: bytes | str,
    *,
    codec: str = "raw",
    already_hashed: bool = True,
) -> str:
    """Build a CIDv1 by wrapping an existing sha2-256 digest (no double hashing).

    ``already_hashed`` must remain true.  Passing raw payload bytes with
    ``already_hashed=False`` is rejected so callers cannot accidentally treat
    a digest and a payload interchangeably.
    """

    _require_codec(codec)
    if not already_hashed:
        raise MultiformatsIdentityError(
            "refusing ambiguous digest/payload input: use cid_for_bytes for "
            "payloads or cid_from_sha256_digest(..., already_hashed=True) for digests"
        )
    if isinstance(digest, str):
        text = digest.strip().lower()
        if text.startswith(PAYLOAD_DIGEST_PREFIX):
            text = text[len(PAYLOAD_DIGEST_PREFIX) :]
        if not _SHA256_HEX_RE.fullmatch(text):
            raise MultiformatsIdentityError(
                "sha2-256 digest must be 64 lowercase hex characters"
            )
        digest_bytes = bytes.fromhex(text)
    elif type(digest) is bytes:
        digest_bytes = digest
    else:
        raise MultiformatsIdentityError(
            "digest must be bytes or hex string; refusing ambiguous input"
        )
    if len(digest_bytes) != DIGEST_SIZE:
        raise MultiformatsIdentityError(
            f"sha2-256 digest must be exactly {DIGEST_SIZE} bytes"
        )
    try:
        from multiformats import CID, multihash
    except ImportError as exc:  # pragma: no cover
        raise MultiformatsIdentityError(
            "multiformats package is required to wrap digests"
        ) from exc
    # multihash.wrap embeds the digest; multihash.digest would hash it again.
    wrapped = multihash.wrap(digest_bytes, MH_TYPE)
    double_hashed = multihash.digest(digest_bytes, MH_TYPE)
    if bytes(wrapped) == bytes(double_hashed):
        raise MultiformatsIdentityError(
            "malformed multihash: wrap and digest produced identical bytes"
        )
    probe = str(CID(CID_BASE, CID_VERSION, codec, wrapped))
    # The raw digest inside the wrapped multihash must be the caller's digest,
    # not a re-hash of it.
    if digest_hex_from_cid(probe, codecs=(codec,)) != digest_bytes.hex():
        raise MultiformatsIdentityError(
            "malformed multihash: wrapped digest does not round-trip"
        )
    return validate_cid(probe, codecs=(codec,))


def reject_double_hashed_multihash(data: bytes, claimed_cid: str) -> None:
    """Fail closed when ``claimed_cid`` hashes the digest of ``data`` twice."""

    payload = _reject_ambiguous_raw_input(data)
    direct = cid_for_bytes(payload, codec="raw")
    if claimed_cid == direct:
        return
    # If claimed equals CID(sha256(sha256(payload))), that is double hashing.
    inner = hashlib.sha256(payload).digest()
    double_cid = cid_for_bytes(inner, codec="raw")
    if claimed_cid == double_cid:
        raise MultiformatsIdentityError(
            "double hashing rejected: CID addresses sha256(payload) as raw bytes"
        )
    raise MultiformatsIdentityError(
        "claimed CID does not match payload under the frozen profile"
    )


def parse_runtime_artifact_id(artifact_id: str) -> str:
    """Return the 64-hex digest from a ``runtime-artifact:sha256:…`` ID."""

    if not isinstance(artifact_id, str) or not artifact_id:
        raise MultiformatsIdentityError(
            "runtime artifact id must be a nonempty string"
        )
    if not artifact_id.startswith(RUNTIME_ARTIFACT_PREFIX):
        raise MultiformatsIdentityError(
            "runtime artifact id must use the runtime-artifact:sha256: prefix"
        )
    digest_hex = artifact_id[len(RUNTIME_ARTIFACT_PREFIX) :].lower()
    if not _SHA256_HEX_RE.fullmatch(digest_hex):
        raise MultiformatsIdentityError(
            "runtime artifact id digest must be 64 lowercase hex characters"
        )
    if artifact_id[len(RUNTIME_ARTIFACT_PREFIX) :] != digest_hex:
        raise MultiformatsIdentityError(
            "runtime artifact id digest must be lowercase hex"
        )
    return digest_hex


def parse_payload_digest(payload_digest: str) -> str:
    """Return the 64-hex digest from a ``sha256:…`` payload digest."""

    if not isinstance(payload_digest, str) or not payload_digest:
        raise MultiformatsIdentityError("payload digest must be a nonempty string")
    if not payload_digest.startswith(PAYLOAD_DIGEST_PREFIX):
        raise MultiformatsIdentityError(
            "payload digest must use the sha256: prefix"
        )
    digest_hex = payload_digest[len(PAYLOAD_DIGEST_PREFIX) :].lower()
    if not _SHA256_HEX_RE.fullmatch(digest_hex):
        raise MultiformatsIdentityError(
            "payload digest must be 64 lowercase hex characters"
        )
    if payload_digest[len(PAYLOAD_DIGEST_PREFIX) :] != digest_hex:
        raise MultiformatsIdentityError("payload digest must be lowercase hex")
    return digest_hex


def _link(
    *,
    kind: IdentityKind,
    local_id: str,
    cid: str,
    codec: str,
) -> IdentityLink:
    digest_hex = digest_hex_from_cid(cid, codecs=(codec,))
    return IdentityLink(
        kind=kind.value,
        local_id=local_id,
        cid=cid,
        codec=codec,
        digest_hex=digest_hex,
    )


def link_content_identity(
    local_id: str,
    *,
    value: Any | None = None,
    expected_cid: str | None = None,
) -> IdentityLink:
    """Link a supervisor ``content_identity`` string to a validated dag-json CID.

    The persisted ``local_id`` is retained verbatim.  When ``value`` is
    supplied, the multiformats CID is derived independently and must agree
    with ``local_id`` when the local id is itself a CIDv1.  ``expected_cid``,
    if provided, must match the derived CID; it never replaces ``local_id``.
    """

    if not isinstance(local_id, str) or not local_id:
        raise MultiformatsIdentityError("content_identity local_id is required")

    derived_cid: str | None = None
    if value is not None:
        derived_cid = cid_for_dag_json(value, for_identity=True)
        # Prefer formal content_identity when available for cross-check.
        try:
            from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
                content_identity as formal_content_identity,
            )
        except Exception:  # pragma: no cover - package always present in-tree
            formal_content_identity = None  # type: ignore[assignment]
        if formal_content_identity is not None:
            try:
                formal_id = formal_content_identity(value)
            except Exception as exc:
                raise MultiformatsIdentityError(
                    f"formal content_identity rejected value: {exc}"
                ) from exc
            if formal_id != derived_cid:
                raise MultiformatsIdentityError(
                    "cross-package codec drift: content_identity disagrees "
                    "with multiformats dag-json CID"
                )

    if local_id.startswith("b") and len(local_id) >= 16:
        validated_local = validate_cid(local_id, codecs=("dag-json",))
        if derived_cid is not None and derived_cid != validated_local:
            raise MultiformatsIdentityError(
                "content_identity local_id does not match value-derived CID"
            )
        cid = validated_local
    elif derived_cid is not None:
        cid = derived_cid
    else:
        raise MultiformatsIdentityError(
            "content_identity link requires a CIDv1 local_id and/or value"
        )

    if expected_cid is not None:
        validated_expected = validate_cid(expected_cid, codecs=("dag-json",))
        if validated_expected != cid:
            raise MultiformatsIdentityError(
                "expected_cid does not match linked content_identity CID"
            )
        # expected_cid is advisory verification only — local_id stays put.

    return _link(
        kind=IdentityKind.CONTENT_IDENTITY,
        local_id=local_id,
        cid=cid,
        codec="dag-json",
    )


def link_runtime_artifact(
    artifact_id: str,
    *,
    payload_bytes: bytes | None = None,
    payload_digest: str | None = None,
    codec: str = "raw",
) -> IdentityLink:
    """Link a ``runtime-artifact:sha256:…`` ID to a CIDv1 without replacing it.

    The CID is formed by *wrapping* the existing sha2-256 digest (not by
    hashing the hex string or re-hashing the digest).  When payload bytes or
    a payload digest are provided they must independently confirm the same
    digest.
    """

    _require_codec(codec)
    digest_hex = parse_runtime_artifact_id(artifact_id)

    if payload_digest is not None:
        pd = parse_payload_digest(payload_digest)
        if pd != digest_hex:
            raise MultiformatsIdentityError(
                "payload_digest does not match runtime artifact id"
            )

    if payload_bytes is not None:
        raw = _reject_ambiguous_raw_input(payload_bytes)
        actual = hashlib.sha256(raw).hexdigest()
        if actual != digest_hex:
            raise MultiformatsIdentityError(
                "payload_bytes digest does not match runtime artifact id"
            )
        # Payload path: CID of the exact bytes must equal wrap(digest).
        payload_cid = cid_for_bytes(raw, codec=codec)
        wrapped_cid = cid_from_sha256_digest(digest_hex, codec=codec)
        if payload_cid != wrapped_cid:
            raise MultiformatsIdentityError(
                "cross-package codec drift: payload CID disagrees with wrapped digest"
            )
        cid = payload_cid
    else:
        cid = cid_from_sha256_digest(digest_hex, codec=codec)

    link = _link(
        kind=IdentityKind.RUNTIME_ARTIFACT,
        local_id=artifact_id,
        cid=cid,
        codec=codec,
    )
    if link.digest_hex != digest_hex:
        raise MultiformatsIdentityError(
            "runtime artifact digest disagrees with CID multihash"
        )
    # Never rewrite the persisted artifact id into the CID field.
    if link.local_id == link.cid:
        raise MultiformatsIdentityError(
            "runtime artifact local_id must remain distinct from the CID"
        )
    return link


def link_payload_digest(
    payload_digest: str,
    *,
    codec: str = "raw",
) -> IdentityLink:
    """Link a ``sha256:…`` payload digest to a CIDv1 (wrap, do not re-hash)."""

    _require_codec(codec)
    digest_hex = parse_payload_digest(payload_digest)
    cid = cid_from_sha256_digest(digest_hex, codec=codec)
    link = _link(
        kind=IdentityKind.PAYLOAD_DIGEST,
        local_id=payload_digest,
        cid=cid,
        codec=codec,
    )
    if link.local_id == link.cid:
        raise MultiformatsIdentityError(
            "payload digest local_id must remain distinct from the CID"
        )
    return link


def link_raw_bytes(data: bytes, *, local_id: str | None = None) -> IdentityLink:
    """Address exact bytes as raw CIDv1 and optionally retain a local label."""

    raw = _reject_ambiguous_raw_input(data)
    cid = cid_for_bytes(raw, codec="raw")
    retained = local_id if local_id is not None else cid
    if not isinstance(retained, str) or not retained:
        raise MultiformatsIdentityError("local_id must be a nonempty string")
    return _link(
        kind=IdentityKind.RAW_BYTES,
        local_id=retained,
        cid=cid,
        codec="raw",
    )


def link_dag_json(
    obj: Any,
    *,
    local_id: str | None = None,
    for_identity: bool = True,
) -> IdentityLink:
    """Address a structured value as dag-json CIDv1 with an optional local label."""

    cid = cid_for_dag_json(obj, for_identity=for_identity)
    retained = local_id if local_id is not None else cid
    if not isinstance(retained, str) or not retained:
        raise MultiformatsIdentityError("local_id must be a nonempty string")
    return _link(
        kind=IdentityKind.DAG_JSON,
        local_id=retained,
        cid=cid,
        codec="dag-json",
    )


def independent_round_trip_cid(
    data: bytes,
    *,
    codec: str = "raw",
) -> str:
    """Build a CID via cid_utils and multiformats independently; require equality."""

    _require_codec(codec)
    payload = _reject_ambiguous_raw_input(data)
    via_utils = _cid_utils.cid_for_bytes(
        payload,
        base=CID_BASE,
        codec=codec,
        mh_type=MH_TYPE,
        version=CID_VERSION,
    )
    from multiformats import CID, multihash

    via_multi = str(
        CID(CID_BASE, CID_VERSION, codec, multihash.digest(payload, MH_TYPE))
    )
    if via_utils != via_multi:
        raise MultiformatsIdentityError(
            "independent round trip failed: cid_utils and multiformats disagree"
        )
    return validate_cid(via_utils, codecs=(codec,))


def independent_round_trip_dag_json(obj: Any) -> str:
    """Round-trip a structured value through DAG-JSON encode/decode and dual CIDs."""

    encoded = canonical_dag_json_bytes(obj)
    parsed = json.loads(encoded.decode("utf-8"))
    reencoded = canonical_dag_json_bytes(parsed)
    if reencoded != encoded:
        raise MultiformatsIdentityError(
            "independent DAG-JSON round trip changed canonical bytes"
        )
    cid = independent_round_trip_cid(encoded, codec="dag-json")
    via_helper = cid_for_dag_json(obj)
    if cid != via_helper:
        raise MultiformatsIdentityError(
            "independent DAG-JSON round trip CID mismatch"
        )
    return cid


__all__ = [
    "ALLOWED_CODECS",
    "CID_BASE",
    "CID_VERSION",
    "DIGEST_SIZE",
    "IDENTITY_LINK_SCHEMA",
    "IdentityKind",
    "IdentityLink",
    "MH_TYPE",
    "MULTIFORMATS_IDENTITY_SCHEMA",
    "MultiformatsIdentityError",
    "PAYLOAD_DIGEST_PREFIX",
    "RUNTIME_ARTIFACT_PREFIX",
    "canonical_dag_json_bytes",
    "cid_for_bytes",
    "cid_for_dag_json",
    "cid_from_sha256_digest",
    "digest_hex_from_cid",
    "independent_round_trip_cid",
    "independent_round_trip_dag_json",
    "link_content_identity",
    "link_dag_json",
    "link_payload_digest",
    "link_raw_bytes",
    "link_runtime_artifact",
    "parse_payload_digest",
    "parse_runtime_artifact_id",
    "reject_double_hashed_multihash",
    "require_canonical_dag_json_bytes",
    "validate_cid",
]
