"""FACP-026: Replace Accelerate pseudo-CID paths with canonical content identity.

Legacy ``MockIPFSMultiformats.get_cid`` returned raw SHA-256 hex and labeled it
a CID. This adapter owns the supported-path replacement:

* mint CIDv1 / lowercase base32 / ``sha2-256`` via in-tree ``cid_utils``;
* reject raw hex and truncated Qm-like identifiers as ``integrity.unchecked``;
* decode and recompute claimed CIDs against retained canonical bytes;
* fail one-bit content or identifier mutations with the expected integrity state.

No regex-only CID admission. No fabricated hex/Qm identities. Codecs stay in
the closed ``raw`` / ``dag-json`` profile already admitted by ``cid_utils``.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from enum import Enum
from typing import Any, Final, Iterable, Mapping, Optional, Sequence, Tuple

from ipfs_accelerate_py.utils import cid_utils

TASK_ID: Final[str] = "FACP-026"
GOAL_ID: Final[str] = "FACP-G220"
BUNDLE: Final[str] = "facp/migration/accelerate-cid"
EVIDENCE_ID: Final[str] = "facp/canonical-cid@1"
INTERFACE: Final[str] = "AccelerateContentIdentity@1"
SCHEMA: Final[str] = "ipfs_accelerate_py/assurance/content-identity@1"
FCA_VOCABULARY_SCHEMA: Final[str] = "facp/formal-claim-algebra-v1@1"

CID_VERSION: Final[int] = 1
CID_BASE: Final[str] = "base32"
MH_TYPE: Final[str] = "sha2-256"
ADMITTED_CODECS: Final[Tuple[str, ...]] = ("raw", "dag-json")
DEFAULT_CODEC: Final[str] = "raw"
DIGEST_SIZE: Final[int] = 32

# Inventoried Accelerate pseudo-CID construction sites (FACP-002).
INVENTORIED_PSEUDO_CID_SITES: Final[Tuple[Mapping[str, Any], ...]] = (
    {
        "defect_id": "defect:accelerate-mock-multiformats-raw-sha256-cid",
        "path": "external/ipfs_accelerate/ipfs_accelerate_py/ipfs_accelerate.py",
        "symbol": "MockIPFSMultiformats.get_cid",
        "start_line": 145,
        "end_line": 150,
        "unsafe_claim": "hashlib.sha256(...).hexdigest() labeled as a CID",
        "call_sites": (
            "ipfs_accelerate_py.queue",
            "ipfs_accelerate_py.fetch",
        ),
        "replacement": "CanonicalIPFSMultiformats.get_cid / mint_content_identity",
    },
)

_HEX64_RE = re.compile(r"^[0-9a-fA-F]{64}$")
_HEX_PREFIXED_RE = re.compile(r"^(?:sha256:|SHA256:|cid:|CID:)([0-9a-fA-F]{64})$")
# CIDv0 / base58btc Qm… forms (full or truncated). Not admitted on supported paths.
_QM_LIKE_RE = re.compile(r"^[Qm][1-9A-HJ-NP-Za-km-z]{0,50}$")


class Integrity(str, Enum):
    """FCA integrity dimension values (facp/formal-claim-algebra-v1@1)."""

    UNCHECKED = "unchecked"
    STRUCTURALLY_VALID = "structurally_valid"
    DIGEST_VALID = "digest_valid"
    SIGNATURE_VALID = "signature_valid"


class IdentityErrorCode(str, Enum):
    """Stable error codes for content-identity admission failures."""

    EMPTY_IDENTIFIER = "EMPTY_IDENTIFIER"
    PSEUDO_CID_RAW_HEX = "PSEUDO_CID_RAW_HEX"
    PSEUDO_CID_QM_FORM = "PSEUDO_CID_QM_FORM"
    PSEUDO_CID_TRUNCATED = "PSEUDO_CID_TRUNCATED"
    PSEUDO_CID_LABELED = "PSEUDO_CID_LABELED"
    CID_DECODE_FAILED = "CID_DECODE_FAILED"
    DIGEST_MISMATCH = "DIGEST_MISMATCH"
    BYTE_DOMAIN_INVALID = "BYTE_DOMAIN_INVALID"
    CODEC_NOT_ADMITTED = "CODEC_NOT_ADMITTED"


class ContentIdentityError(ValueError):
    """Hard failure when an identity claim cannot be admitted."""

    __test__ = False

    def __init__(
        self,
        message: str,
        *,
        code: IdentityErrorCode | str,
        integrity: Integrity | str = Integrity.UNCHECKED,
    ) -> None:
        super().__init__(message)
        self.code = IdentityErrorCode(code) if not isinstance(code, IdentityErrorCode) else code
        self.integrity = (
            Integrity(integrity) if not isinstance(integrity, Integrity) else integrity
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": False,
            "code": self.code.value,
            "message": str(self),
            "integrity": self.integrity.value,
        }


@dataclass(frozen=True)
class ContentIdentity:
    """Retained canonical bytes bound to a verified CIDv1 under the closed profile."""

    cid: str
    digest_hex: str
    canonical_bytes: bytes
    codec: str = DEFAULT_CODEC
    version: int = CID_VERSION
    base: str = CID_BASE
    mh_type: str = MH_TYPE
    integrity: str = Integrity.DIGEST_VALID.value
    profile: str = INTERFACE
    schema: str = SCHEMA

    def __post_init__(self) -> None:
        if type(self.canonical_bytes) is not bytes:
            raise ContentIdentityError(
                "canonical_bytes must be exact bytes",
                code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
            )
        if not self.canonical_bytes:
            raise ContentIdentityError(
                "canonical_bytes must be nonempty",
                code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
            )
        if self.codec not in ADMITTED_CODECS:
            raise ContentIdentityError(
                "codec is outside the admitted profile",
                code=IdentityErrorCode.CODEC_NOT_ADMITTED,
            )
        if (
            self.version != CID_VERSION
            or self.base != CID_BASE
            or self.mh_type != MH_TYPE
        ):
            raise ContentIdentityError(
                "only CIDv1/base32/sha2-256 is admitted",
                code=IdentityErrorCode.CID_DECODE_FAILED,
            )
        if len(self.digest_hex) != DIGEST_SIZE * 2 or self.digest_hex != self.digest_hex.lower():
            raise ContentIdentityError(
                "digest_hex must be 64 lowercase hex characters",
                code=IdentityErrorCode.DIGEST_MISMATCH,
            )
        actual = hashlib.sha256(self.canonical_bytes).hexdigest()
        if actual != self.digest_hex:
            raise ContentIdentityError(
                "digest_hex does not match sha2-256 of retained canonical bytes",
                code=IdentityErrorCode.DIGEST_MISMATCH,
            )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "interface": INTERFACE,
            "profile": self.profile,
            "cid": self.cid,
            "digest_hex": self.digest_hex,
            "codec": self.codec,
            "version": self.version,
            "base": self.base,
            "mh_type": self.mh_type,
            "integrity": self.integrity,
            "byte_length": len(self.canonical_bytes),
            "evidence_id": EVIDENCE_ID,
            "task_id": TASK_ID,
        }


@dataclass(frozen=True)
class IdentityVerification:
    """Result of verifying a claimed identifier against retained bytes."""

    ok: bool
    integrity: str
    code: str
    message: str
    cid: str | None = None
    digest_hex: str | None = None
    codec: str | None = None
    recomputed_cid: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "integrity": self.integrity,
            "code": self.code,
            "message": self.message,
            "cid": self.cid,
            "digest_hex": self.digest_hex,
            "codec": self.codec,
            "recomputed_cid": self.recomputed_cid,
            "evidence_id": EVIDENCE_ID,
            "task_id": TASK_ID,
        }


def _as_integrity(value: Integrity | str) -> str:
    return value.value if isinstance(value, Integrity) else str(value)


def is_raw_sha256_hex(value: str) -> bool:
    """Return True for bare or labeled 64-char SHA-256 hex digests."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if _HEX64_RE.fullmatch(text) is not None:
        return True
    return _HEX_PREFIXED_RE.fullmatch(text) is not None


def is_qm_like(value: str) -> bool:
    """Return True for CIDv0 / base58 Qm… forms including truncated prefixes."""

    if not isinstance(value, str):
        return False
    text = value.strip()
    if not text:
        return False
    # Full or truncated Qm/base58-looking tokens that are not CIDv1 base32.
    if text.startswith(("Qm", "qm")) and not text.startswith("b"):
        return True
    return _QM_LIKE_RE.fullmatch(text) is not None and not text.startswith("b")


def classify_pseudo_cid(value: Any) -> Optional[IdentityErrorCode]:
    """Classify a non-canonical identifier, or return None when not pseudo-shaped."""

    if not isinstance(value, str) or not value.strip():
        return IdentityErrorCode.EMPTY_IDENTIFIER
    text = value.strip()
    if text.startswith(("sha256:", "SHA256:", "cid:", "CID:")):
        return IdentityErrorCode.PSEUDO_CID_LABELED
    if is_raw_sha256_hex(text):
        return IdentityErrorCode.PSEUDO_CID_RAW_HEX
    if is_qm_like(text):
        # Truncated Qm forms are still Qm-like; distinguish short ones.
        if text.startswith(("Qm", "qm")) and len(text) < 46:
            return IdentityErrorCode.PSEUDO_CID_TRUNCATED
        return IdentityErrorCode.PSEUDO_CID_QM_FORM
    # Truncated CIDv1 base32 (starts with b but too short / non-canonical).
    if text.startswith("b") and len(text) < 16:
        return IdentityErrorCode.PSEUDO_CID_TRUNCATED
    return None


def reject_pseudo_cid(value: Any, *, field_name: str = "cid") -> str:
    """Reject raw hex / Qm-like / labeled pseudo identifiers before CID decode.

    Returns the stripped string when it is not an obvious pseudo form. Full
    profile validation still happens through :func:`validate_canonical_cid`.
    """

    code = classify_pseudo_cid(value)
    if code is IdentityErrorCode.EMPTY_IDENTIFIER:
        raise ContentIdentityError(
            "%s must be a nonempty string" % field_name,
            code=code,
            integrity=Integrity.UNCHECKED,
        )
    if code is not None:
        raise ContentIdentityError(
            "%s rejects pseudo-CID form (%s)" % (field_name, code.value),
            code=code,
            integrity=Integrity.UNCHECKED,
        )
    assert isinstance(value, str)
    return value.strip()


def validate_canonical_cid(
    value: Any,
    *,
    codecs: Iterable[str] = ADMITTED_CODECS,
) -> str:
    """Validate a CIDv1 through ``cid_utils`` after pseudo-CID rejection."""

    text = reject_pseudo_cid(value)
    allowed = tuple(codecs)
    for codec in allowed:
        if codec not in ADMITTED_CODECS:
            raise ContentIdentityError(
                "codec is outside the admitted profile",
                code=IdentityErrorCode.CODEC_NOT_ADMITTED,
            )
    try:
        return cid_utils.validate_cid(
            text,
            codecs=allowed,
            mh_type=MH_TYPE,
            version=CID_VERSION,
            base=CID_BASE,
        )
    except Exception as exc:
        raise ContentIdentityError(
            "CID is not a canonical admitted CIDv1: %s" % exc,
            code=IdentityErrorCode.CID_DECODE_FAILED,
            integrity=Integrity.UNCHECKED,
        ) from exc


def canonicalize_payload(
    data: Any,
    *,
    codec: Optional[str] = None,
) -> Tuple[bytes, str]:
    """Normalize Accelerate batch/item payloads to retained bytes and a codec.

    * ``bytes`` / ``bytearray`` / ``memoryview`` → ``raw``
    * ``str`` → UTF-8 bytes under ``raw`` (replaces legacy ``str(data).encode``)
    * mappings / sequences → strict DAG-JSON under ``dag-json``
    """

    if codec is not None and codec not in ADMITTED_CODECS:
        raise ContentIdentityError(
            "codec is outside the admitted profile",
            code=IdentityErrorCode.CODEC_NOT_ADMITTED,
        )

    if isinstance(data, (bytes, bytearray, memoryview)):
        payload = bytes(data)
        if not payload:
            raise ContentIdentityError(
                "canonical bytes must be nonempty",
                code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
            )
        # Explicit codec allows retained DAG-JSON bytes to be re-verified without
        # re-parsing; bare bytes default to the raw codec.
        chosen = codec or DEFAULT_CODEC
        if chosen not in ADMITTED_CODECS:
            raise ContentIdentityError(
                "codec is outside the admitted profile",
                code=IdentityErrorCode.CODEC_NOT_ADMITTED,
            )
        if chosen == "dag-json":
            try:
                parsed = json.loads(payload.decode("utf-8"))
                required = cid_utils.canonical_dag_json_bytes(parsed)
            except Exception as exc:
                raise ContentIdentityError(
                    "retained dag-json bytes are not canonical: %s" % exc,
                    code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
                ) from exc
            if required != payload:
                raise ContentIdentityError(
                    "retained dag-json bytes do not re-encode identically",
                    code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
                )
        return payload, chosen

    if isinstance(data, str):
        payload = data.encode("utf-8")
        if not payload:
            raise ContentIdentityError(
                "canonical bytes must be nonempty",
                code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
            )
        chosen = codec or DEFAULT_CODEC
        if chosen != "raw":
            raise ContentIdentityError(
                "string payloads require the raw codec",
                code=IdentityErrorCode.CODEC_NOT_ADMITTED,
            )
        return payload, chosen

    if isinstance(data, (dict, list, tuple)):
        chosen = codec or "dag-json"
        if chosen != "dag-json":
            raise ContentIdentityError(
                "structured payloads require the dag-json codec",
                code=IdentityErrorCode.CODEC_NOT_ADMITTED,
            )
        try:
            payload = cid_utils.canonical_dag_json_bytes(
                list(data) if isinstance(data, tuple) else data
            )
        except Exception as exc:
            raise ContentIdentityError(
                "failed to canonicalize structured payload: %s" % exc,
                code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
            ) from exc
        if not payload:
            raise ContentIdentityError(
                "canonical bytes must be nonempty",
                code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
            )
        return payload, chosen

    raise ContentIdentityError(
        "unsupported payload type for content identity: %s" % type(data).__name__,
        code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
    )


def mint_content_identity(
    data: Any,
    *,
    codec: Optional[str] = None,
) -> ContentIdentity:
    """Mint a verified content identity for Accelerate payload bytes."""

    canonical_bytes, chosen = canonicalize_payload(data, codec=codec)
    cid = cid_utils.cid_for_bytes(
        canonical_bytes,
        base=CID_BASE,
        codec=chosen,
        mh_type=MH_TYPE,
        version=CID_VERSION,
    )
    validated = validate_canonical_cid(cid, codecs=(chosen,))
    digest = cid_utils.digest_hex_from_cid(validated, codecs=(chosen,))
    identity = ContentIdentity(
        cid=validated,
        digest_hex=digest,
        canonical_bytes=canonical_bytes,
        codec=chosen,
        integrity=Integrity.DIGEST_VALID.value,
    )
    # Round-trip decode/recompute gate before returning.
    verified = verify_content_identity(validated, canonical_bytes, codec=chosen)
    if not verified.ok:
        raise ContentIdentityError(
            verified.message,
            code=verified.code,
            integrity=verified.integrity,
        )
    return identity


def get_cid(data: Any, *, codec: Optional[str] = None) -> str:
    """Drop-in replacement for ``MockIPFSMultiformats.get_cid``.

    Returns a canonical CIDv1 string. Never returns raw SHA-256 hex.
    """

    return mint_content_identity(data, codec=codec).cid


def verify_content_identity(
    claimed_cid: Any,
    data: Any,
    *,
    codec: Optional[str] = None,
) -> IdentityVerification:
    """Decode ``claimed_cid`` and recompute against retained payload bytes.

    Successful verification yields ``integrity.digest_valid``. Pseudo-CIDs,
    decode failures, and digest mismatches yield ``integrity.unchecked`` with a
    stable error code (identity claim rejected).
    """

    pseudo = classify_pseudo_cid(claimed_cid)
    if pseudo is not None:
        return IdentityVerification(
            ok=False,
            integrity=Integrity.UNCHECKED.value,
            code=pseudo.value,
            message="claimed identifier is a rejected pseudo-CID (%s)" % pseudo.value,
            cid=str(claimed_cid).strip() if isinstance(claimed_cid, str) else None,
        )

    try:
        canonical_bytes, chosen = canonicalize_payload(data, codec=codec)
    except ContentIdentityError as exc:
        return IdentityVerification(
            ok=False,
            integrity=_as_integrity(exc.integrity),
            code=exc.code.value,
            message=str(exc),
        )

    try:
        validated = validate_canonical_cid(claimed_cid, codecs=(chosen,) if codec else ADMITTED_CODECS)
    except ContentIdentityError as exc:
        return IdentityVerification(
            ok=False,
            integrity=_as_integrity(exc.integrity),
            code=exc.code.value,
            message=str(exc),
            cid=str(claimed_cid).strip() if isinstance(claimed_cid, str) else None,
        )

    # If codec was not forced, discover it from the validated CID.
    try:
        digest = cid_utils.digest_hex_from_cid(validated, codecs=ADMITTED_CODECS)
        # Prefer the codec that matches the retained payload.
        decoded_codecs: Sequence[str]
        if codec is not None:
            decoded_codecs = (chosen,)
        else:
            decoded_codecs = ADMITTED_CODECS
        # Recompute under the payload codec.
        recomputed = cid_utils.cid_for_bytes(
            canonical_bytes,
            base=CID_BASE,
            codec=chosen,
            mh_type=MH_TYPE,
            version=CID_VERSION,
        )
        recomputed = validate_canonical_cid(recomputed, codecs=(chosen,))
    except Exception as exc:
        return IdentityVerification(
            ok=False,
            integrity=Integrity.UNCHECKED.value,
            code=IdentityErrorCode.CID_DECODE_FAILED.value,
            message="CID decode/recompute failed: %s" % exc,
            cid=validated,
        )

    actual_digest = hashlib.sha256(canonical_bytes).hexdigest()
    if digest != actual_digest or validated != recomputed:
        return IdentityVerification(
            ok=False,
            integrity=Integrity.UNCHECKED.value,
            code=IdentityErrorCode.DIGEST_MISMATCH.value,
            message="claimed CID does not recompute against retained bytes",
            cid=validated,
            digest_hex=digest,
            codec=chosen,
            recomputed_cid=recomputed,
        )

    # Confirm multihash embedded in CID matches retained bytes.
    try:
        embedded = cid_utils.digest_hex_from_cid(validated, codecs=decoded_codecs)
    except Exception as exc:
        return IdentityVerification(
            ok=False,
            integrity=Integrity.UNCHECKED.value,
            code=IdentityErrorCode.CID_DECODE_FAILED.value,
            message="CID multihash extraction failed: %s" % exc,
            cid=validated,
            recomputed_cid=recomputed,
        )
    if embedded != actual_digest:
        return IdentityVerification(
            ok=False,
            integrity=Integrity.UNCHECKED.value,
            code=IdentityErrorCode.DIGEST_MISMATCH.value,
            message="CID multihash digest does not match sha2-256 of retained bytes",
            cid=validated,
            digest_hex=embedded,
            codec=chosen,
            recomputed_cid=recomputed,
        )

    return IdentityVerification(
        ok=True,
        integrity=Integrity.DIGEST_VALID.value,
        code=Integrity.DIGEST_VALID.value,
        message="canonical CID decodes and recomputes against retained bytes",
        cid=validated,
        digest_hex=actual_digest,
        codec=chosen,
        recomputed_cid=recomputed,
    )


def verify_or_raise(
    claimed_cid: Any,
    data: Any,
    *,
    codec: Optional[str] = None,
) -> ContentIdentity:
    """Verify a claimed CID or raise :class:`ContentIdentityError`."""

    result = verify_content_identity(claimed_cid, data, codec=codec)
    if not result.ok:
        raise ContentIdentityError(
            result.message,
            code=result.code,
            integrity=result.integrity,
        )
    canonical_bytes, chosen = canonicalize_payload(data, codec=codec or result.codec)
    assert result.cid is not None and result.digest_hex is not None
    return ContentIdentity(
        cid=result.cid,
        digest_hex=result.digest_hex,
        canonical_bytes=canonical_bytes,
        codec=chosen,
        integrity=Integrity.DIGEST_VALID.value,
    )


def legacy_pseudo_cid(data: Any) -> str:
    """Reproduce the inventoried MockIPFSMultiformats pseudo-CID (raw hex).

    Exists only so migration tests can prove the legacy form is rejected. Never
    use this on supported production paths.
    """

    return hashlib.sha256(str(data).encode("utf-8")).hexdigest()


class CanonicalIPFSMultiformats:
    """Drop-in replacement for Accelerate ``MockIPFSMultiformats``.

    ``get_cid`` returns a canonical CIDv1. Claims that present the legacy raw
    hex form fail :func:`verify_content_identity` with ``integrity.unchecked``.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        del args, kwargs

    def get_cid(self, data: Any) -> str:
        return get_cid(data)


def install_canonical_multiformats(owner: Any) -> CanonicalIPFSMultiformats:
    """Bind :class:`CanonicalIPFSMultiformats` onto an Accelerate-like owner.

    Migrates inventoried ``owner.ipfs_multiformats`` pseudo-CID construction
    without requiring a broad coordinator rewrite in this task.
    """

    adapter = CanonicalIPFSMultiformats()
    try:
        owner.ipfs_multiformats = adapter
    except Exception as exc:
        raise ContentIdentityError(
            "failed to install canonical multiformats adapter: %s" % exc,
            code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
        ) from exc
    resources = getattr(owner, "resources", None)
    if isinstance(resources, dict):
        resources["ipfs_multiformats"] = adapter
    return adapter


def flip_one_bit(data: bytes, *, bit_index: int = 0) -> bytes:
    """Return a copy of ``data`` with a single bit flipped (mutation oracle)."""

    if type(data) is not bytes or not data:
        raise ContentIdentityError(
            "flip_one_bit requires nonempty bytes",
            code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
        )
    if bit_index < 0 or bit_index >= len(data) * 8:
        raise ContentIdentityError(
            "bit_index out of range",
            code=IdentityErrorCode.BYTE_DOMAIN_INVALID,
        )
    byte_i, bit_i = divmod(bit_index, 8)
    mutable = bytearray(data)
    mutable[byte_i] ^= 1 << (7 - bit_i)
    return bytes(mutable)


def mutate_cid_one_bit(cid: str) -> str:
    """Flip one payload bit inside a canonical CIDv1 base32 string.

    Produces a different identifier string for mutation tests. The result is
    not required to remain decodable; verification must fail closed either way.
    """

    text = validate_canonical_cid(cid)
    # Multibase payload after the leading 'b'.
    chars = list(text)
    # Prefer mutating a base32 character that stays in alphabet when possible.
    alphabet = "abcdefghijklmnopqrstuvwxyz234567"
    for index in range(len(chars) - 1, 0, -1):
        ch = chars[index]
        if ch not in alphabet:
            continue
        pos = alphabet.index(ch)
        chars[index] = alphabet[(pos + 1) % len(alphabet)]
        mutated = "".join(chars)
        if mutated != text:
            return mutated
    # Fallback: append an extra base32 char (truncation/extension failure).
    return text + "a"


__all__ = [
    "ADMITTED_CODECS",
    "BUNDLE",
    "CID_BASE",
    "CID_VERSION",
    "CanonicalIPFSMultiformats",
    "ContentIdentity",
    "ContentIdentityError",
    "DEFAULT_CODEC",
    "DIGEST_SIZE",
    "EVIDENCE_ID",
    "FCA_VOCABULARY_SCHEMA",
    "GOAL_ID",
    "INTERFACE",
    "INVENTORIED_PSEUDO_CID_SITES",
    "IdentityErrorCode",
    "IdentityVerification",
    "Integrity",
    "MH_TYPE",
    "SCHEMA",
    "TASK_ID",
    "canonicalize_payload",
    "classify_pseudo_cid",
    "flip_one_bit",
    "get_cid",
    "install_canonical_multiformats",
    "is_qm_like",
    "is_raw_sha256_hex",
    "legacy_pseudo_cid",
    "mint_content_identity",
    "mutate_cid_one_bit",
    "reject_pseudo_cid",
    "validate_canonical_cid",
    "verify_content_identity",
    "verify_or_raise",
]
