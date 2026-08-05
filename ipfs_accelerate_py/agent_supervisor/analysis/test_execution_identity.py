"""Core locator and execution identity compiler (PTR-010 / PTR-G020).

Interfaces:

* ``ContentIdentity@1`` — retained canonical DAG-JSON bytes plus a verified
  CIDv1 / lowercase base32 / ``dag-json`` / ``sha2-256`` address.
* ``TestExecutionIdentityCompiler@1`` — compile ``TestLocatorKey@1`` and
  ``TestExecutionKey@1`` into content-addressed artifacts.

Authority doctrine (fail-closed):

* A CID only identifies bytes; it never authorizes reuse by itself.
* Locator identities narrow candidate retrieval; execution identities bind the
  exact reusable context (forest, source/AST roots, fixtures, policy, …).
* Missing multiformats / CID support returns an explicit **non-reusable**
  artifact and never labels a fallback digest or kit pseudo-hash as a CID.
* Retained canonical bytes must decode and rehash to the stored CID.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, ClassVar, Dict, Final, Optional

from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
    EligibilityClass,
    ReuseReasonCode,
    TEST_EXECUTION_CONTRACT_VERSION,
    TEST_EXECUTION_KEY_INTERFACE,
    TEST_LOCATOR_KEY_INTERFACE,
    TestExecutionContractError,
    TestExecutionKey,
    TestLocatorKey,
)


# ---------------------------------------------------------------------------
# Schema / interface constants
# ---------------------------------------------------------------------------

CONTENT_IDENTITY_INTERFACE: Final = "ContentIdentity@1"
TEST_EXECUTION_IDENTITY_COMPILER_INTERFACE: Final = (
    "TestExecutionIdentityCompiler@1"
)

CONTENT_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/content-identity@1"
)
TEST_EXECUTION_IDENTITY_COMPILER_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/test-execution-identity-compiler@1"
)
COMPILED_LOCATOR_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/compiled-test-locator@1"
)
COMPILED_EXECUTION_KEY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/compiled-test-execution-key@1"
)

# Frozen CID profile (matches multiformats_identity / plan §6.1).
CID_VERSION: Final = 1
CID_BASE: Final = "base32"
CID_CODEC: Final = "dag-json"
MH_TYPE: Final = "sha2-256"
DIGEST_SIZE: Final = 32

# Explicit non-reusable reason codes retained on artifacts (string form of enum).
REASON_CID_PROVIDER_UNAVAILABLE: Final = (
    ReuseReasonCode.CID_PROVIDER_UNAVAILABLE.value
)
REASON_NON_REUSABLE: Final = ReuseReasonCode.NON_REUSABLE.value
REASON_MALFORMED_ARTIFACT: Final = ReuseReasonCode.MALFORMED_ARTIFACT.value
REASON_UNSUPPORTED: Final = ReuseReasonCode.UNSUPPORTED.value

class CidSupportStatus(str, Enum):
    """Cold probe result for the multiformats content-identity bridge."""

    AVAILABLE = "available"
    MISSING = "missing"
    INCOMPATIBLE = "incompatible"
    UNKNOWN = "unknown"


class TestExecutionIdentityError(ValueError):
    """Raised when identity compilation fails hard (not a non-reusable soft path)."""

    # Not a pytest test class.
    __test__ = False


# ---------------------------------------------------------------------------
# Lazy multiformats bridge access (never invent pseudo-CIDs)
# ---------------------------------------------------------------------------


class _LocalContentIdentityBridge:
    """Hermetic CIDv1/base32/dag-json/sha2-256 bridge.

    Prefer the shared formal contract encoder (same bytes as
    ``TestLocatorKey.content_id`` / ``TestExecutionKey.content_id``).  When the
    ``multiformats`` package is installed, CIDs are cross-checked against an
    independent construction.  When the supervisor ``multiformats_identity``
    module (and its ``ipfs_datasets_py.utils.cid_utils`` dependency) is
    unavailable, this bridge still provides a full reusable profile so identity
    compilation does not soft-fail solely because of that optional import path.
    """

    def canonical_dag_json_bytes(
        self, obj: Any, *, for_identity: bool = False
    ) -> bytes:
        del for_identity
        return _formal_canonical_json_bytes(obj)

    def require_canonical_dag_json_bytes(self, data: bytes) -> bytes:
        if type(data) is not bytes:
            raise TestExecutionIdentityError(
                "DAG-JSON identity input must be exact bytes"
            )
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise TestExecutionIdentityError("DAG-JSON bytes must be UTF-8") from exc
        try:
            parsed = json.loads(text, parse_constant=_reject_json_constant)
        except json.JSONDecodeError as exc:
            raise TestExecutionIdentityError(
                "DAG-JSON bytes are not valid JSON"
            ) from exc
        expected = _formal_canonical_json_bytes(parsed)
        if data != expected:
            raise TestExecutionIdentityError(
                "DAG-JSON bytes are not canonical (unsorted keys, non-compact "
                "separators, or non-normalized form)"
            )
        return data

    def cid_for_bytes(
        self,
        data: bytes,
        *,
        base: str = CID_BASE,
        codec: str = "raw",
        mh_type: str = MH_TYPE,
        version: int = CID_VERSION,
    ) -> str:
        if type(data) is not bytes:
            raise TestExecutionIdentityError("cid_for_bytes requires exact bytes")
        if base != CID_BASE or mh_type != MH_TYPE or version != CID_VERSION:
            raise TestExecutionIdentityError(
                "only CIDv1/base32/sha2-256 is admitted for ContentIdentity@1"
            )
        if codec not in (CID_CODEC, "raw"):
            raise TestExecutionIdentityError(
                "only dag-json or raw codec is admitted for ContentIdentity@1"
            )
        return _cidv1_for_payload(data, codec=codec)

    def cid_for_dag_json(
        self,
        obj: Any,
        *,
        base: str = CID_BASE,
        mh_type: str = MH_TYPE,
        version: int = CID_VERSION,
        for_identity: bool = False,
    ) -> str:
        del for_identity
        if base != CID_BASE or mh_type != MH_TYPE or version != CID_VERSION:
            raise TestExecutionIdentityError(
                "only CIDv1/base32/sha2-256 is admitted for ContentIdentity@1"
            )
        encoded = self.canonical_dag_json_bytes(obj, for_identity=True)
        return self.cid_for_bytes(
            encoded,
            base=base,
            codec=CID_CODEC,
            mh_type=mh_type,
            version=version,
        )

    def validate_cid(
        self,
        value: Any,
        *,
        codecs: Any = ("raw", "dag-json"),
    ) -> str:
        if not isinstance(value, str) or not value:
            raise TestExecutionIdentityError("CID must be a nonempty lowercase string")
        if value != value.lower():
            raise TestExecutionIdentityError("CID must be canonical lowercase form")
        if len(value) < 16 or not value.startswith("b"):
            raise TestExecutionIdentityError(
                "CID is truncated or not a CIDv1 base32 address"
            )
        allowed = tuple(codecs) if codecs is not None else (CID_CODEC, "raw")
        try:
            from multiformats import CID

            parsed = CID.decode(value)
        except ImportError:
            # Pure structural admission: formal / multiformats-compatible base32
            # CIDv1 strings only. Digest extraction still rehashes retained bytes
            # at ContentIdentity construction time.
            return value
        except Exception as exc:
            raise TestExecutionIdentityError("CID failed to decode") from exc
        if parsed.version != CID_VERSION:
            raise TestExecutionIdentityError("CID version must be 1")
        if parsed.base.name != CID_BASE:
            raise TestExecutionIdentityError("CID base must be base32")
        if parsed.hashfun.name != MH_TYPE:
            raise TestExecutionIdentityError("CID multihash must be sha2-256")
        if len(parsed.raw_digest) != DIGEST_SIZE:
            raise TestExecutionIdentityError("CID multihash digest size is not 32 bytes")
        codec_name = parsed.codec.name
        if codec_name not in allowed:
            raise TestExecutionIdentityError(
                "CID codec %r is not in allowed set %s" % (codec_name, allowed)
            )
        # Decode success under the frozen profile implies canonical form for the
        # multiformats versions we support (lowercase base32 CIDv1).
        return value

    def digest_hex_from_cid(
        self,
        value: str,
        *,
        codecs: Any = ("raw", "dag-json"),
    ) -> str:
        canonical = self.validate_cid(value, codecs=codecs)
        try:
            from multiformats import CID

            parsed = CID.decode(canonical)
            digest = bytes(parsed.raw_digest)
            if len(digest) != DIGEST_SIZE:
                raise TestExecutionIdentityError(
                    "CID multihash digest size is not 32 bytes"
                )
            return digest.hex()
        except ImportError:
            # Decode multihash from the CIDv1 binary form produced by formal
            # content_identity: 0x01 | dag-json varint | 0x12 0x20 | digest.
            import base64

            padded = canonical[1:] + ("=" * ((8 - len(canonical[1:]) % 8) % 8))
            try:
                raw = base64.b32decode(padded.upper())
            except Exception as exc:
                raise TestExecutionIdentityError(
                    "CID could not be base32-decoded for digest extraction"
                ) from exc
            # Find sha2-256 multihash header 0x12 0x20.
            idx = raw.find(b"\x12\x20")
            if idx < 0 or idx + 2 + DIGEST_SIZE > len(raw):
                raise TestExecutionIdentityError(
                    "CID does not carry a sha2-256 multihash digest"
                )
            return raw[idx + 2 : idx + 2 + DIGEST_SIZE].hex()


def _formal_canonical_json_bytes(value: Any) -> bytes:
    """Encode with the shared formal DAG-JSON profile (fail-closed)."""

    try:
        from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
            canonical_json_bytes as formal_canonical_json_bytes,
        )
    except ImportError as exc:  # pragma: no cover - in-tree dependency
        raise TestExecutionIdentityError(
            "formal verification contracts unavailable for canonicalization"
        ) from exc
    try:
        return formal_canonical_json_bytes(value)
    except Exception as exc:
        raise TestExecutionIdentityError(
            "failed to canonicalize structured value: %s" % exc
        ) from exc


def _cidv1_for_payload(payload: bytes, *, codec: str) -> str:
    """Mint CIDv1/base32 for exact payload bytes under ``codec``."""

    # Prefer independent multiformats construction when the package is present.
    try:
        from multiformats import CID, multihash

        return str(
            CID(CID_BASE, CID_VERSION, codec, multihash.digest(payload, MH_TYPE))
        )
    except ImportError:
        pass

    # Pure formal construction (matches formal_verification_contracts.content_identity).
    import base64

    digest = hashlib.sha256(payload).digest()
    if codec == CID_CODEC:
        # CIDv1 + dag-json (0x0129 varint as 0xa9 0x02) + sha2-256 multihash.
        raw = b"\x01\xa9\x02\x12\x20" + digest
    elif codec == "raw":
        # CIDv1 + raw (0x55) + sha2-256 multihash.
        raw = b"\x01\x55\x12\x20" + digest
    else:
        raise TestExecutionIdentityError(
            "unsupported codec for local CID minting: %s" % codec
        )
    return "b" + base64.b32encode(raw).decode("ascii").rstrip("=").lower()


_LOCAL_BRIDGE: Final = _LocalContentIdentityBridge()


def _import_multiformats_bridge() -> Any:
    """Return a CID identity bridge (supervisor module or hermetic local fallback).

    The supervisor ``multiformats_identity`` module depends on
    ``ipfs_datasets_py.utils.cid_utils``.  Under some validation PYTHONPATH
    orderings an empty ``ipfs_datasets_py`` namespace can shadow that package and
    break imports.  Fall back to the hermetic local bridge so ContentIdentity
    remains usable without inventing pseudo-CIDs.
    """

    try:
        from ipfs_accelerate_py.agent_supervisor import multiformats_identity as bridge

        # Cold probe: ensure the module is not only importable but also able to
        # mint under the frozen profile (cid_utils may be broken after import).
        bridge.cid_for_dag_json({}, for_identity=True)
        return bridge
    except Exception:
        return _LOCAL_BRIDGE


def probe_cid_support(
    *,
    bridge_import: Optional[Callable[[], Any]] = None,
) -> CidSupportStatus:
    """Probe whether strict CIDv1/base32/dag-json/sha2-256 minting is available.

    Cold, deterministic, and free of network or cache side effects. Distinguishes
    missing imports from incompatible probe results without installing packages
    or starting daemons.
    """

    importer = bridge_import or _import_multiformats_bridge
    try:
        bridge = importer()
    except ImportError:
        return CidSupportStatus.MISSING
    except Exception:
        return CidSupportStatus.UNKNOWN

    try:
        # Minimal known vector: empty object under the frozen profile.
        encoded = bridge.canonical_dag_json_bytes({}, for_identity=True)
        cid = bridge.cid_for_dag_json({}, for_identity=True)
        validated = bridge.validate_cid(cid, codecs=("dag-json",))
        if validated != cid:
            return CidSupportStatus.INCOMPATIBLE
        digest = bridge.digest_hex_from_cid(cid, codecs=("dag-json",))
        if len(digest) != DIGEST_SIZE * 2:
            return CidSupportStatus.INCOMPATIBLE
        if hashlib.sha256(encoded).hexdigest() != digest:
            return CidSupportStatus.INCOMPATIBLE
        # Independent multiformats library check when present.
        try:
            from multiformats import CID, multihash

            independent = str(
                CID(CID_BASE, CID_VERSION, CID_CODEC, multihash.digest(encoded, MH_TYPE))
            )
            if independent != cid:
                return CidSupportStatus.INCOMPATIBLE
        except ImportError:
            # Bridge itself may still work via formal / local construction; treat
            # as available if the bridge probe passed. Cross-package vectors are
            # PTR-012.
            pass
        return CidSupportStatus.AVAILABLE
    except Exception:
        return CidSupportStatus.INCOMPATIBLE


def cid_support_available(
    *,
    bridge_import: Optional[Callable[[], Any]] = None,
) -> bool:
    """Return True only when the frozen CID profile can mint and verify."""

    return probe_cid_support(bridge_import=bridge_import) is CidSupportStatus.AVAILABLE


def reject_pseudo_cid(value: str, *, field_name: str = "cid") -> str:
    """Reject obvious non-profile identifiers that must not be labeled as CIDs.

    Real CIDv1 base32 values are validated through the multiformats bridge
    separately. This helper only blocks known pseudo / kit forms before any
    authority boundary labels them as CIDs.
    """

    if not isinstance(value, str) or not value.strip():
        raise TestExecutionIdentityError("%s must be a nonempty string" % field_name)
    text = value.strip()
    # Admitted profile CIDs are lowercase base32 CIDv1 starting with "b" and
    # long enough to carry version/codec/multihash. Everything else is rejected
    # when callers insist on a "CID" field for retained identities.
    if (
        text.startswith("cid:")
        or text.startswith("sha256:")
        or text.startswith("runtime-artifact:")
        or text.startswith("CID:")
        or text.startswith("SHA256:")
    ):
        raise TestExecutionIdentityError(
            "%s must not use a pseudo-hash or kit identity label" % field_name
        )
    if text.startswith("Qm") or text.startswith("qm"):
        raise TestExecutionIdentityError(
            "%s rejects CIDv0 / base58 pseudo forms" % field_name
        )
    if text != text.lower():
        raise TestExecutionIdentityError(
            "%s must be canonical lowercase CIDv1 base32" % field_name
        )
    if len(text) < 16 or not text.startswith("b"):
        raise TestExecutionIdentityError(
            "%s is truncated or not a CIDv1 base32 address" % field_name
        )
    return text


# ---------------------------------------------------------------------------
# ContentIdentity@1
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ContentIdentity:
    """Retained canonical bytes bound to a verified CIDv1 under the frozen profile.

    Construction is closed: use :func:`mint_content_identity` or
    :func:`content_identity_from_retained_bytes`.  Callers never supply a CID
    string without the matching retained bytes.
    """

    # Not a pytest test class.
    __test__: ClassVar[bool] = False

    SCHEMA: ClassVar[str] = CONTENT_IDENTITY_SCHEMA

    cid: str
    digest_hex: str
    canonical_bytes: bytes
    codec: str = CID_CODEC
    version: int = CID_VERSION
    base: str = CID_BASE
    mh_type: str = MH_TYPE
    profile: str = CONTENT_IDENTITY_INTERFACE

    def __post_init__(self) -> None:
        if type(self.canonical_bytes) is not bytes:
            raise TestExecutionIdentityError(
                "canonical_bytes must be exact bytes (not str or memoryview)"
            )
        if not self.canonical_bytes:
            raise TestExecutionIdentityError("canonical_bytes must be nonempty")
        if self.codec != CID_CODEC:
            raise TestExecutionIdentityError(
                "only dag-json codec is admitted for ContentIdentity@1"
            )
        if self.version != CID_VERSION or self.base != CID_BASE or self.mh_type != MH_TYPE:
            raise TestExecutionIdentityError(
                "only CIDv1/base32/sha2-256 is admitted for ContentIdentity@1"
            )
        if self.profile != CONTENT_IDENTITY_INTERFACE:
            raise TestExecutionIdentityError(
                "content identity profile must be ContentIdentity@1"
            )
        if not isinstance(self.cid, str) or not self.cid:
            raise TestExecutionIdentityError("cid is required")
        if not isinstance(self.digest_hex, str) or len(self.digest_hex) != DIGEST_SIZE * 2:
            raise TestExecutionIdentityError(
                "digest_hex must be 64 lowercase hex characters"
            )
        if self.digest_hex != self.digest_hex.lower() or any(
            ch not in "0123456789abcdef" for ch in self.digest_hex
        ):
            raise TestExecutionIdentityError(
                "digest_hex must be lowercase hex"
            )
        # Structural self-check against retained bytes (hash only; full CID
        # validation is performed at mint time).
        actual = hashlib.sha256(self.canonical_bytes).hexdigest()
        if actual != self.digest_hex:
            raise TestExecutionIdentityError(
                "digest_hex does not match sha2-256 of retained canonical bytes"
            )

    @property
    def interface(self) -> str:
        return CONTENT_IDENTITY_INTERFACE

    @property
    def schema(self) -> str:
        return self.SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        """Public projection (does not re-embed full retained bytes by default)."""

        return {
            "schema": self.SCHEMA,
            "interface": CONTENT_IDENTITY_INTERFACE,
            "profile": self.profile,
            "cid": self.cid,
            "digest_hex": self.digest_hex,
            "codec": self.codec,
            "version": self.version,
            "base": self.base,
            "mh_type": self.mh_type,
            "byte_length": len(self.canonical_bytes),
        }

    def rehash_cid(
        self,
        *,
        bridge_import: Optional[Callable[[], Any]] = None,
    ) -> str:
        """Recompute the CIDv1 from retained bytes; must equal ``self.cid``."""

        importer = bridge_import or _import_multiformats_bridge
        bridge = importer()
        recomputed = bridge.cid_for_bytes(
            self.canonical_bytes,
            base=self.base,
            codec=self.codec,
            mh_type=self.mh_type,
            version=self.version,
        )
        if recomputed != self.cid:
            raise TestExecutionIdentityError(
                "retained canonical bytes rehash to a different CID"
            )
        return recomputed

    def verify(
        self,
        *,
        bridge_import: Optional[Callable[[], Any]] = None,
    ) -> "ContentIdentity":
        """Decode retained bytes, validate the CID, and confirm multihash match."""

        importer = bridge_import or _import_multiformats_bridge
        bridge = importer()
        # Canonical form of retained bytes (sorted keys, compact separators).
        bridge.require_canonical_dag_json_bytes(self.canonical_bytes)
        validated = bridge.validate_cid(self.cid, codecs=(self.codec,))
        if validated != self.cid:
            raise TestExecutionIdentityError("CID is not the validated canonical form")
        digest = bridge.digest_hex_from_cid(self.cid, codecs=(self.codec,))
        if digest != self.digest_hex:
            raise TestExecutionIdentityError(
                "CID multihash digest does not match retained digest_hex"
            )
        actual = hashlib.sha256(self.canonical_bytes).hexdigest()
        if actual != digest:
            raise TestExecutionIdentityError(
                "CID multihash digest does not match sha2-256 of retained bytes"
            )
        # Round-trip JSON decode → re-encode must preserve exact bytes.
        try:
            parsed = json.loads(
                self.canonical_bytes.decode("utf-8"),
                parse_constant=_reject_json_constant,
            )
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise TestExecutionIdentityError(
                "retained canonical bytes are not valid UTF-8 JSON"
            ) from exc
        reencoded = bridge.canonical_dag_json_bytes(parsed, for_identity=True)
        if reencoded != self.canonical_bytes:
            raise TestExecutionIdentityError(
                "retained canonical bytes do not re-encode identically"
            )
        self.rehash_cid(bridge_import=importer)
        return self


def _reject_json_constant(name: str) -> None:
    raise TestExecutionIdentityError(
        "JSON constant %r is not allowed in retained DAG-JSON identity material"
        % name
    )


def _structured_canonical_bytes(value: Any) -> bytes:
    """Encode a structured value to the shared formal DAG-JSON profile.

    ``str`` Enum members (used by PTR contracts) survive ``json.dumps`` but are
    rejected by the multiformats bridge's pre-validation.  Routing through the
    formal encoder first yields exact bytes that both ``content_identity`` and
    multiformats agree on, without labeling any fallback digest as a CID.
    """

    # CanonicalContract (and siblings) already expose profile-stable bytes.
    canonical_bytes_fn = getattr(value, "canonical_bytes", None)
    if callable(canonical_bytes_fn) and not isinstance(value, (dict, list, tuple)):
        try:
            encoded = canonical_bytes_fn()
        except Exception as exc:
            raise TestExecutionIdentityError(
                "contract canonical_bytes failed: %s" % exc
            ) from exc
        if type(encoded) is not bytes or not encoded:
            raise TestExecutionIdentityError(
                "contract canonical_bytes must return nonempty bytes"
            )
        return encoded

    try:
        from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
            canonical_json_bytes as formal_canonical_json_bytes,
        )
    except ImportError as exc:  # pragma: no cover - in-tree dependency
        raise TestExecutionIdentityError(
            "formal verification contracts unavailable for canonicalization"
        ) from exc

    try:
        return formal_canonical_json_bytes(value)
    except Exception as exc:
        raise TestExecutionIdentityError(
            "failed to canonicalize structured value: %s" % exc
        ) from exc


def mint_content_identity(
    value: Any,
    *,
    bridge_import: Optional[Callable[[], Any]] = None,
    for_identity: bool = True,
) -> ContentIdentity:
    """Mint a :class:`ContentIdentity` for a structured DAG-JSON value.

    Uses the supervisor multiformats bridge (``cid_utils`` + independent
    multiformats checks). Never falls back to a pseudo-hash labeled as a CID.
    """

    del for_identity  # retained for API stability; formal profile is always used
    importer = bridge_import or _import_multiformats_bridge
    try:
        bridge = importer()
    except ImportError as exc:
        raise TestExecutionIdentityError(
            "multiformats content-identity bridge is unavailable"
        ) from exc

    try:
        encoded = _structured_canonical_bytes(value)
        # Multiformats path: require exact canonical form, then mint raw-on-bytes
        # as dag-json (same as cid_for_dag_json on the decoded object).
        required = bridge.require_canonical_dag_json_bytes(encoded)
        cid = bridge.cid_for_bytes(
            required,
            base=CID_BASE,
            codec=CID_CODEC,
            mh_type=MH_TYPE,
            version=CID_VERSION,
        )
        validated = bridge.validate_cid(cid, codecs=(CID_CODEC,))
        digest = bridge.digest_hex_from_cid(validated, codecs=(CID_CODEC,))
    except TestExecutionIdentityError:
        raise
    except Exception as exc:
        raise TestExecutionIdentityError(
            "failed to mint ContentIdentity@1: %s" % exc
        ) from exc

    identity = ContentIdentity(
        cid=validated,
        digest_hex=digest,
        canonical_bytes=required,
    )
    return identity.verify(bridge_import=importer)


def content_identity_from_retained_bytes(
    canonical_bytes: bytes,
    *,
    claimed_cid: str | None = None,
    bridge_import: Optional[Callable[[], Any]] = None,
) -> ContentIdentity:
    """Rebuild and verify a content identity from retained canonical bytes."""

    if type(canonical_bytes) is not bytes:
        raise TestExecutionIdentityError(
            "canonical_bytes must be exact bytes"
        )
    importer = bridge_import or _import_multiformats_bridge
    try:
        bridge = importer()
    except ImportError as exc:
        raise TestExecutionIdentityError(
            "multiformats content-identity bridge is unavailable"
        ) from exc

    try:
        required = bridge.require_canonical_dag_json_bytes(canonical_bytes)
        cid = bridge.cid_for_bytes(
            required,
            base=CID_BASE,
            codec=CID_CODEC,
            mh_type=MH_TYPE,
            version=CID_VERSION,
        )
        digest = bridge.digest_hex_from_cid(cid, codecs=(CID_CODEC,))
    except TestExecutionIdentityError:
        raise
    except Exception as exc:
        raise TestExecutionIdentityError(
            "failed to rehash retained bytes: %s" % exc
        ) from exc

    if claimed_cid is not None and claimed_cid != cid:
        raise TestExecutionIdentityError(
            "claimed CID does not match rehash of retained canonical bytes"
        )

    identity = ContentIdentity(
        cid=cid,
        digest_hex=digest,
        canonical_bytes=required,
    )
    return identity.verify(bridge_import=importer)


# ---------------------------------------------------------------------------
# Compiled artifacts
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CompiledTestLocator:
    """Result of compiling a test locator into a content-addressed artifact."""

    # Not a pytest test class.
    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = COMPILED_LOCATOR_SCHEMA

    reusable: bool
    reason_code: str
    non_reusable_reason: str = ""
    locator: Optional[TestLocatorKey] = None
    content_identity: Optional[ContentIdentity] = None
    locator_cid: str = ""
    cid_support: CidSupportStatus = CidSupportStatus.AVAILABLE
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reusable", bool(self.reusable))
        reason = str(self.reason_code or "").strip()
        if not self.reusable and not reason:
            reason = REASON_NON_REUSABLE
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(
            self,
            "non_reusable_reason",
            str(self.non_reusable_reason or "").strip(),
        )
        object.__setattr__(
            self,
            "locator_cid",
            str(self.locator_cid or "").strip(),
        )
        if not isinstance(self.cid_support, CidSupportStatus):
            object.__setattr__(
                self,
                "cid_support",
                CidSupportStatus(str(self.cid_support)),
            )
        object.__setattr__(
            self,
            "diagnostics",
            dict(self.diagnostics or {}),
        )
        if self.reusable:
            if self.content_identity is None or not self.locator_cid:
                raise TestExecutionIdentityError(
                    "reusable compiled locator requires content_identity and locator_cid"
                )
            if self.locator is None:
                raise TestExecutionIdentityError(
                    "reusable compiled locator requires the TestLocatorKey payload"
                )
            if self.locator_cid != self.content_identity.cid:
                raise TestExecutionIdentityError(
                    "locator_cid must equal content_identity.cid"
                )
            if self.non_reusable_reason:
                raise TestExecutionIdentityError(
                    "reusable compiled locator cannot carry non_reusable_reason"
                )
        else:
            # Non-reusable soft path: never invent a CID. Empty locator_cid is
            # allowed; a retained identity may still be present for diagnostics
            # when the payload itself was well-formed but policy-marked
            # non-reusable (parameter serialization failure, etc.).
            if self.locator_cid and self.content_identity is not None:
                if self.locator_cid != self.content_identity.cid:
                    raise TestExecutionIdentityError(
                        "locator_cid must equal content_identity.cid when both set"
                    )

    @property
    def interface(self) -> str:
        return TEST_LOCATOR_KEY_INTERFACE

    @property
    def schema(self) -> str:
        return self.SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": TEST_EXECUTION_CONTRACT_VERSION,
            "interface": TEST_LOCATOR_KEY_INTERFACE,
            "reusable": self.reusable,
            "reason_code": self.reason_code,
            "non_reusable_reason": self.non_reusable_reason,
            "locator_cid": self.locator_cid,
            "cid_support": self.cid_support.value,
            "content_identity": (
                self.content_identity.to_dict()
                if self.content_identity is not None
                else None
            ),
            "locator": self.locator.to_dict() if self.locator is not None else None,
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class CompiledTestExecutionKey:
    """Result of compiling a test execution key into a content-addressed artifact."""

    # Not a pytest test class.
    __test__: ClassVar[bool] = False
    SCHEMA: ClassVar[str] = COMPILED_EXECUTION_KEY_SCHEMA

    reusable: bool
    reason_code: str
    non_reusable_reason: str = ""
    execution_key: Optional[TestExecutionKey] = None
    content_identity: Optional[ContentIdentity] = None
    execution_cid: str = ""
    locator_cid: str = ""
    cid_support: CidSupportStatus = CidSupportStatus.AVAILABLE
    diagnostics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "reusable", bool(self.reusable))
        reason = str(self.reason_code or "").strip()
        if not self.reusable and not reason:
            reason = REASON_NON_REUSABLE
        object.__setattr__(self, "reason_code", reason)
        object.__setattr__(
            self,
            "non_reusable_reason",
            str(self.non_reusable_reason or "").strip(),
        )
        object.__setattr__(
            self, "execution_cid", str(self.execution_cid or "").strip()
        )
        object.__setattr__(
            self, "locator_cid", str(self.locator_cid or "").strip()
        )
        if not isinstance(self.cid_support, CidSupportStatus):
            object.__setattr__(
                self,
                "cid_support",
                CidSupportStatus(str(self.cid_support)),
            )
        object.__setattr__(
            self,
            "diagnostics",
            dict(self.diagnostics or {}),
        )
        if self.reusable:
            if self.content_identity is None or not self.execution_cid:
                raise TestExecutionIdentityError(
                    "reusable compiled execution key requires content_identity "
                    "and execution_cid"
                )
            if self.execution_key is None:
                raise TestExecutionIdentityError(
                    "reusable compiled execution key requires the TestExecutionKey"
                )
            if self.execution_cid != self.content_identity.cid:
                raise TestExecutionIdentityError(
                    "execution_cid must equal content_identity.cid"
                )
            if self.non_reusable_reason:
                raise TestExecutionIdentityError(
                    "reusable compiled execution key cannot carry non_reusable_reason"
                )
        else:
            if self.execution_cid and self.content_identity is not None:
                if self.execution_cid != self.content_identity.cid:
                    raise TestExecutionIdentityError(
                        "execution_cid must equal content_identity.cid when both set"
                    )

    @property
    def interface(self) -> str:
        return TEST_EXECUTION_KEY_INTERFACE

    @property
    def schema(self) -> str:
        return self.SCHEMA

    def to_dict(self) -> Dict[str, Any]:
        return {
            "schema": self.SCHEMA,
            "contract_version": TEST_EXECUTION_CONTRACT_VERSION,
            "interface": TEST_EXECUTION_KEY_INTERFACE,
            "reusable": self.reusable,
            "reason_code": self.reason_code,
            "non_reusable_reason": self.non_reusable_reason,
            "execution_cid": self.execution_cid,
            "locator_cid": self.locator_cid,
            "cid_support": self.cid_support.value,
            "content_identity": (
                self.content_identity.to_dict()
                if self.content_identity is not None
                else None
            ),
            "execution_key": (
                self.execution_key.to_dict()
                if self.execution_key is not None
                else None
            ),
            "diagnostics": dict(self.diagnostics),
        }


# ---------------------------------------------------------------------------
# Normalization helpers
# ---------------------------------------------------------------------------


def normalize_pytest_node_id(node_id: str) -> str:
    """Normalize a pytest node ID for stable locator binding.

    Collapses redundant separators, uses POSIX-style path separators, and
    strips surrounding whitespace. Does not reinterpret parameterization.
    """

    if not isinstance(node_id, str):
        raise TestExecutionIdentityError("node_id must be a string")
    text = node_id.strip().replace("\\", "/")
    while "//" in text:
        text = text.replace("//", "/")
    if not text:
        raise TestExecutionIdentityError("node_id is required")
    return text


def _non_reusable_locator(
    *,
    reason_code: str,
    non_reusable_reason: str,
    cid_support: CidSupportStatus,
    locator: Optional[TestLocatorKey] = None,
    content_identity: Optional[ContentIdentity] = None,
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> CompiledTestLocator:
    locator_cid = content_identity.cid if content_identity is not None else ""
    return CompiledTestLocator(
        reusable=False,
        reason_code=reason_code,
        non_reusable_reason=non_reusable_reason,
        locator=locator,
        content_identity=content_identity,
        locator_cid=locator_cid,
        cid_support=cid_support,
        diagnostics=dict(diagnostics or {}),
    )


def _non_reusable_execution(
    *,
    reason_code: str,
    non_reusable_reason: str,
    cid_support: CidSupportStatus,
    execution_key: Optional[TestExecutionKey] = None,
    content_identity: Optional[ContentIdentity] = None,
    locator_cid: str = "",
    diagnostics: Optional[Mapping[str, Any]] = None,
) -> CompiledTestExecutionKey:
    execution_cid = content_identity.cid if content_identity is not None else ""
    return CompiledTestExecutionKey(
        reusable=False,
        reason_code=reason_code,
        non_reusable_reason=non_reusable_reason,
        execution_key=execution_key,
        content_identity=content_identity,
        execution_cid=execution_cid,
        locator_cid=locator_cid,
        cid_support=cid_support,
        diagnostics=dict(diagnostics or {}),
    )


# ---------------------------------------------------------------------------
# Compiler
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TestExecutionIdentityCompiler:
    """Compile locator and execution keys under the frozen ContentIdentity profile.

    Injectable ``bridge_import`` and ``cid_probe`` support hermetic tests of the
    missing-CID non-reusable path without mutating global imports.
    """

    # Not a pytest test class.
    __test__: ClassVar[bool] = False

    SCHEMA: ClassVar[str] = TEST_EXECUTION_IDENTITY_COMPILER_SCHEMA

    bridge_import: Optional[Callable[[], Any]] = None
    cid_probe: Optional[Callable[[], CidSupportStatus]] = None

    @property
    def interface(self) -> str:
        return TEST_EXECUTION_IDENTITY_COMPILER_INTERFACE

    @property
    def schema(self) -> str:
        return self.SCHEMA

    def probe(self) -> CidSupportStatus:
        if self.cid_probe is not None:
            status = self.cid_probe()
            return status if isinstance(status, CidSupportStatus) else CidSupportStatus(status)
        return probe_cid_support(bridge_import=self.bridge_import)

    def compile_locator(
        self,
        locator: TestLocatorKey | Mapping[str, Any] | None = None,
        /,
        **fields: Any,
    ) -> CompiledTestLocator:
        """Compile a :class:`TestLocatorKey` into a content-addressed locator."""

        status = self.probe()
        if status is not CidSupportStatus.AVAILABLE:
            return _non_reusable_locator(
                reason_code=REASON_CID_PROVIDER_UNAVAILABLE,
                non_reusable_reason="cid_provider_unavailable:%s" % status.value,
                cid_support=status,
                diagnostics={"cid_support": status.value},
            )

        try:
            key = self._coerce_locator(locator, fields)
        except (TestExecutionContractError, TestExecutionIdentityError, TypeError, ValueError) as exc:
            return _non_reusable_locator(
                reason_code=REASON_MALFORMED_ARTIFACT,
                non_reusable_reason="malformed_locator:%s" % _bounded_reason(exc),
                cid_support=status,
                diagnostics={"error": _bounded_reason(exc)},
            )

        # Explicit parameter non-reusability is part of the locator payload and
        # remains content-addressed for index lookup, but marks non-reusable.
        try:
            # Prefer contract.canonical_bytes so str-Enum fields match content_id.
            identity = mint_content_identity(
                key,
                bridge_import=self.bridge_import,
            )
        except TestExecutionIdentityError as exc:
            return _non_reusable_locator(
                reason_code=REASON_CID_PROVIDER_UNAVAILABLE,
                non_reusable_reason="cid_mint_failed:%s" % _bounded_reason(exc),
                cid_support=status,
                locator=key,
                diagnostics={"error": _bounded_reason(exc)},
            )

        # Cross-check formal contract content_id (same DAG-JSON profile).
        if key.content_id != identity.cid:
            return _non_reusable_locator(
                reason_code=REASON_MALFORMED_ARTIFACT,
                non_reusable_reason="locator_content_id_drift",
                cid_support=status,
                locator=key,
                content_identity=identity,
                diagnostics={
                    "contract_content_id": key.content_id,
                    "multiformats_cid": identity.cid,
                },
            )

        if key.non_reusable_reason:
            return CompiledTestLocator(
                reusable=False,
                reason_code=REASON_NON_REUSABLE,
                non_reusable_reason=key.non_reusable_reason,
                locator=key,
                content_identity=identity,
                locator_cid=identity.cid,
                cid_support=status,
                diagnostics={"parameter_non_reusable": True},
            )

        return CompiledTestLocator(
            reusable=True,
            reason_code="",
            non_reusable_reason="",
            locator=key,
            content_identity=identity,
            locator_cid=identity.cid,
            cid_support=status,
            diagnostics={},
        )

    def compile_execution_key(
        self,
        execution_key: TestExecutionKey | Mapping[str, Any] | None = None,
        /,
        **fields: Any,
    ) -> CompiledTestExecutionKey:
        """Compile a :class:`TestExecutionKey` into a content-addressed execution CID."""

        status = self.probe()
        if status is not CidSupportStatus.AVAILABLE:
            return _non_reusable_execution(
                reason_code=REASON_CID_PROVIDER_UNAVAILABLE,
                non_reusable_reason="cid_provider_unavailable:%s" % status.value,
                cid_support=status,
                diagnostics={"cid_support": status.value},
            )

        try:
            key = self._coerce_execution_key(execution_key, fields)
        except (TestExecutionContractError, TestExecutionIdentityError, TypeError, ValueError) as exc:
            return _non_reusable_execution(
                reason_code=REASON_MALFORMED_ARTIFACT,
                non_reusable_reason="malformed_execution_key:%s" % _bounded_reason(exc),
                cid_support=status,
                diagnostics={"error": _bounded_reason(exc)},
            )

        try:
            identity = mint_content_identity(
                key,
                bridge_import=self.bridge_import,
            )
        except TestExecutionIdentityError as exc:
            return _non_reusable_execution(
                reason_code=REASON_CID_PROVIDER_UNAVAILABLE,
                non_reusable_reason="cid_mint_failed:%s" % _bounded_reason(exc),
                cid_support=status,
                execution_key=key,
                locator_cid=key.locator_cid,
                diagnostics={"error": _bounded_reason(exc)},
            )

        if key.content_id != identity.cid:
            return _non_reusable_execution(
                reason_code=REASON_MALFORMED_ARTIFACT,
                non_reusable_reason="execution_content_id_drift",
                cid_support=status,
                execution_key=key,
                content_identity=identity,
                locator_cid=key.locator_cid,
                diagnostics={
                    "contract_content_id": key.content_id,
                    "multiformats_cid": identity.cid,
                },
            )

        non_reusable_reason = ""
        if key.eligibility_class is EligibilityClass.NON_REUSABLE:
            non_reusable_reason = "eligibility_class:non_reusable"
        # Explicit component marker for unsupported / uncontrolled inputs.
        if "non_reusable_reason" in key.components:
            non_reusable_reason = (
                non_reusable_reason or key.components["non_reusable_reason"]
            )
        if non_reusable_reason:
            return CompiledTestExecutionKey(
                reusable=False,
                reason_code=REASON_NON_REUSABLE,
                non_reusable_reason=non_reusable_reason,
                execution_key=key,
                content_identity=identity,
                execution_cid=identity.cid,
                locator_cid=key.locator_cid,
                cid_support=status,
                diagnostics={"eligibility_class": key.eligibility_class.value},
            )

        return CompiledTestExecutionKey(
            reusable=True,
            reason_code="",
            non_reusable_reason="",
            execution_key=key,
            content_identity=identity,
            execution_cid=identity.cid,
            locator_cid=key.locator_cid,
            cid_support=status,
            diagnostics={},
        )

    # -- coercion ------------------------------------------------------------

    def _coerce_locator(
        self,
        locator: TestLocatorKey | Mapping[str, Any] | None,
        fields: Mapping[str, Any],
    ) -> TestLocatorKey:
        if locator is not None and fields:
            raise TestExecutionIdentityError(
                "compile_locator accepts either a TestLocatorKey/mapping or fields, not both"
            )
        if isinstance(locator, TestLocatorKey):
            # Re-normalize node id for stability even on prebuilt keys.
            if locator.node_id != normalize_pytest_node_id(locator.node_id):
                return TestLocatorKey(
                    repository_id=locator.repository_id,
                    package_identity=locator.package_identity,
                    node_id=normalize_pytest_node_id(locator.node_id),
                    collection_schema_version=locator.collection_schema_version,
                    parameter_id=locator.parameter_id,
                    parameter_values_cid=locator.parameter_values_cid,
                    non_reusable_reason=locator.non_reusable_reason,
                    selection_semantics=locator.selection_semantics,
                    root_identity=locator.root_identity,
                    metadata=dict(locator.metadata),
                )
            return locator
        if locator is None:
            raw: Mapping[str, Any] = fields
        elif isinstance(locator, Mapping):
            raw = locator
        else:
            raise TestExecutionIdentityError(
                "locator must be a TestLocatorKey, mapping, or field kwargs"
            )

        node_id = raw.get("node_id", "")
        if node_id:
            node_id = normalize_pytest_node_id(str(node_id))
        return TestLocatorKey(
            repository_id=raw.get("repository_id", ""),
            package_identity=raw.get("package_identity", ""),
            node_id=node_id,
            collection_schema_version=raw.get("collection_schema_version", "1"),
            parameter_id=raw.get("parameter_id", ""),
            parameter_values_cid=raw.get("parameter_values_cid", ""),
            non_reusable_reason=raw.get("non_reusable_reason", ""),
            selection_semantics=raw.get("selection_semantics", "exact_node"),
            root_identity=raw.get("root_identity", ""),
            metadata=raw.get("metadata") or {},
        )

    def _coerce_execution_key(
        self,
        execution_key: TestExecutionKey | Mapping[str, Any] | None,
        fields: Mapping[str, Any],
    ) -> TestExecutionKey:
        if execution_key is not None and fields:
            raise TestExecutionIdentityError(
                "compile_execution_key accepts either a TestExecutionKey/mapping "
                "or fields, not both"
            )
        if isinstance(execution_key, TestExecutionKey):
            return execution_key
        if execution_key is None:
            raw: Mapping[str, Any] = fields
        elif isinstance(execution_key, Mapping):
            raw = execution_key
        else:
            raise TestExecutionIdentityError(
                "execution_key must be a TestExecutionKey, mapping, or field kwargs"
            )

        eligibility = raw.get(
            "eligibility_class", EligibilityClass.REPOSITORY_FOREST_BOUND
        )
        return TestExecutionKey(
            locator_cid=raw.get("locator_cid", ""),
            repository_forest_cid=raw.get("repository_forest_cid", ""),
            git_commit_id=raw.get("git_commit_id", ""),
            git_tree_id=raw.get("git_tree_id", ""),
            gitlink_state_cid=raw.get("gitlink_state_cid", ""),
            dirty_overlay_cid=raw.get("dirty_overlay_cid", ""),
            test_module_cid=raw.get("test_module_cid", ""),
            test_class_cid=raw.get("test_class_cid", ""),
            test_function_cid=raw.get("test_function_cid", ""),
            decorator_cids=tuple(raw.get("decorator_cids") or ()),
            parameter_source_cid=raw.get("parameter_source_cid", ""),
            test_ast_cid=raw.get("test_ast_cid", ""),
            fixture_cids=tuple(raw.get("fixture_cids") or ()),
            conftest_closure_cid=raw.get("conftest_closure_cid", ""),
            hook_plugin_cids=tuple(raw.get("hook_plugin_cids") or ()),
            static_trace_root_cid=raw.get("static_trace_root_cid", ""),
            static_unknown_frontier=tuple(raw.get("static_unknown_frontier") or ()),
            runtime_trace_root_cid=raw.get("runtime_trace_root_cid", ""),
            runtime_completeness_policy=raw.get("runtime_completeness_policy", ""),
            pytest_version=raw.get("pytest_version", ""),
            python_version=raw.get("python_version", ""),
            plugin_versions_cid=raw.get("plugin_versions_cid", ""),
            command_semantics_cid=raw.get("command_semantics_cid", ""),
            config_cid=raw.get("config_cid", ""),
            markers=tuple(raw.get("markers") or ()),
            dependency_lock_cid=raw.get("dependency_lock_cid", ""),
            installed_distributions_cid=raw.get("installed_distributions_cid", ""),
            environment_cid=raw.get("environment_cid", ""),
            platform_cid=raw.get("platform_cid", ""),
            interpreter_abi_cid=raw.get("interpreter_abi_cid", ""),
            hardware_capability_cid=raw.get("hardware_capability_cid", ""),
            external_snapshot_cids=tuple(raw.get("external_snapshot_cids") or ()),
            policy_cid=raw.get("policy_cid", ""),
            canonicalization_schema_cid=raw.get("canonicalization_schema_cid", ""),
            tracer_schema_cid=raw.get("tracer_schema_cid", ""),
            certificate_schema_cid=raw.get("certificate_schema_cid", ""),
            eligibility_class=eligibility,
            components=raw.get("components") or {},
            metadata=raw.get("metadata") or {},
        )


def _bounded_reason(exc: BaseException, *, max_chars: int = 200) -> str:
    text = str(exc).strip().replace("\n", " ")
    if not text:
        text = type(exc).__name__
    if len(text) > max_chars:
        return text[: max_chars - 3] + "..."
    return text


# ---------------------------------------------------------------------------
# Module-level entry points (predicted symbols)
# ---------------------------------------------------------------------------


def compile_test_locator(
    locator: TestLocatorKey | Mapping[str, Any] | None = None,
    /,
    **fields: Any,
) -> CompiledTestLocator:
    """Compile a test locator key into a stable ContentIdentity-backed artifact."""

    return TestExecutionIdentityCompiler().compile_locator(locator, **fields)


def compile_test_execution_key(
    execution_key: TestExecutionKey | Mapping[str, Any] | None = None,
    /,
    **fields: Any,
) -> CompiledTestExecutionKey:
    """Compile a test execution key into a stable ContentIdentity-backed artifact."""

    return TestExecutionIdentityCompiler().compile_execution_key(
        execution_key, **fields
    )


__all__ = (
    "CID_BASE",
    "CID_CODEC",
    "CID_VERSION",
    "COMPILED_EXECUTION_KEY_SCHEMA",
    "COMPILED_LOCATOR_SCHEMA",
    "CONTENT_IDENTITY_INTERFACE",
    "CONTENT_IDENTITY_SCHEMA",
    "CidSupportStatus",
    "CompiledTestExecutionKey",
    "CompiledTestLocator",
    "ContentIdentity",
    "DIGEST_SIZE",
    "MH_TYPE",
    "REASON_CID_PROVIDER_UNAVAILABLE",
    "REASON_MALFORMED_ARTIFACT",
    "REASON_NON_REUSABLE",
    "REASON_UNSUPPORTED",
    "TEST_EXECUTION_IDENTITY_COMPILER_INTERFACE",
    "TEST_EXECUTION_IDENTITY_COMPILER_SCHEMA",
    "TestExecutionIdentityCompiler",
    "TestExecutionIdentityError",
    "cid_support_available",
    "compile_test_execution_key",
    "compile_test_locator",
    "content_identity_from_retained_bytes",
    "mint_content_identity",
    "normalize_pytest_node_id",
    "probe_cid_support",
    "reject_pseudo_cid",
)
