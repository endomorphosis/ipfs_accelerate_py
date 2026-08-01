"""Accelerator-owned multiformats and CID identity bridge (ContentIdentity@1).

SCA content identity is profile-tagged: every authoritative artifact carries
an explicit canonicalization profile, retained canonical bytes, a plain
SHA-256 digest, and a validated CIDv1 with multibase/multicodec/multihash
metadata.  This bridge reuses the datasets identity modules rather than
forking canonicalizers.

Declared profiles
-----------------
* ``strict-dag-json-v1`` — protocol artifacts: lowercase base32 CIDv1,
  multicodec ``dag-json``, multihash ``sha2-256``.  Canonical bytes come
  from :mod:`ipfs_datasets_py.utils.cid_utils`.
* ``ir-canonical-identity-v1`` — domain-separated logic IR: CIDv1, multicodec
  ``raw``, multihash ``sha2-256``.  Canonical envelope bytes come from
  :mod:`ipfs_datasets_py.logic.ir_core.identity`.

Conformance inputs ``logic.ipld_cid`` and ``logic.profile_g`` are compared
explicitly; any canonical-byte or codec difference is a typed profile
contradiction, never an alias.  Cross-profile equality is never inferred
from matching payloads or digests.

SCA-220 extends the bridge with :class:`ContentIdentityBridge@1` real-module
conformance: datasets CID helpers and ``multiformats.CID`` /
``multiformats.multihash`` are invoked when available, agreement and
negative vectors are recorded as receipts, and missing or incompatible
providers become typed blockers (never model-mediated).

Provider imports (``multiformats``, ``ipfs_datasets_py``) remain lazy.
Missing multiformats fails closed for CID-required operations.  A digest-
shaped string is never labeled or accepted as a CID.
"""

from __future__ import annotations

import hashlib
import importlib
import json
import os
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from types import ModuleType
from typing import Any, Final

CONTENT_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/content-identity@1"
)
CONTENT_IDENTITY_SCHEMA_VERSION: Final = 1
CONTENT_IDENTITY_INTERFACE: Final = "ContentIdentity@1"
CONTENT_IDENTITY_BRIDGE_INTERFACE: Final = "ContentIdentityBridge@1"
DATASETS_CONTENT_IDENTITY_CAPABILITY_ID: Final = "datasets-content-identity"
DATASETS_CONTENT_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/datasets-content-identity@1"
)
DATASETS_CONTENT_IDENTITY_SCHEMA_VERSION: Final = 1
DEFAULT_CAPABILITY_RELATIVE_PATH: Final = (
    "data/agent_supervisor/swissknife_contract_assurance/capabilities/"
    "datasets-content-identity.json"
)

STRICT_ARTIFACT_PROFILE: Final = "strict-dag-json-v1"
LOGIC_IR_PROFILE: Final = "ir-canonical-identity-v1"

CID_VERSION: Final = 1
MULTIBASE_BASE32: Final = "base32"
MULTICODEC_DAG_JSON: Final = "dag-json"
MULTICODEC_RAW: Final = "raw"
MULTIHASH_SHA2_256: Final = "sha2-256"
DIGEST_SIZE: Final = 32

PROVIDER_CID_UTILS: Final = "ipfs_datasets_py.utils.cid_utils"
PROVIDER_IR_CORE_IDENTITY: Final = "ipfs_datasets_py.logic.ir_core.identity"
PROVIDER_IPLD_CID: Final = "ipfs_datasets_py.logic.ipld_cid"
PROVIDER_PROFILE_G: Final = "ipfs_datasets_py.logic.profile_g"
PROVIDER_MULTIFORMATS: Final = "multiformats"
PROVIDER_MULTIFORMATS_CID: Final = "multiformats.CID"
PROVIDER_MULTIFORMATS_MULTIHASH: Final = "multiformats.multihash"

AUTHORITY_ROOT_KINDS: Final = (
    "graph",
    "obligation",
    "proof",
    "cache",
    "packet",
)

# Required public symbols per provider (signature compatibility probe).
_PROVIDER_REQUIRED_SYMBOLS: Final[dict[str, tuple[str, ...]]] = {
    PROVIDER_CID_UTILS: (
        "canonical_dag_json_bytes",
        "cid_for_dag_json",
        "cid_for_bytes",
        "validate_cid",
    ),
    PROVIDER_IPLD_CID: (
        "canonical_dag_json",
        "dag_json_cid",
    ),
    PROVIDER_PROFILE_G: (
        "canonical_profile_g_bytes",
        "profile_g_cid",
    ),
    PROVIDER_IR_CORE_IDENTITY: (
        "canonical_identity",
    ),
    PROVIDER_MULTIFORMATS: (
        "CID",
        "multihash",
    ),
}

# ASCII-only payload so dag-json producers (cid_utils / ipld_cid / profile_g)
# share one canonical preimage for the positive agreement vector.
CONFORMANCE_AGREEMENT_PAYLOAD: Final[dict[str, Any]] = {
    "sca": "220",
    "flag": True,
    "nested": {"a": 1, "z": 2},
    "items": [None, False, "x"],
}

_IMPORT_LOCK: Final = threading.Lock()
_MODULE_CACHE: dict[str, ModuleType | None] = {}
_MODULE_ERRORS: dict[str, BaseException] = {}


class ContentIdentityError(ValueError):
    """Base error for content-identity failures."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "content_identity_error",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.reason_code = reason_code
        self.details = dict(details or {})


class MultiformatsUnavailableError(ContentIdentityError):
    """Raised when a CID-required operation cannot load multiformats."""

    def __init__(
        self,
        message: str = (
            "multiformats is required for CID operations and is unavailable"
        ),
        *,
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(
            message,
            reason_code="multiformats_unavailable",
            details=details,
        )


class ProviderUnavailableError(ContentIdentityError):
    """Raised when a datasets identity provider cannot be imported."""

    def __init__(
        self,
        provider: str,
        *,
        cause: BaseException | None = None,
    ) -> None:
        details: dict[str, Any] = {"provider": provider}
        if cause is not None:
            details["cause"] = f"{type(cause).__name__}: {cause}"
        super().__init__(
            f"identity provider unavailable: {provider}",
            reason_code="provider_unavailable",
            details=details,
        )
        self.provider = provider


class CidValidationError(ContentIdentityError):
    """Raised when a CID fails decode or preimage revalidation."""

    def __init__(
        self,
        message: str,
        *,
        reason_code: str = "cid_validation_failed",
        details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message, reason_code=reason_code, details=details)


class ProviderIncompatibleError(ContentIdentityError):
    """Raised when a provider imports but lacks the required public surface."""

    def __init__(
        self,
        provider: str,
        *,
        missing_symbols: Sequence[str] = (),
        details: Mapping[str, Any] | None = None,
    ) -> None:
        payload: dict[str, Any] = {
            "provider": provider,
            "missing_symbols": list(missing_symbols),
        }
        if details:
            payload.update(dict(details))
        super().__init__(
            f"identity provider incompatible: {provider}",
            reason_code="provider_incompatible",
            details=payload,
        )
        self.provider = provider
        self.missing_symbols = tuple(missing_symbols)


class ProfileContradictionKind(str, Enum):
    """Typed differences among identity providers or profiles."""

    CANONICAL_BYTES_MISMATCH = "canonical_bytes_mismatch"
    CODEC_MISMATCH = "codec_mismatch"
    CID_MISMATCH = "cid_mismatch"
    DIGEST_MISMATCH = "digest_mismatch"
    PROFILE_MISMATCH = "profile_mismatch"
    CROSS_PROFILE_EQUALITY = "cross_profile_equality"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_INCOMPATIBLE = "provider_incompatible"


class TypedBlockerKind(str, Enum):
    """Fail-closed blocker categories for missing/incompatible providers."""

    MISSING_PROVIDER = "missing_provider"
    INCOMPATIBLE_PROVIDER = "incompatible_provider"
    MULTIFORMATS_UNAVAILABLE = "multiformats_unavailable"
    PROFILE_CONTRADICTION = "profile_contradiction"
    VALIDATION_FAILURE = "validation_failure"


@dataclass(frozen=True, slots=True)
class ProfileContradiction:
    """One explicit, non-aliasable difference between identity producers."""

    kind: ProfileContradictionKind
    left_provider: str
    right_provider: str
    left_profile: str
    right_profile: str
    reason_code: str
    left_value: str = ""
    right_value: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "left_provider": self.left_provider,
            "right_provider": self.right_provider,
            "left_profile": self.left_profile,
            "right_profile": self.right_profile,
            "reason_code": self.reason_code,
        }
        if self.left_value:
            payload["left_value"] = self.left_value
        if self.right_value:
            payload["right_value"] = self.right_value
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class TypedBlocker:
    """Typed fail-closed blocker (never a silent skip or model fallback)."""

    kind: TypedBlockerKind
    reason_code: str
    message: str
    provider: str = ""
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "kind": self.kind.value,
            "reason_code": self.reason_code,
            "message": self.message,
        }
        if self.provider:
            payload["provider"] = self.provider
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class ProviderBindingReceipt:
    """Exact module/symbol binding for one content-identity provider."""

    module: str
    available: bool
    compatible: bool
    required_symbols: tuple[str, ...]
    present_symbols: tuple[str, ...]
    missing_symbols: tuple[str, ...]
    version: str = ""
    module_file: str = ""
    source_digest: str = ""
    invoked_symbols: tuple[str, ...] = ()
    role: str = ""
    blocker: TypedBlocker | None = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "module": self.module,
            "available": self.available,
            "compatible": self.compatible,
            "required_symbols": list(self.required_symbols),
            "present_symbols": list(self.present_symbols),
            "missing_symbols": list(self.missing_symbols),
            "role": self.role,
        }
        if self.version:
            payload["version"] = self.version
        if self.module_file:
            payload["module_file"] = self.module_file
        if self.source_digest:
            payload["source_digest"] = self.source_digest
        if self.invoked_symbols:
            payload["invoked_symbols"] = list(self.invoked_symbols)
        if self.blocker is not None:
            payload["blocker"] = self.blocker.to_dict()
        return payload


@dataclass(frozen=True, slots=True)
class ConformanceVectorReceipt:
    """One positive or negative multiformats/CID conformance vector."""

    vector_id: str
    polarity: str
    profile: str
    passed: bool
    expected_outcome: str
    actual_outcome: str
    codec: str = ""
    multibase: str = ""
    multihash: str = ""
    cid: str = ""
    digest: str = ""
    raw_digest: str = ""
    byte_length: int = 0
    providers: tuple[str, ...] = ()
    reason_codes: tuple[str, ...] = ()
    details: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "vector_id": self.vector_id,
            "polarity": self.polarity,
            "profile": self.profile,
            "passed": self.passed,
            "expected_outcome": self.expected_outcome,
            "actual_outcome": self.actual_outcome,
        }
        if self.codec:
            payload["codec"] = self.codec
        if self.multibase:
            payload["multibase"] = self.multibase
        if self.multihash:
            payload["multihash"] = self.multihash
        if self.cid:
            payload["cid"] = self.cid
        if self.digest:
            payload["digest"] = self.digest
        if self.raw_digest:
            payload["raw_digest"] = self.raw_digest
        if self.byte_length:
            payload["byte_length"] = self.byte_length
        if self.providers:
            payload["providers"] = list(self.providers)
        if self.reason_codes:
            payload["reason_codes"] = list(self.reason_codes)
        if self.details:
            payload["details"] = dict(self.details)
        return payload


@dataclass(frozen=True, slots=True)
class ContentIdentityConformanceReceipt:
    """SCA-220 real-module CID/multiformats/multihash conformance receipt."""

    schema: str
    schema_version: int
    interface: str
    capability_id: str
    passed: bool
    model_calls: int
    artifact_profile: str
    logic_ir_profile: str
    providers: tuple[ProviderBindingReceipt, ...]
    positive_vectors: tuple[ConformanceVectorReceipt, ...]
    negative_vectors: tuple[ConformanceVectorReceipt, ...]
    blockers: tuple[TypedBlocker, ...]
    agreement: Mapping[str, Any] = field(default_factory=dict)
    root_bindings: Mapping[str, Any] = field(default_factory=dict)
    multiformats_invoked: Mapping[str, Any] = field(default_factory=dict)
    reason_codes: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "schema_version": self.schema_version,
            "interface": self.interface,
            "capability_id": self.capability_id,
            "passed": self.passed,
            "model_calls": self.model_calls,
            "artifact_profile": self.artifact_profile,
            "logic_ir_profile": self.logic_ir_profile,
            "providers": [item.to_dict() for item in self.providers],
            "positive_vectors": [item.to_dict() for item in self.positive_vectors],
            "negative_vectors": [item.to_dict() for item in self.negative_vectors],
            "blockers": [item.to_dict() for item in self.blockers],
            "agreement": dict(self.agreement),
            "root_bindings": dict(self.root_bindings),
            "multiformats_invoked": dict(self.multiformats_invoked),
            "reason_codes": list(self.reason_codes),
            "policies": {
                "cross_profile_equality_allowed": False,
                "digest_labeled_as_cid_allowed": False,
                "missing_or_incompatible_provider": "typed_blocker",
                "decoded_multihash_must_match_canonical_bytes": True,
                "package_root_fallback_can_satisfy_exact_binding": False,
            },
        }


@dataclass(frozen=True, slots=True)
class ContentIdentity:
    """Typed identity record for one retained canonical preimage.

    The ``cid`` field is always a real multiformat CIDv1 string when present.
    Plain digests live only in ``digest`` (``sha256:<hex>``) and must never be
    copied into ``cid``.
    """

    profile: str
    canonical_bytes: bytes
    byte_length: int
    digest: str
    cid: str
    cid_version: int
    multibase: str
    multicodec: str
    multihash: str
    domain: str = ""
    schema_version: str = ""
    provider: str = ""
    reason_codes: tuple[str, ...] = ()
    validated: bool = True

    def to_dict(self, *, include_canonical_bytes: bool = False) -> dict[str, Any]:
        """Return JSON-ready metadata; canonical bytes are optional."""

        payload: dict[str, Any] = {
            "schema": CONTENT_IDENTITY_SCHEMA,
            "schema_version": CONTENT_IDENTITY_SCHEMA_VERSION,
            "interface": CONTENT_IDENTITY_INTERFACE,
            "profile": self.profile,
            "byte_length": self.byte_length,
            "digest": self.digest,
            "cid": self.cid,
            "cid_version": self.cid_version,
            "multibase": self.multibase,
            "multicodec": self.multicodec,
            "multihash": self.multihash,
            "validated": self.validated,
            "reason_codes": list(self.reason_codes),
        }
        if self.domain:
            payload["domain"] = self.domain
        if self.schema_version:
            payload["schema_version"] = self.schema_version
        if self.provider:
            payload["provider"] = self.provider
        if include_canonical_bytes:
            payload["canonical_bytes_hex"] = self.canonical_bytes.hex()
        return payload

    @property
    def hexdigest(self) -> str:
        """Return the bare lowercase SHA-256 hex without the ``sha256:`` label."""

        return self.digest.removeprefix("sha256:")


def _is_multiformats_failure(exc: BaseException | None) -> bool:
    if exc is None:
        return False
    if isinstance(exc, ModuleNotFoundError):
        name = getattr(exc, "name", "") or ""
        text = f"{name} {exc}"
        return "multiformats" in text
    if isinstance(exc, ImportError):
        return "multiformats" in str(exc)
    # Walk chained causes (provider code may wrap the import error).
    cause = exc.__cause__ or exc.__context__
    if cause is not None and cause is not exc:
        return _is_multiformats_failure(cause)
    return "multiformats" in str(exc)


def _module_import_error(
    name: str,
    cause: BaseException | None,
) -> ContentIdentityError:
    if (
        name == PROVIDER_MULTIFORMATS
        or name.startswith("multiformats.")
        or _is_multiformats_failure(cause)
    ):
        return MultiformatsUnavailableError(
            details={
                "provider": name,
                "cause": repr(cause),
            }
        )
    return ProviderUnavailableError(name, cause=cause)


def _cache_module(name: str) -> ModuleType:
    with _IMPORT_LOCK:
        if name in _MODULE_CACHE:
            module = _MODULE_CACHE[name]
            if module is None:
                cause = _MODULE_ERRORS.get(name)
                raise _module_import_error(name, cause) from cause
            return module
        try:
            module = importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - normalize provider boundary failures
            _MODULE_CACHE[name] = None
            _MODULE_ERRORS[name] = exc
            raise _module_import_error(name, exc) from exc
        _MODULE_CACHE[name] = module
        _MODULE_ERRORS.pop(name, None)
        return module


def _require_module(name: str) -> ModuleType:
    return _cache_module(name)


def reset_provider_import_cache() -> None:
    """Clear lazy import caches (intended for tests)."""

    with _IMPORT_LOCK:
        _MODULE_CACHE.clear()
        _MODULE_ERRORS.clear()


def multiformats_available() -> bool:
    """Return whether the optional multiformats package can be imported."""

    try:
        _cache_module(PROVIDER_MULTIFORMATS)
    except MultiformatsUnavailableError:
        return False
    return True


def provider_available(provider: str) -> bool:
    """Return whether a named identity provider module can be imported."""

    try:
        _cache_module(provider)
    except (MultiformatsUnavailableError, ProviderUnavailableError):
        return False
    return True


def require_multiformats() -> ModuleType:
    """Import multiformats or fail closed for CID-required work."""

    return _require_module(PROVIDER_MULTIFORMATS)


def sha256_digest_label(data: bytes | bytearray | memoryview) -> str:
    """Return the plain digest label ``sha256:<hex>`` for *data*.

    This is intentionally not a CID and must never be stored in a ``cid``
    field or accepted by CID validation helpers.
    """

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("sha256_digest_label expects bytes-like input")
    raw = bytes(data)
    return f"sha256:{hashlib.sha256(raw).hexdigest()}"


def is_digest_shaped(value: Any) -> bool:
    """Return True when *value* looks like a labeled or bare hex digest."""

    if not isinstance(value, str) or not value:
        return False
    text = value.strip()
    if text.startswith("sha256:"):
        body = text[len("sha256:") :]
        return len(body) == 64 and all(
            ch in "0123456789abcdefABCDEF" for ch in body
        )
    if len(text) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in text):
        return True
    return False


def _assert_not_digest_as_cid(value: Any, *, field_name: str = "cid") -> None:
    if is_digest_shaped(value):
        raise CidValidationError(
            f"{field_name} must be a multiformat CID, not a digest-shaped string",
            reason_code="digest_labeled_as_cid",
            details={"field": field_name, "value_prefix": str(value)[:16]},
        )


def _sha256_digest_bytes(data: bytes) -> bytes:
    return hashlib.sha256(data).digest()


def _build_identity(
    *,
    profile: str,
    canonical_bytes: bytes,
    cid: str,
    multicodec: str,
    provider: str,
    domain: str = "",
    schema_version: str = "",
    reason_codes: Sequence[str] = (),
    validate: bool = True,
) -> ContentIdentity:
    if not isinstance(canonical_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("canonical_bytes must be bytes-like")
    retained = bytes(canonical_bytes)
    digest = sha256_digest_label(retained)
    _assert_not_digest_as_cid(cid)

    identity = ContentIdentity(
        profile=profile,
        canonical_bytes=retained,
        byte_length=len(retained),
        digest=digest,
        cid=cid,
        cid_version=CID_VERSION,
        multibase=MULTIBASE_BASE32,
        multicodec=multicodec,
        multihash=MULTIHASH_SHA2_256,
        domain=domain,
        schema_version=schema_version,
        provider=provider,
        reason_codes=tuple(reason_codes),
        validated=False,
    )
    if validate:
        return decode_and_verify_identity(identity)
    return identity


def decode_and_verify_cid(
    cid_text: str,
    canonical_bytes: bytes,
    *,
    expected_codec: str,
    expected_profile: str | None = None,
    expected_base: str = MULTIBASE_BASE32,
    expected_version: int = CID_VERSION,
    expected_multihash: str = MULTIHASH_SHA2_256,
) -> dict[str, Any]:
    """Decode *cid_text* and require its raw digest equals SHA-256 of *bytes*.

    Fails closed when multiformats is unavailable.  Rejects digest-shaped
    strings that are not real CIDs.
    """

    _assert_not_digest_as_cid(cid_text)
    if not isinstance(cid_text, str) or not cid_text or cid_text != cid_text.lower():
        raise CidValidationError(
            "CID must be a nonempty lowercase string",
            reason_code="cid_not_lowercase",
        )
    if not isinstance(canonical_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("canonical_bytes must be bytes-like")
    retained = bytes(canonical_bytes)

    mf = require_multiformats()
    try:
        from multiformats import CID, multihash  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - require_multiformats already gates
        raise MultiformatsUnavailableError(details={"cause": repr(exc)}) from exc

    # Prefer the already-imported package if the direct import path differs.
    del mf  # silence unused after side-effect import path above

    try:
        parsed = CID.decode(cid_text)
    except Exception as exc:
        raise CidValidationError(
            "CID is not decodable",
            reason_code="cid_not_decodable",
            details={"cid": cid_text, "cause": repr(exc)},
        ) from exc

    expected_digest = _sha256_digest_bytes(retained)
    expected_digest_size = multihash.get(expected_multihash).max_digest_size
    raw_digest = bytes(parsed.raw_digest)

    failures: list[str] = []
    if parsed.version != expected_version:
        failures.append("cid_version_mismatch")
    if parsed.base.name != expected_base:
        failures.append("multibase_mismatch")
    if parsed.codec.name != expected_codec:
        failures.append("multicodec_mismatch")
    if parsed.hashfun.name != expected_multihash:
        failures.append("multihash_mismatch")
    if expected_digest_size is not None and len(raw_digest) != expected_digest_size:
        failures.append("digest_size_mismatch")
    if raw_digest != expected_digest:
        failures.append("multihash_digest_mismatch")
    if str(parsed) != cid_text:
        failures.append("cid_roundtrip_mismatch")

    if failures:
        raise CidValidationError(
            "CID failed decode/preimage verification: " + ",".join(failures),
            reason_code=failures[0],
            details={
                "cid": cid_text,
                "expected_codec": expected_codec,
                "expected_profile": expected_profile,
                "expected_digest": expected_digest.hex(),
                "raw_digest": raw_digest.hex(),
                "failures": failures,
            },
        )

    return {
        "cid": cid_text,
        "cid_version": int(parsed.version),
        "multibase": parsed.base.name,
        "multicodec": parsed.codec.name,
        "multihash": parsed.hashfun.name,
        "raw_digest": raw_digest.hex(),
        "digest": f"sha256:{expected_digest.hex()}",
        "byte_length": len(retained),
        "validated": True,
        "reason_codes": ["cid_verified"],
    }


def decode_and_verify_identity(identity: ContentIdentity) -> ContentIdentity:
    """Revalidate an existing :class:`ContentIdentity` against its bytes."""

    result = decode_and_verify_cid(
        identity.cid,
        identity.canonical_bytes,
        expected_codec=identity.multicodec,
        expected_profile=identity.profile,
        expected_base=identity.multibase,
        expected_version=identity.cid_version,
        expected_multihash=identity.multihash,
    )
    reason_codes = tuple(
        dict.fromkeys((*identity.reason_codes, *result["reason_codes"]))
    )
    return ContentIdentity(
        profile=identity.profile,
        canonical_bytes=identity.canonical_bytes,
        byte_length=identity.byte_length,
        digest=identity.digest,
        cid=identity.cid,
        cid_version=identity.cid_version,
        multibase=identity.multibase,
        multicodec=identity.multicodec,
        multihash=identity.multihash,
        domain=identity.domain,
        schema_version=identity.schema_version,
        provider=identity.provider,
        reason_codes=reason_codes,
        validated=True,
    )


def identify_strict_artifact(value: Any) -> ContentIdentity:
    """Identity for a protocol artifact under ``strict-dag-json-v1``.

    Uses datasets ``cid_utils`` for strict DAG-JSON canonicalization and
    CIDv1/dag-json/sha2-256.  Requires multiformats (via cid_utils).
    """

    require_multiformats()
    cid_utils = _require_module(PROVIDER_CID_UTILS)
    try:
        canonical_bytes = cid_utils.canonical_dag_json_bytes(value)
        cid = cid_utils.cid_for_dag_json(value)
        # Defense in depth: re-check the provider's codec contract.
        cid_utils.validate_cid(
            cid,
            codecs=(MULTICODEC_DAG_JSON,),
            mh_type=MULTIHASH_SHA2_256,
            version=CID_VERSION,
            base=MULTIBASE_BASE32,
        )
    except MultiformatsUnavailableError:
        raise
    except Exception as exc:
        if _is_multiformats_failure(exc):
            raise MultiformatsUnavailableError(
                details={"provider": PROVIDER_CID_UTILS, "cause": repr(exc)}
            ) from exc
        raise ContentIdentityError(
            f"strict artifact canonicalization failed: {exc}",
            reason_code="strict_artifact_canonicalization_failed",
            details={"provider": PROVIDER_CID_UTILS, "cause": repr(exc)},
        ) from exc

    return _build_identity(
        profile=STRICT_ARTIFACT_PROFILE,
        canonical_bytes=canonical_bytes,
        cid=cid,
        multicodec=MULTICODEC_DAG_JSON,
        provider=PROVIDER_CID_UTILS,
        reason_codes=("strict_dag_json_v1",),
    )


def identify_strict_artifact_bytes(data: bytes | bytearray | memoryview) -> ContentIdentity:
    """Identity for already-canonical DAG-JSON bytes under the artifact profile."""

    if not isinstance(data, (bytes, bytearray, memoryview)):
        raise TypeError("data must be bytes-like")
    retained = bytes(data)
    require_multiformats()
    cid_utils = _require_module(PROVIDER_CID_UTILS)
    try:
        cid = cid_utils.cid_for_bytes(
            retained,
            base=MULTIBASE_BASE32,
            codec=MULTICODEC_DAG_JSON,
            mh_type=MULTIHASH_SHA2_256,
            version=CID_VERSION,
        )
    except MultiformatsUnavailableError:
        raise
    except Exception as exc:
        if _is_multiformats_failure(exc):
            raise MultiformatsUnavailableError(
                details={"provider": PROVIDER_CID_UTILS, "cause": repr(exc)}
            ) from exc
        raise
    return _build_identity(
        profile=STRICT_ARTIFACT_PROFILE,
        canonical_bytes=retained,
        cid=cid,
        multicodec=MULTICODEC_DAG_JSON,
        provider=PROVIDER_CID_UTILS,
        reason_codes=("strict_dag_json_v1", "bytes_preimage"),
    )


def identify_logic_ir(
    payload: Any,
    *,
    domain: str,
    schema_version: str,
    collection_schema: Any = None,
    collection_semantics: Any = None,
) -> ContentIdentity:
    """Identity for logic IR under domain-separated ``ir-canonical-identity-v1``.

    The IR provider assembles CIDs without consulting multiformats, but this
    bridge still revalidates the emitted CID with multiformats so the
    multihash-digest contract holds for every retained preimage.
    """

    require_multiformats()
    ir_identity = _require_module(PROVIDER_IR_CORE_IDENTITY)
    try:
        kwargs: dict[str, Any] = {
            "domain": domain,
            "schema_version": schema_version,
        }
        if collection_schema is not None:
            kwargs["collection_schema"] = collection_schema
        if collection_semantics is not None:
            kwargs["collection_semantics"] = collection_semantics
        record = ir_identity.canonical_identity(payload, **kwargs)
    except Exception as exc:
        raise ContentIdentityError(
            f"logic IR canonicalization failed: {exc}",
            reason_code="logic_ir_canonicalization_failed",
            details={"provider": PROVIDER_IR_CORE_IDENTITY, "cause": repr(exc)},
        ) from exc

    if record.profile != LOGIC_IR_PROFILE:
        raise ContentIdentityError(
            f"unexpected IR identity profile: {record.profile!r}",
            reason_code="logic_ir_profile_mismatch",
            details={"expected": LOGIC_IR_PROFILE, "actual": record.profile},
        )

    return _build_identity(
        profile=LOGIC_IR_PROFILE,
        canonical_bytes=record.canonical_bytes,
        cid=record.cid,
        multicodec=MULTICODEC_RAW,
        provider=PROVIDER_IR_CORE_IDENTITY,
        domain=record.domain,
        schema_version=record.schema_version,
        reason_codes=("ir_canonical_identity_v1",),
    )


def identify_for_profile(
    value: Any,
    *,
    profile: str,
    domain: str = "",
    schema_version: str = "",
    collection_schema: Any = None,
    collection_semantics: Any = None,
) -> ContentIdentity:
    """Dispatch identity computation for a declared profile name."""

    if profile == STRICT_ARTIFACT_PROFILE:
        return identify_strict_artifact(value)
    if profile == LOGIC_IR_PROFILE:
        if not domain or not schema_version:
            raise ContentIdentityError(
                "logic IR identity requires domain and schema_version",
                reason_code="logic_ir_discriminators_required",
            )
        return identify_logic_ir(
            value,
            domain=domain,
            schema_version=schema_version,
            collection_schema=collection_schema,
            collection_semantics=collection_semantics,
        )
    raise ContentIdentityError(
        f"unknown content-identity profile: {profile!r}",
        reason_code="unknown_profile",
        details={"profile": profile},
    )


def _provider_snapshot(
    provider: str,
    value: Any,
    *,
    domain: str,
    schema_version: str,
) -> dict[str, Any]:
    """Compute one provider's identity view for contradiction comparison."""

    # Wire-level DAG-JSON producers share the strict artifact profile name so
    # only real byte/codec/CID divergences become contradictions.  The modules
    # remain distinct via ``provider`` and are never silently aliased when
    # their canonical bytes differ (e.g. ipld_cid ensure_ascii=True).
    if provider == PROVIDER_CID_UTILS:
        module = _require_module(provider)
        try:
            canonical_bytes = module.canonical_dag_json_bytes(value)
            cid = module.cid_for_dag_json(value)
        except Exception as exc:
            if _is_multiformats_failure(exc):
                raise MultiformatsUnavailableError(
                    details={"provider": provider, "cause": repr(exc)}
                ) from exc
            raise
        return {
            "provider": provider,
            "profile": STRICT_ARTIFACT_PROFILE,
            "canonical_bytes": canonical_bytes,
            "digest": sha256_digest_label(canonical_bytes),
            "cid": cid,
            "multicodec": MULTICODEC_DAG_JSON,
        }
    if provider == PROVIDER_IPLD_CID:
        module = _require_module(provider)
        try:
            canonical_bytes = module.canonical_dag_json(value)
            cid = module.dag_json_cid(value)
        except Exception as exc:
            if _is_multiformats_failure(exc):
                raise MultiformatsUnavailableError(
                    details={"provider": provider, "cause": repr(exc)}
                ) from exc
            raise
        return {
            "provider": provider,
            "profile": STRICT_ARTIFACT_PROFILE,
            "canonical_bytes": canonical_bytes,
            "digest": sha256_digest_label(canonical_bytes),
            "cid": cid,
            "multicodec": MULTICODEC_DAG_JSON,
        }
    if provider == PROVIDER_PROFILE_G:
        module = _require_module(provider)
        try:
            canonical_bytes = module.canonical_profile_g_bytes(value)
            cid = module.profile_g_cid(value)
        except Exception as exc:
            if _is_multiformats_failure(exc):
                raise MultiformatsUnavailableError(
                    details={"provider": provider, "cause": repr(exc)}
                ) from exc
            raise
        return {
            "provider": provider,
            "profile": STRICT_ARTIFACT_PROFILE,
            "canonical_bytes": canonical_bytes,
            "digest": sha256_digest_label(canonical_bytes),
            "cid": cid,
            "multicodec": MULTICODEC_DAG_JSON,
        }
    if provider == PROVIDER_IR_CORE_IDENTITY:
        module = _require_module(provider)
        record = module.canonical_identity(
            value,
            domain=domain,
            schema_version=schema_version,
        )
        return {
            "provider": provider,
            "profile": LOGIC_IR_PROFILE,
            "canonical_bytes": record.canonical_bytes,
            "digest": record.digest,
            "cid": record.cid,
            "multicodec": MULTICODEC_RAW,
        }
    raise ContentIdentityError(
        f"unknown provider: {provider}",
        reason_code="unknown_provider",
        details={"provider": provider},
    )


def compare_provider_identities(
    value: Any,
    *,
    domain: str = "sca-compare",
    schema_version: str = "1.0.0",
    providers: Sequence[str] | None = None,
) -> tuple[ProfileContradiction, ...]:
    """Compare datasets identity producers; return typed contradictions.

    Differences in canonical bytes, codec, CID, or profile are never aliased.
    A matching payload does not make ``strict-dag-json-v1`` equal to
    ``ir-canonical-identity-v1``.
    """

    selected = tuple(
        providers
        or (
            PROVIDER_CID_UTILS,
            PROVIDER_IPLD_CID,
            PROVIDER_PROFILE_G,
            PROVIDER_IR_CORE_IDENTITY,
        )
    )
    snapshots: list[dict[str, Any]] = []
    contradictions: list[ProfileContradiction] = []

    for provider in selected:
        try:
            if provider in {
                PROVIDER_CID_UTILS,
                PROVIDER_IPLD_CID,
                PROVIDER_PROFILE_G,
            }:
                require_multiformats()
            snapshots.append(
                _provider_snapshot(
                    provider,
                    value,
                    domain=domain,
                    schema_version=schema_version,
                )
            )
        except (MultiformatsUnavailableError, ProviderUnavailableError) as exc:
            contradictions.append(
                ProfileContradiction(
                    kind=ProfileContradictionKind.PROVIDER_UNAVAILABLE,
                    left_provider=provider,
                    right_provider=provider,
                    left_profile="",
                    right_profile="",
                    reason_code=exc.reason_code,
                    details=dict(exc.details),
                )
            )
        except Exception as exc:  # noqa: BLE001 - surface as typed contradiction
            contradictions.append(
                ProfileContradiction(
                    kind=ProfileContradictionKind.PROVIDER_UNAVAILABLE,
                    left_provider=provider,
                    right_provider=provider,
                    left_profile="",
                    right_profile="",
                    reason_code="provider_computation_failed",
                    details={"cause": repr(exc)},
                )
            )

    for index, left in enumerate(snapshots):
        for right in snapshots[index + 1 :]:
            left_profile = str(left["profile"])
            right_profile = str(right["profile"])
            same_declared_profile = left_profile == right_profile

            if left["multicodec"] != right["multicodec"]:
                contradictions.append(
                    ProfileContradiction(
                        kind=ProfileContradictionKind.CODEC_MISMATCH,
                        left_provider=str(left["provider"]),
                        right_provider=str(right["provider"]),
                        left_profile=left_profile,
                        right_profile=right_profile,
                        reason_code="multicodec_mismatch",
                        left_value=str(left["multicodec"]),
                        right_value=str(right["multicodec"]),
                    )
                )

            if left["canonical_bytes"] != right["canonical_bytes"]:
                contradictions.append(
                    ProfileContradiction(
                        kind=ProfileContradictionKind.CANONICAL_BYTES_MISMATCH,
                        left_provider=str(left["provider"]),
                        right_provider=str(right["provider"]),
                        left_profile=left_profile,
                        right_profile=right_profile,
                        reason_code="canonical_bytes_mismatch",
                        left_value=sha256_digest_label(left["canonical_bytes"]),
                        right_value=sha256_digest_label(right["canonical_bytes"]),
                        details={
                            "left_byte_length": len(left["canonical_bytes"]),
                            "right_byte_length": len(right["canonical_bytes"]),
                        },
                    )
                )

            if left["digest"] != right["digest"]:
                contradictions.append(
                    ProfileContradiction(
                        kind=ProfileContradictionKind.DIGEST_MISMATCH,
                        left_provider=str(left["provider"]),
                        right_provider=str(right["provider"]),
                        left_profile=left_profile,
                        right_profile=right_profile,
                        reason_code="digest_mismatch",
                        left_value=str(left["digest"]),
                        right_value=str(right["digest"]),
                    )
                )

            if left["cid"] != right["cid"]:
                contradictions.append(
                    ProfileContradiction(
                        kind=ProfileContradictionKind.CID_MISMATCH,
                        left_provider=str(left["provider"]),
                        right_provider=str(right["provider"]),
                        left_profile=left_profile,
                        right_profile=right_profile,
                        reason_code="cid_mismatch",
                        left_value=str(left["cid"]),
                        right_value=str(right["cid"]),
                    )
                )

            if left_profile != right_profile:
                contradictions.append(
                    ProfileContradiction(
                        kind=ProfileContradictionKind.PROFILE_MISMATCH,
                        left_provider=str(left["provider"]),
                        right_provider=str(right["provider"]),
                        left_profile=left_profile,
                        right_profile=right_profile,
                        reason_code="profile_mismatch",
                        left_value=left_profile,
                        right_value=right_profile,
                    )
                )
                # Even if digests or CIDs collide across profiles, equality is
                # never inferred; record an explicit cross-profile contradiction.
                if (
                    left["digest"] == right["digest"]
                    or left["cid"] == right["cid"]
                    or left["canonical_bytes"] == right["canonical_bytes"]
                ):
                    contradictions.append(
                        ProfileContradiction(
                            kind=ProfileContradictionKind.CROSS_PROFILE_EQUALITY,
                            left_provider=str(left["provider"]),
                            right_provider=str(right["provider"]),
                            left_profile=left_profile,
                            right_profile=right_profile,
                            reason_code="cross_profile_equality_forbidden",
                            left_value=str(left["cid"]),
                            right_value=str(right["cid"]),
                        )
                    )
            elif same_declared_profile and (
                left["canonical_bytes"] != right["canonical_bytes"]
                or left["cid"] != right["cid"]
            ):
                # Same declared profile name but divergent producers remain
                # contradictions (already recorded above).
                pass

    return tuple(contradictions)


def profiles_are_interchangeable(left: ContentIdentity, right: ContentIdentity) -> bool:
    """Return True only when profile and full multiformat metadata match.

    Matching digests alone never authorize cross-profile equality.
    """

    return (
        left.profile == right.profile
        and left.multicodec == right.multicodec
        and left.multihash == right.multihash
        and left.multibase == right.multibase
        and left.cid_version == right.cid_version
        and left.canonical_bytes == right.canonical_bytes
        and left.cid == right.cid
        and left.digest == right.digest
    )


def content_identity_probe() -> dict[str, Any]:
    """Return a capability probe for analyzer-health and diagnostics."""

    providers = {
        PROVIDER_MULTIFORMATS: multiformats_available(),
        PROVIDER_CID_UTILS: provider_available(PROVIDER_CID_UTILS),
        PROVIDER_IR_CORE_IDENTITY: provider_available(PROVIDER_IR_CORE_IDENTITY),
        PROVIDER_IPLD_CID: provider_available(PROVIDER_IPLD_CID),
        PROVIDER_PROFILE_G: provider_available(PROVIDER_PROFILE_G),
        PROVIDER_MULTIFORMATS_CID: False,
        PROVIDER_MULTIFORMATS_MULTIHASH: False,
    }
    if providers[PROVIDER_MULTIFORMATS]:
        try:
            from multiformats import CID, multihash  # type: ignore[attr-defined]

            providers[PROVIDER_MULTIFORMATS_CID] = callable(
                getattr(CID, "decode", None)
            )
            providers[PROVIDER_MULTIFORMATS_MULTIHASH] = callable(
                getattr(multihash, "digest", None)
            )
        except Exception:  # noqa: BLE001 - probe must never raise
            providers[PROVIDER_MULTIFORMATS_CID] = False
            providers[PROVIDER_MULTIFORMATS_MULTIHASH] = False
    cid_ready = bool(
        providers[PROVIDER_MULTIFORMATS]
        and providers[PROVIDER_CID_UTILS]
        and providers[PROVIDER_MULTIFORMATS_CID]
        and providers[PROVIDER_MULTIFORMATS_MULTIHASH]
    )
    ir_ready = bool(providers[PROVIDER_IR_CORE_IDENTITY])
    return {
        "schema": CONTENT_IDENTITY_SCHEMA,
        "schema_version": CONTENT_IDENTITY_SCHEMA_VERSION,
        "interface": CONTENT_IDENTITY_INTERFACE,
        "bridge_interface": CONTENT_IDENTITY_BRIDGE_INTERFACE,
        "artifact_profile": STRICT_ARTIFACT_PROFILE,
        "logic_ir_profile": LOGIC_IR_PROFILE,
        "providers": providers,
        "cid_required_operations_ready": cid_ready,
        "logic_ir_ready": ir_ready,
        "cross_profile_equality_allowed": False,
        "digest_labeled_as_cid_allowed": False,
        "missing_or_incompatible_provider": "typed_blocker",
        "model_calls": 0,
    }


def _module_version(module: ModuleType) -> str:
    version = getattr(module, "__version__", None)
    if isinstance(version, str) and version:
        return version
    package_name = (getattr(module, "__package__", None) or module.__name__).split(
        ".", 1
    )[0]
    if package_name:
        try:
            package = importlib.import_module(package_name)
        except Exception:  # noqa: BLE001
            package = None
        if package is not None:
            package_version = getattr(package, "__version__", None)
            if isinstance(package_version, str) and package_version:
                return package_version
    return ""


def _module_source_digest(module: ModuleType) -> tuple[str, str]:
    module_file = getattr(module, "__file__", None) or ""
    if not module_file or not os.path.isfile(module_file):
        return str(module_file or ""), ""
    try:
        raw = Path(module_file).read_bytes()
    except OSError:
        return module_file, ""
    return module_file, f"sha256:{hashlib.sha256(raw).hexdigest()}"


def _role_for_provider(provider: str) -> str:
    if provider == PROVIDER_CID_UTILS:
        return "strict_dag_json"
    if provider == PROVIDER_IPLD_CID:
        return "strict_dag_json_legacy_ascii"
    if provider == PROVIDER_PROFILE_G:
        return "strict_dag_json_profile_g"
    if provider == PROVIDER_IR_CORE_IDENTITY:
        return "logic_ir_raw"
    if provider == PROVIDER_MULTIFORMATS:
        return "multiformats_package"
    if provider == PROVIDER_MULTIFORMATS_CID:
        return "multiformats_cid"
    if provider == PROVIDER_MULTIFORMATS_MULTIHASH:
        return "multiformats_multihash"
    return "unknown"


def inspect_provider_binding(provider: str) -> ProviderBindingReceipt:
    """Probe one provider for availability and required public symbols.

    Missing or signature-incompatible providers emit a typed blocker receipt
    rather than a silent pass.  Multiformats sub-symbols are reported as
    distinct bindings (``multiformats.CID``, ``multiformats.multihash``).
    """

    role = _role_for_provider(provider)
    if provider == PROVIDER_MULTIFORMATS_CID:
        try:
            require_multiformats()
            from multiformats import CID  # type: ignore[attr-defined]

            present = []
            if callable(getattr(CID, "decode", None)):
                present.append("CID.decode")
            if hasattr(CID, "version") or callable(CID):
                present.append("CID")
            compatible = "CID.decode" in present
            blocker = None
            if not compatible:
                blocker = TypedBlocker(
                    kind=TypedBlockerKind.INCOMPATIBLE_PROVIDER,
                    reason_code="provider_incompatible",
                    message="multiformats.CID lacks decode()",
                    provider=provider,
                    details={"missing_symbols": ["CID.decode"]},
                )
            return ProviderBindingReceipt(
                module=provider,
                available=True,
                compatible=compatible,
                required_symbols=("CID.decode",),
                present_symbols=tuple(present),
                missing_symbols=() if compatible else ("CID.decode",),
                version=_module_version(require_multiformats()),
                module_file=getattr(CID, "__module__", "multiformats"),
                invoked_symbols=(),
                role=role,
                blocker=blocker,
            )
        except MultiformatsUnavailableError as exc:
            return ProviderBindingReceipt(
                module=provider,
                available=False,
                compatible=False,
                required_symbols=("CID.decode",),
                present_symbols=(),
                missing_symbols=("CID.decode",),
                role=role,
                blocker=TypedBlocker(
                    kind=TypedBlockerKind.MULTIFORMATS_UNAVAILABLE,
                    reason_code=exc.reason_code,
                    message=str(exc),
                    provider=provider,
                    details=dict(exc.details),
                ),
            )

    if provider == PROVIDER_MULTIFORMATS_MULTIHASH:
        try:
            require_multiformats()
            from multiformats import multihash  # type: ignore[attr-defined]

            present = []
            if callable(getattr(multihash, "digest", None)):
                present.append("multihash.digest")
            if callable(getattr(multihash, "get", None)):
                present.append("multihash.get")
            compatible = "multihash.digest" in present
            blocker = None
            if not compatible:
                blocker = TypedBlocker(
                    kind=TypedBlockerKind.INCOMPATIBLE_PROVIDER,
                    reason_code="provider_incompatible",
                    message="multiformats.multihash lacks digest()",
                    provider=provider,
                    details={"missing_symbols": ["multihash.digest"]},
                )
            return ProviderBindingReceipt(
                module=provider,
                available=True,
                compatible=compatible,
                required_symbols=("multihash.digest",),
                present_symbols=tuple(present),
                missing_symbols=() if compatible else ("multihash.digest",),
                version=_module_version(require_multiformats()),
                module_file=getattr(multihash, "__name__", "multiformats.multihash"),
                invoked_symbols=(),
                role=role,
                blocker=blocker,
            )
        except MultiformatsUnavailableError as exc:
            return ProviderBindingReceipt(
                module=provider,
                available=False,
                compatible=False,
                required_symbols=("multihash.digest",),
                present_symbols=(),
                missing_symbols=("multihash.digest",),
                role=role,
                blocker=TypedBlocker(
                    kind=TypedBlockerKind.MULTIFORMATS_UNAVAILABLE,
                    reason_code=exc.reason_code,
                    message=str(exc),
                    provider=provider,
                    details=dict(exc.details),
                ),
            )

    required = _PROVIDER_REQUIRED_SYMBOLS.get(provider, ())
    try:
        module = _require_module(
            PROVIDER_MULTIFORMATS if provider == PROVIDER_MULTIFORMATS else provider
        )
    except (MultiformatsUnavailableError, ProviderUnavailableError) as exc:
        kind = (
            TypedBlockerKind.MULTIFORMATS_UNAVAILABLE
            if isinstance(exc, MultiformatsUnavailableError)
            else TypedBlockerKind.MISSING_PROVIDER
        )
        return ProviderBindingReceipt(
            module=provider,
            available=False,
            compatible=False,
            required_symbols=required,
            present_symbols=(),
            missing_symbols=required,
            role=role,
            blocker=TypedBlocker(
                kind=kind,
                reason_code=exc.reason_code,
                message=str(exc),
                provider=provider,
                details=dict(exc.details),
            ),
        )

    present = tuple(name for name in required if hasattr(module, name))
    missing = tuple(name for name in required if name not in present)
    module_file, source_digest = _module_source_digest(module)
    if missing:
        blocker = TypedBlocker(
            kind=TypedBlockerKind.INCOMPATIBLE_PROVIDER,
            reason_code="provider_incompatible",
            message=f"provider missing required symbols: {provider}",
            provider=provider,
            details={"missing_symbols": list(missing)},
        )
        return ProviderBindingReceipt(
            module=provider,
            available=True,
            compatible=False,
            required_symbols=required,
            present_symbols=present,
            missing_symbols=missing,
            version=_module_version(module),
            module_file=module_file,
            source_digest=source_digest,
            role=role,
            blocker=blocker,
        )
    return ProviderBindingReceipt(
        module=provider,
        available=True,
        compatible=True,
        required_symbols=required,
        present_symbols=present,
        missing_symbols=(),
        version=_module_version(module),
        module_file=module_file,
        source_digest=source_digest,
        role=role,
        blocker=None,
    )


def require_provider(provider: str) -> ModuleType:
    """Import *provider* and require its declared public symbols."""

    binding = inspect_provider_binding(provider)
    if not binding.available:
        if binding.blocker and binding.blocker.kind is TypedBlockerKind.MULTIFORMATS_UNAVAILABLE:
            raise MultiformatsUnavailableError(details=dict(binding.blocker.details))
        raise ProviderUnavailableError(provider)
    if not binding.compatible:
        raise ProviderIncompatibleError(
            provider,
            missing_symbols=binding.missing_symbols,
            details=dict(binding.blocker.details) if binding.blocker else None,
        )
    if provider in {PROVIDER_MULTIFORMATS_CID, PROVIDER_MULTIFORMATS_MULTIHASH}:
        return require_multiformats()
    return _require_module(provider)


def invoke_multiformats_cid_and_multihash(
    canonical_bytes: bytes,
    *,
    codec: str = MULTICODEC_DAG_JSON,
    base: str = MULTIBASE_BASE32,
    mh_type: str = MULTIHASH_SHA2_256,
    version: int = CID_VERSION,
) -> dict[str, Any]:
    """Invoke real ``multiformats.CID`` and ``multiformats.multihash`` entrypoints.

    Returns codec/base/multihash metadata plus the multihash digest of
    *canonical_bytes*.  Fails closed when multiformats is unavailable.
    """

    if not isinstance(canonical_bytes, (bytes, bytearray, memoryview)):
        raise TypeError("canonical_bytes must be bytes-like")
    retained = bytes(canonical_bytes)
    require_multiformats()
    try:
        from multiformats import CID, multihash  # type: ignore[attr-defined]
    except Exception as exc:  # pragma: no cover - require_multiformats gates
        raise MultiformatsUnavailableError(details={"cause": repr(exc)}) from exc

    mh_digest = multihash.digest(retained, mh_type)
    cid_obj = CID(base, version, codec, mh_digest)
    cid_text = str(cid_obj)
    parsed = CID.decode(cid_text)
    raw_digest = bytes(parsed.raw_digest)
    expected = _sha256_digest_bytes(retained)
    if raw_digest != expected:
        raise CidValidationError(
            "multiformats multihash digest does not match SHA-256 of bytes",
            reason_code="multihash_digest_mismatch",
            details={
                "raw_digest": raw_digest.hex(),
                "expected_digest": expected.hex(),
            },
        )
    return {
        "cid": cid_text,
        "cid_version": int(parsed.version),
        "multibase": parsed.base.name,
        "multicodec": parsed.codec.name,
        "multihash": parsed.hashfun.name,
        "raw_digest": raw_digest.hex(),
        "digest": f"sha256:{expected.hex()}",
        "byte_length": len(retained),
        "invoked": {
            PROVIDER_MULTIFORMATS_CID: ["CID", "CID.decode"],
            PROVIDER_MULTIFORMATS_MULTIHASH: ["multihash.digest"],
        },
        "validated": True,
    }


def bind_authority_root(
    root_kind: str,
    value: Any,
    *,
    profile: str = STRICT_ARTIFACT_PROFILE,
    domain: str = "",
    schema_version: str = "",
) -> ContentIdentity:
    """Bind a graph/obligation/proof/cache/packet root to canonical bytes + CID.

    Authority roots use the strict artifact profile unless an explicit logic-IR
    profile is requested.  The returned identity retains exact canonical bytes
    and a multiformats-verified CID.
    """

    kind = str(root_kind or "").strip().lower()
    if kind not in AUTHORITY_ROOT_KINDS:
        raise ContentIdentityError(
            f"unknown authority root kind: {root_kind!r}",
            reason_code="unknown_root_kind",
            details={"root_kind": root_kind, "allowed": list(AUTHORITY_ROOT_KINDS)},
        )
    if profile == LOGIC_IR_PROFILE:
        if not domain or not schema_version:
            raise ContentIdentityError(
                "logic IR root binding requires domain and schema_version",
                reason_code="logic_ir_discriminators_required",
            )
        identity = identify_logic_ir(
            value,
            domain=domain,
            schema_version=schema_version,
        )
    elif profile == STRICT_ARTIFACT_PROFILE:
        # Envelope the root kind so distinct roots never alias solely by payload.
        identity = identify_strict_artifact(
            {"root_kind": kind, "payload": value}
        )
    else:
        raise ContentIdentityError(
            f"unsupported root-binding profile: {profile!r}",
            reason_code="unknown_profile",
            details={"profile": profile},
        )
    return ContentIdentity(
        profile=identity.profile,
        canonical_bytes=identity.canonical_bytes,
        byte_length=identity.byte_length,
        digest=identity.digest,
        cid=identity.cid,
        cid_version=identity.cid_version,
        multibase=identity.multibase,
        multicodec=identity.multicodec,
        multihash=identity.multihash,
        domain=identity.domain,
        schema_version=identity.schema_version,
        provider=identity.provider,
        reason_codes=tuple(
            dict.fromkeys((*identity.reason_codes, f"root:{kind}"))
        ),
        validated=identity.validated,
    )


def _vector_from_identity(
    *,
    vector_id: str,
    polarity: str,
    identity: ContentIdentity,
    providers: Sequence[str],
    expected_outcome: str,
    actual_outcome: str,
    passed: bool,
    reason_codes: Sequence[str] = (),
    details: Mapping[str, Any] | None = None,
) -> ConformanceVectorReceipt:
    return ConformanceVectorReceipt(
        vector_id=vector_id,
        polarity=polarity,
        profile=identity.profile,
        passed=passed,
        expected_outcome=expected_outcome,
        actual_outcome=actual_outcome,
        codec=identity.multicodec,
        multibase=identity.multibase,
        multihash=identity.multihash,
        cid=identity.cid,
        digest=identity.digest,
        raw_digest=identity.hexdigest,
        byte_length=identity.byte_length,
        providers=tuple(providers),
        reason_codes=tuple(reason_codes),
        details=dict(details or {}),
    )


def prove_content_identity_conformance(
    *,
    agreement_payload: Mapping[str, Any] | None = None,
    include_root_bindings: bool = True,
) -> ContentIdentityConformanceReceipt:
    """Run real-module CID/multiformats/multihash conformance (SCA-220).

    * Invokes datasets CID helpers and ``multiformats.CID`` /
      ``multiformats.multihash`` when available.
    * Requires agreeing dag-json providers to share canonical bytes and the
      decoded multihash digest for the agreement payload.
    * Records negative vectors for altered bytes, codec, and profile.
    * Emits typed blockers for missing or incompatible providers.
    * Never performs a model call (``model_calls`` is always ``0``).
    """

    model_calls = 0
    payload = dict(agreement_payload or CONFORMANCE_AGREEMENT_PAYLOAD)
    provider_names = (
        PROVIDER_CID_UTILS,
        PROVIDER_IPLD_CID,
        PROVIDER_PROFILE_G,
        PROVIDER_IR_CORE_IDENTITY,
        PROVIDER_MULTIFORMATS,
        PROVIDER_MULTIFORMATS_CID,
        PROVIDER_MULTIFORMATS_MULTIHASH,
    )
    provider_receipts = tuple(
        inspect_provider_binding(name) for name in provider_names
    )
    blockers: list[TypedBlocker] = [
        item.blocker for item in provider_receipts if item.blocker is not None
    ]

    positive: list[ConformanceVectorReceipt] = []
    negative: list[ConformanceVectorReceipt] = []
    reason_codes: list[str] = []
    agreement: dict[str, Any] = {
        "required": True,
        "payload_keys": sorted(payload.keys()),
        "providers_compared": [],
        "agreed": False,
    }
    multiformats_invoked: dict[str, Any] = {
        PROVIDER_MULTIFORMATS_CID: False,
        PROVIDER_MULTIFORMATS_MULTIHASH: False,
        "symbols": [],
    }
    root_bindings: dict[str, Any] = {}

    dag_providers_ready = all(
        receipt.available and receipt.compatible
        for receipt in provider_receipts
        if receipt.module
        in {PROVIDER_CID_UTILS, PROVIDER_IPLD_CID, PROVIDER_PROFILE_G}
    )
    multiformats_ready = all(
        receipt.available and receipt.compatible
        for receipt in provider_receipts
        if receipt.module
        in {
            PROVIDER_MULTIFORMATS,
            PROVIDER_MULTIFORMATS_CID,
            PROVIDER_MULTIFORMATS_MULTIHASH,
        }
    )
    ir_ready = any(
        receipt.module == PROVIDER_IR_CORE_IDENTITY
        and receipt.available
        and receipt.compatible
        for receipt in provider_receipts
    )

    # --- Positive: strict artifact via cid_utils + multiformats decode ----
    if (
        multiformats_ready
        and any(
            r.module == PROVIDER_CID_UTILS and r.available and r.compatible
            for r in provider_receipts
        )
    ):
        try:
            identity = identify_strict_artifact(payload)
            verified = decode_and_verify_cid(
                identity.cid,
                identity.canonical_bytes,
                expected_codec=MULTICODEC_DAG_JSON,
                expected_profile=STRICT_ARTIFACT_PROFILE,
            )
            mf = invoke_multiformats_cid_and_multihash(
                identity.canonical_bytes,
                codec=MULTICODEC_DAG_JSON,
            )
            multiformats_invoked[PROVIDER_MULTIFORMATS_CID] = True
            multiformats_invoked[PROVIDER_MULTIFORMATS_MULTIHASH] = True
            multiformats_invoked["symbols"] = [
                "CID",
                "CID.decode",
                "multihash.digest",
            ]
            # Real multiformats reconstruction must match datasets CID.
            cid_match = mf["cid"] == identity.cid
            digest_match = mf["raw_digest"] == verified["raw_digest"]
            passed = bool(
                identity.validated
                and verified["validated"]
                and cid_match
                and digest_match
                and identity.multicodec == MULTICODEC_DAG_JSON
                and identity.multibase == MULTIBASE_BASE32
                and identity.multihash == MULTIHASH_SHA2_256
                and identity.cid_version == CID_VERSION
            )
            positive.append(
                _vector_from_identity(
                    vector_id="positive.strict_dag_json_cidv1_base32_sha2_256",
                    polarity="positive",
                    identity=identity,
                    providers=(PROVIDER_CID_UTILS, PROVIDER_MULTIFORMATS),
                    expected_outcome="cid_verified",
                    actual_outcome="cid_verified" if passed else "mismatch",
                    passed=passed,
                    reason_codes=("strict_dag_json_v1", "multiformats_invoked"),
                    details={
                        "multiformats_cid": mf["cid"],
                        "decoded_raw_digest": verified["raw_digest"],
                        "cid_match": cid_match,
                        "digest_match": digest_match,
                    },
                )
            )
            if passed:
                reason_codes.append("positive_strict_artifact_ok")
            else:
                reason_codes.append("positive_strict_artifact_failed")
                blockers.append(
                    TypedBlocker(
                        kind=TypedBlockerKind.VALIDATION_FAILURE,
                        reason_code="positive_strict_artifact_failed",
                        message="strict artifact positive vector failed",
                        provider=PROVIDER_CID_UTILS,
                    )
                )
        except ContentIdentityError as exc:
            positive.append(
                ConformanceVectorReceipt(
                    vector_id="positive.strict_dag_json_cidv1_base32_sha2_256",
                    polarity="positive",
                    profile=STRICT_ARTIFACT_PROFILE,
                    passed=False,
                    expected_outcome="cid_verified",
                    actual_outcome=exc.reason_code,
                    providers=(PROVIDER_CID_UTILS, PROVIDER_MULTIFORMATS),
                    reason_codes=(exc.reason_code,),
                    details=dict(exc.details),
                )
            )
            blockers.append(
                TypedBlocker(
                    kind=TypedBlockerKind.VALIDATION_FAILURE,
                    reason_code=exc.reason_code,
                    message=str(exc),
                    provider=PROVIDER_CID_UTILS,
                    details=dict(exc.details),
                )
            )
    else:
        reason_codes.append("positive_strict_artifact_skipped_provider_blocker")

    # --- Positive: provider agreement on canonical bytes + decoded digest --
    if dag_providers_ready and multiformats_ready:
        try:
            snapshots = [
                _provider_snapshot(
                    name,
                    payload,
                    domain="sca-conformance",
                    schema_version="1.0.0",
                )
                for name in (
                    PROVIDER_CID_UTILS,
                    PROVIDER_IPLD_CID,
                    PROVIDER_PROFILE_G,
                )
            ]
            first = snapshots[0]
            bytes_agree = all(
                item["canonical_bytes"] == first["canonical_bytes"]
                for item in snapshots
            )
            cids_agree = all(item["cid"] == first["cid"] for item in snapshots)
            digests_agree = all(
                item["digest"] == first["digest"] for item in snapshots
            )
            decoded = decode_and_verify_cid(
                str(first["cid"]),
                first["canonical_bytes"],
                expected_codec=MULTICODEC_DAG_JSON,
                expected_profile=STRICT_ARTIFACT_PROFILE,
            )
            agreement = {
                "required": True,
                "payload_keys": sorted(payload.keys()),
                "providers_compared": [str(item["provider"]) for item in snapshots],
                "agreed": bool(
                    bytes_agree and cids_agree and digests_agree and decoded["validated"]
                ),
                "canonical_digest": first["digest"],
                "cid": first["cid"],
                "raw_digest": decoded["raw_digest"],
                "codec": MULTICODEC_DAG_JSON,
                "multibase": MULTIBASE_BASE32,
                "multihash": MULTIHASH_SHA2_256,
                "byte_length": len(first["canonical_bytes"]),
            }
            positive.append(
                ConformanceVectorReceipt(
                    vector_id="positive.providers_agree_canonical_bytes_and_digest",
                    polarity="positive",
                    profile=STRICT_ARTIFACT_PROFILE,
                    passed=bool(agreement["agreed"]),
                    expected_outcome="providers_agree",
                    actual_outcome=(
                        "providers_agree" if agreement["agreed"] else "providers_disagree"
                    ),
                    codec=MULTICODEC_DAG_JSON,
                    multibase=MULTIBASE_BASE32,
                    multihash=MULTIHASH_SHA2_256,
                    cid=str(first["cid"]),
                    digest=str(first["digest"]),
                    raw_digest=str(decoded["raw_digest"]),
                    byte_length=len(first["canonical_bytes"]),
                    providers=tuple(str(item["provider"]) for item in snapshots),
                    reason_codes=("provider_agreement",),
                    details={
                        "bytes_agree": bytes_agree,
                        "cids_agree": cids_agree,
                        "digests_agree": digests_agree,
                    },
                )
            )
            if agreement["agreed"]:
                reason_codes.append("provider_agreement_ok")
            else:
                reason_codes.append("provider_agreement_failed")
                blockers.append(
                    TypedBlocker(
                        kind=TypedBlockerKind.PROFILE_CONTRADICTION,
                        reason_code="provider_agreement_failed",
                        message="dag-json providers disagreed on agreement payload",
                        details=dict(agreement),
                    )
                )
        except ContentIdentityError as exc:
            agreement["agreed"] = False
            agreement["error"] = exc.reason_code
            blockers.append(
                TypedBlocker(
                    kind=TypedBlockerKind.VALIDATION_FAILURE,
                    reason_code=exc.reason_code,
                    message=str(exc),
                    details=dict(exc.details),
                )
            )
    else:
        agreement["agreed"] = False
        agreement["skipped"] = "provider_blocker"
        reason_codes.append("provider_agreement_skipped_provider_blocker")

    # --- Positive: logic IR raw-codec profile -----------------------------
    if ir_ready and multiformats_ready:
        try:
            ir_identity = identify_logic_ir(
                payload,
                domain="sca-conformance",
                schema_version="1.0.0",
            )
            verified_ir = decode_and_verify_cid(
                ir_identity.cid,
                ir_identity.canonical_bytes,
                expected_codec=MULTICODEC_RAW,
                expected_profile=LOGIC_IR_PROFILE,
            )
            passed_ir = bool(
                ir_identity.validated
                and ir_identity.multicodec == MULTICODEC_RAW
                and verified_ir["raw_digest"] == ir_identity.hexdigest
            )
            positive.append(
                _vector_from_identity(
                    vector_id="positive.logic_ir_raw_codec_domain_separated",
                    polarity="positive",
                    identity=ir_identity,
                    providers=(PROVIDER_IR_CORE_IDENTITY, PROVIDER_MULTIFORMATS),
                    expected_outcome="cid_verified",
                    actual_outcome="cid_verified" if passed_ir else "mismatch",
                    passed=passed_ir,
                    reason_codes=("ir_canonical_identity_v1",),
                    details={"domain": ir_identity.domain},
                )
            )
            if passed_ir:
                reason_codes.append("positive_logic_ir_ok")
            else:
                reason_codes.append("positive_logic_ir_failed")
        except ContentIdentityError as exc:
            positive.append(
                ConformanceVectorReceipt(
                    vector_id="positive.logic_ir_raw_codec_domain_separated",
                    polarity="positive",
                    profile=LOGIC_IR_PROFILE,
                    passed=False,
                    expected_outcome="cid_verified",
                    actual_outcome=exc.reason_code,
                    providers=(PROVIDER_IR_CORE_IDENTITY, PROVIDER_MULTIFORMATS),
                    reason_codes=(exc.reason_code,),
                    details=dict(exc.details),
                )
            )

    # --- Negative vectors (must fail) ------------------------------------
    if multiformats_ready and any(
        r.module == PROVIDER_CID_UTILS and r.available and r.compatible
        for r in provider_receipts
    ):
        identity = identify_strict_artifact(payload)

        # Altered bytes
        try:
            decode_and_verify_cid(
                identity.cid,
                b'{"altered":true}',
                expected_codec=MULTICODEC_DAG_JSON,
            )
            negative.append(
                ConformanceVectorReceipt(
                    vector_id="negative.altered_bytes_must_fail",
                    polarity="negative",
                    profile=STRICT_ARTIFACT_PROFILE,
                    passed=False,
                    expected_outcome="multihash_digest_mismatch",
                    actual_outcome="unexpected_success",
                    cid=identity.cid,
                    providers=(PROVIDER_CID_UTILS, PROVIDER_MULTIFORMATS),
                    reason_codes=("negative_vector_did_not_fail",),
                )
            )
            blockers.append(
                TypedBlocker(
                    kind=TypedBlockerKind.VALIDATION_FAILURE,
                    reason_code="negative_altered_bytes_passed",
                    message="altered bytes vector unexpectedly passed",
                    provider=PROVIDER_CID_UTILS,
                )
            )
        except CidValidationError as exc:
            negative.append(
                ConformanceVectorReceipt(
                    vector_id="negative.altered_bytes_must_fail",
                    polarity="negative",
                    profile=STRICT_ARTIFACT_PROFILE,
                    passed=True,
                    expected_outcome="multihash_digest_mismatch",
                    actual_outcome=exc.reason_code,
                    cid=identity.cid,
                    providers=(PROVIDER_CID_UTILS, PROVIDER_MULTIFORMATS),
                    reason_codes=(exc.reason_code,),
                    details={"failures": list(exc.details.get("failures", []))},
                )
            )
            reason_codes.append("negative_altered_bytes_ok")

        # Wrong codec expectation
        try:
            decode_and_verify_cid(
                identity.cid,
                identity.canonical_bytes,
                expected_codec=MULTICODEC_RAW,
            )
            negative.append(
                ConformanceVectorReceipt(
                    vector_id="negative.codec_mismatch_must_fail",
                    polarity="negative",
                    profile=STRICT_ARTIFACT_PROFILE,
                    passed=False,
                    expected_outcome="multicodec_mismatch",
                    actual_outcome="unexpected_success",
                    cid=identity.cid,
                    providers=(PROVIDER_MULTIFORMATS,),
                    reason_codes=("negative_vector_did_not_fail",),
                )
            )
            blockers.append(
                TypedBlocker(
                    kind=TypedBlockerKind.VALIDATION_FAILURE,
                    reason_code="negative_codec_mismatch_passed",
                    message="codec mismatch vector unexpectedly passed",
                )
            )
        except CidValidationError as exc:
            negative.append(
                ConformanceVectorReceipt(
                    vector_id="negative.codec_mismatch_must_fail",
                    polarity="negative",
                    profile=STRICT_ARTIFACT_PROFILE,
                    passed=True,
                    expected_outcome="multicodec_mismatch",
                    actual_outcome=exc.reason_code,
                    cid=identity.cid,
                    codec=MULTICODEC_DAG_JSON,
                    providers=(PROVIDER_MULTIFORMATS,),
                    reason_codes=(exc.reason_code,),
                )
            )
            reason_codes.append("negative_codec_mismatch_ok")

        # Digest-shaped string must not be accepted as CID
        digest = sha256_digest_label(identity.canonical_bytes)
        try:
            decode_and_verify_cid(
                digest,
                identity.canonical_bytes,
                expected_codec=MULTICODEC_DAG_JSON,
            )
            negative.append(
                ConformanceVectorReceipt(
                    vector_id="negative.digest_labeled_as_cid_must_fail",
                    polarity="negative",
                    profile=STRICT_ARTIFACT_PROFILE,
                    passed=False,
                    expected_outcome="digest_labeled_as_cid",
                    actual_outcome="unexpected_success",
                    providers=(PROVIDER_MULTIFORMATS,),
                    reason_codes=("negative_vector_did_not_fail",),
                )
            )
        except CidValidationError as exc:
            negative.append(
                ConformanceVectorReceipt(
                    vector_id="negative.digest_labeled_as_cid_must_fail",
                    polarity="negative",
                    profile=STRICT_ARTIFACT_PROFILE,
                    passed=True,
                    expected_outcome="digest_labeled_as_cid",
                    actual_outcome=exc.reason_code,
                    digest=digest,
                    providers=(PROVIDER_MULTIFORMATS,),
                    reason_codes=(exc.reason_code,),
                )
            )
            reason_codes.append("negative_digest_as_cid_ok")

        # Cross-profile equality forbidden
        if ir_ready:
            ir_identity = identify_logic_ir(
                payload,
                domain="sca-conformance",
                schema_version="1.0.0",
            )
            interchangeable = profiles_are_interchangeable(identity, ir_identity)
            negative.append(
                ConformanceVectorReceipt(
                    vector_id="negative.cross_profile_equality_forbidden",
                    polarity="negative",
                    profile=f"{STRICT_ARTIFACT_PROFILE}!={LOGIC_IR_PROFILE}",
                    passed=not interchangeable,
                    expected_outcome="profiles_not_interchangeable",
                    actual_outcome=(
                        "profiles_not_interchangeable"
                        if not interchangeable
                        else "profiles_interchangeable"
                    ),
                    providers=(PROVIDER_CID_UTILS, PROVIDER_IR_CORE_IDENTITY),
                    reason_codes=("cross_profile_equality_forbidden",),
                    details={
                        "artifact_cid": identity.cid,
                        "ir_cid": ir_identity.cid,
                        "artifact_codec": identity.multicodec,
                        "ir_codec": ir_identity.multicodec,
                    },
                )
            )
            if not interchangeable:
                reason_codes.append("negative_cross_profile_ok")
            else:
                blockers.append(
                    TypedBlocker(
                        kind=TypedBlockerKind.PROFILE_CONTRADICTION,
                        reason_code="cross_profile_equality_forbidden",
                        message="artifact and IR profiles were treated as interchangeable",
                    )
                )

    # --- Authority root bindings -----------------------------------------
    if include_root_bindings and multiformats_ready and any(
        r.module == PROVIDER_CID_UTILS and r.available and r.compatible
        for r in provider_receipts
    ):
        bindings: dict[str, Any] = {}
        for kind in AUTHORITY_ROOT_KINDS:
            bound = bind_authority_root(kind, {"root": kind, "payload": payload})
            bindings[kind] = {
                "profile": bound.profile,
                "cid": bound.cid,
                "digest": bound.digest,
                "byte_length": bound.byte_length,
                "multicodec": bound.multicodec,
                "multihash": bound.multihash,
                "multibase": bound.multibase,
                "validated": bound.validated,
            }
        # Distinct roots must not alias.
        cids = [bindings[kind]["cid"] for kind in AUTHORITY_ROOT_KINDS]
        root_bindings = {
            "kinds": list(AUTHORITY_ROOT_KINDS),
            "profile": STRICT_ARTIFACT_PROFILE,
            "bindings": bindings,
            "distinct_cids": len(set(cids)) == len(cids),
        }
        if root_bindings["distinct_cids"]:
            reason_codes.append("root_bindings_ok")
        else:
            reason_codes.append("root_bindings_alias")
            blockers.append(
                TypedBlocker(
                    kind=TypedBlockerKind.VALIDATION_FAILURE,
                    reason_code="root_bindings_alias",
                    message="authority roots produced aliasing CIDs",
                )
            )

    # Conformance passes only when required positives/negatives hold and no
    # unexpected validation blockers remain. Provider-missing environments
    # fail closed via blockers rather than a false pass.
    required_positive_ids = {
        "positive.strict_dag_json_cidv1_base32_sha2_256",
        "positive.providers_agree_canonical_bytes_and_digest",
    }
    required_negative_ids = {
        "negative.altered_bytes_must_fail",
        "negative.codec_mismatch_must_fail",
        "negative.digest_labeled_as_cid_must_fail",
        "negative.cross_profile_equality_forbidden",
    }
    positive_ok = {
        item.vector_id: item.passed for item in positive if item.vector_id in required_positive_ids
    }
    negative_ok = {
        item.vector_id: item.passed for item in negative if item.vector_id in required_negative_ids
    }
    hard_blockers = [
        item
        for item in blockers
        if item.kind
        in {
            TypedBlockerKind.MISSING_PROVIDER,
            TypedBlockerKind.INCOMPATIBLE_PROVIDER,
            TypedBlockerKind.MULTIFORMATS_UNAVAILABLE,
            TypedBlockerKind.VALIDATION_FAILURE,
            TypedBlockerKind.PROFILE_CONTRADICTION,
        }
    ]
    vectors_ready = (
        set(positive_ok) >= required_positive_ids
        and set(negative_ok) >= required_negative_ids
        and all(positive_ok.values())
        and all(negative_ok.values())
    )
    # Logic IR positive is required only when the IR provider is present.
    if ir_ready and multiformats_ready:
        ir_vector = next(
            (
                item
                for item in positive
                if item.vector_id == "positive.logic_ir_raw_codec_domain_separated"
            ),
            None,
        )
        if ir_vector is None or not ir_vector.passed:
            vectors_ready = False
    passed = bool(vectors_ready and not hard_blockers and model_calls == 0)
    if passed:
        reason_codes.append("conformance_passed")
    else:
        reason_codes.append("conformance_failed")

    return ContentIdentityConformanceReceipt(
        schema=DATASETS_CONTENT_IDENTITY_SCHEMA,
        schema_version=DATASETS_CONTENT_IDENTITY_SCHEMA_VERSION,
        interface=CONTENT_IDENTITY_BRIDGE_INTERFACE,
        capability_id=DATASETS_CONTENT_IDENTITY_CAPABILITY_ID,
        passed=passed,
        model_calls=model_calls,
        artifact_profile=STRICT_ARTIFACT_PROFILE,
        logic_ir_profile=LOGIC_IR_PROFILE,
        providers=provider_receipts,
        positive_vectors=tuple(positive),
        negative_vectors=tuple(negative),
        blockers=tuple(blockers),
        agreement=agreement,
        root_bindings=root_bindings,
        multiformats_invoked=multiformats_invoked,
        reason_codes=tuple(dict.fromkeys(reason_codes)),
    )


def build_datasets_content_identity_capability(
    *,
    receipt: ContentIdentityConformanceReceipt | None = None,
    repository_relative: bool = True,
) -> dict[str, Any]:
    """Build the ``datasets-content-identity`` capability document."""

    conf = receipt or prove_content_identity_conformance()
    providers_payload: list[dict[str, Any]] = []
    for item in conf.providers:
        entry = item.to_dict()
        if repository_relative and entry.get("module_file"):
            # Drop absolute host paths; keep only the module basename for
            # portable capability documents.
            entry["module_file"] = os.path.basename(str(entry["module_file"]))
        providers_payload.append(entry)

    return {
        "schema": DATASETS_CONTENT_IDENTITY_SCHEMA,
        "schema_version": DATASETS_CONTENT_IDENTITY_SCHEMA_VERSION,
        "capability_id": DATASETS_CONTENT_IDENTITY_CAPABILITY_ID,
        "interface": CONTENT_IDENTITY_BRIDGE_INTERFACE,
        "content_identity_interface": CONTENT_IDENTITY_INTERFACE,
        "task_id": "SCA-220",
        "passed": conf.passed,
        "model_calls": conf.model_calls,
        "artifact_profile": {
            "canonicalization": STRICT_ARTIFACT_PROFILE,
            "cid_version": CID_VERSION,
            "multibase": MULTIBASE_BASE32,
            "multicodec": MULTICODEC_DAG_JSON,
            "multihash": MULTIHASH_SHA2_256,
        },
        "logic_ir_profile": {
            "canonicalization": LOGIC_IR_PROFILE,
            "cid_version": CID_VERSION,
            "multibase": MULTIBASE_BASE32,
            "multicodec": MULTICODEC_RAW,
            "multihash": MULTIHASH_SHA2_256,
        },
        "providers": providers_payload,
        "conformance": {
            "passed": conf.passed,
            "positive_vectors": [item.to_dict() for item in conf.positive_vectors],
            "negative_vectors": [item.to_dict() for item in conf.negative_vectors],
            "blockers": [item.to_dict() for item in conf.blockers],
            "agreement": dict(conf.agreement),
            "multiformats_invoked": dict(conf.multiformats_invoked),
            "reason_codes": list(conf.reason_codes),
        },
        "root_bindings": dict(conf.root_bindings),
        "policies": {
            "cross_profile_equality_allowed": False,
            "digest_labeled_as_cid_allowed": False,
            "missing_or_incompatible_provider": "typed_blocker",
            "decoded_multihash_must_match_canonical_bytes": True,
            "package_root_fallback_can_satisfy_exact_binding": False,
            "cid_required_operations_fail_closed": True,
        },
    }


def write_datasets_content_identity_capability(
    path: str | os.PathLike[str] | None = None,
    *,
    receipt: ContentIdentityConformanceReceipt | None = None,
    repository_root: str | os.PathLike[str] | None = None,
) -> Path:
    """Write the datasets content-identity capability JSON atomically."""

    payload = build_datasets_content_identity_capability(receipt=receipt)
    if path is None:
        root = Path(repository_root) if repository_root is not None else Path.cwd()
        target = root / DEFAULT_CAPABILITY_RELATIVE_PATH
    else:
        target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    rendered = json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
    tmp = target.with_suffix(target.suffix + ".tmp")
    tmp.write_text(rendered, encoding="utf-8")
    os.replace(tmp, target)
    return target


def missing_provider_blockers(
    providers: Sequence[str] | None = None,
) -> tuple[TypedBlocker, ...]:
    """Return typed blockers for every missing/incompatible provider."""

    selected = tuple(
        providers
        or (
            PROVIDER_CID_UTILS,
            PROVIDER_IPLD_CID,
            PROVIDER_PROFILE_G,
            PROVIDER_IR_CORE_IDENTITY,
            PROVIDER_MULTIFORMATS,
            PROVIDER_MULTIFORMATS_CID,
            PROVIDER_MULTIFORMATS_MULTIHASH,
        )
    )
    blockers: list[TypedBlocker] = []
    for name in selected:
        binding = inspect_provider_binding(name)
        if binding.blocker is not None:
            blockers.append(binding.blocker)
    return tuple(blockers)


__all__ = [
    "AUTHORITY_ROOT_KINDS",
    "CID_VERSION",
    "CONFORMANCE_AGREEMENT_PAYLOAD",
    "CONTENT_IDENTITY_BRIDGE_INTERFACE",
    "CONTENT_IDENTITY_INTERFACE",
    "CONTENT_IDENTITY_SCHEMA",
    "CONTENT_IDENTITY_SCHEMA_VERSION",
    "DATASETS_CONTENT_IDENTITY_CAPABILITY_ID",
    "DATASETS_CONTENT_IDENTITY_SCHEMA",
    "DATASETS_CONTENT_IDENTITY_SCHEMA_VERSION",
    "DEFAULT_CAPABILITY_RELATIVE_PATH",
    "DIGEST_SIZE",
    "LOGIC_IR_PROFILE",
    "MULTIBASE_BASE32",
    "MULTICODEC_DAG_JSON",
    "MULTICODEC_RAW",
    "MULTIHASH_SHA2_256",
    "PROVIDER_CID_UTILS",
    "PROVIDER_IPLD_CID",
    "PROVIDER_IR_CORE_IDENTITY",
    "PROVIDER_MULTIFORMATS",
    "PROVIDER_MULTIFORMATS_CID",
    "PROVIDER_MULTIFORMATS_MULTIHASH",
    "PROVIDER_PROFILE_G",
    "STRICT_ARTIFACT_PROFILE",
    "CidValidationError",
    "ConformanceVectorReceipt",
    "ContentIdentity",
    "ContentIdentityConformanceReceipt",
    "ContentIdentityError",
    "MultiformatsUnavailableError",
    "ProfileContradiction",
    "ProfileContradictionKind",
    "ProviderBindingReceipt",
    "ProviderIncompatibleError",
    "ProviderUnavailableError",
    "TypedBlocker",
    "TypedBlockerKind",
    "bind_authority_root",
    "build_datasets_content_identity_capability",
    "compare_provider_identities",
    "content_identity_probe",
    "decode_and_verify_cid",
    "decode_and_verify_identity",
    "identify_for_profile",
    "identify_logic_ir",
    "identify_strict_artifact",
    "identify_strict_artifact_bytes",
    "inspect_provider_binding",
    "invoke_multiformats_cid_and_multihash",
    "is_digest_shaped",
    "missing_provider_blockers",
    "multiformats_available",
    "profiles_are_interchangeable",
    "prove_content_identity_conformance",
    "provider_available",
    "require_multiformats",
    "require_provider",
    "reset_provider_import_cache",
    "sha256_digest_label",
    "write_datasets_content_identity_capability",
]
