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

Provider imports (``multiformats``, ``ipfs_datasets_py``) remain lazy.
Missing multiformats fails closed for CID-required operations.  A digest-
shaped string is never labeled or accepted as a CID.
"""

from __future__ import annotations

import hashlib
import importlib
import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
from types import ModuleType
from typing import Any, Final

CONTENT_IDENTITY_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/content-identity@1"
)
CONTENT_IDENTITY_SCHEMA_VERSION: Final = 1
CONTENT_IDENTITY_INTERFACE: Final = "ContentIdentity@1"

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


class ProfileContradictionKind(str, Enum):
    """Typed differences among identity providers or profiles."""

    CANONICAL_BYTES_MISMATCH = "canonical_bytes_mismatch"
    CODEC_MISMATCH = "codec_mismatch"
    CID_MISMATCH = "cid_mismatch"
    DIGEST_MISMATCH = "digest_mismatch"
    PROFILE_MISMATCH = "profile_mismatch"
    CROSS_PROFILE_EQUALITY = "cross_profile_equality"
    PROVIDER_UNAVAILABLE = "provider_unavailable"


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
    }
    cid_ready = bool(
        providers[PROVIDER_MULTIFORMATS] and providers[PROVIDER_CID_UTILS]
    )
    ir_ready = bool(providers[PROVIDER_IR_CORE_IDENTITY])
    return {
        "schema": CONTENT_IDENTITY_SCHEMA,
        "schema_version": CONTENT_IDENTITY_SCHEMA_VERSION,
        "interface": CONTENT_IDENTITY_INTERFACE,
        "artifact_profile": STRICT_ARTIFACT_PROFILE,
        "logic_ir_profile": LOGIC_IR_PROFILE,
        "providers": providers,
        "cid_required_operations_ready": cid_ready,
        "logic_ir_ready": ir_ready,
        "cross_profile_equality_allowed": False,
        "digest_labeled_as_cid_allowed": False,
    }


__all__ = [
    "CID_VERSION",
    "CONTENT_IDENTITY_INTERFACE",
    "CONTENT_IDENTITY_SCHEMA",
    "CONTENT_IDENTITY_SCHEMA_VERSION",
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
    "PROVIDER_PROFILE_G",
    "STRICT_ARTIFACT_PROFILE",
    "CidValidationError",
    "ContentIdentity",
    "ContentIdentityError",
    "MultiformatsUnavailableError",
    "ProfileContradiction",
    "ProfileContradictionKind",
    "ProviderUnavailableError",
    "compare_provider_identities",
    "content_identity_probe",
    "decode_and_verify_cid",
    "decode_and_verify_identity",
    "identify_for_profile",
    "identify_logic_ir",
    "identify_strict_artifact",
    "identify_strict_artifact_bytes",
    "is_digest_shaped",
    "multiformats_available",
    "profiles_are_interchangeable",
    "provider_available",
    "require_multiformats",
    "reset_provider_import_cache",
    "sha256_digest_label",
]
