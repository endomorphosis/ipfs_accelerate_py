"""Authenticated, explicitly non-production LGCVF receipts for sole-user R&D.

This module verifies retained receipts; it does not generate keys, sign receipts,
issue authority, update a task board, or authorize release/production.  Trust is
supplied by the caller as one locally pinned Ed25519 public key.  A public key
embedded in a receipt is evidence only and never nominates its own trust root.

The deliberately narrow trust model is ``self_signed_single_user_r_and_d``.
It permits the same sole user to record both an R&D self-verification and a
production-declined operator decision.  It must never be represented as
third-party independence, external qualification, release qualification, or
production authorization.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import re
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field as dataclass_field
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    PublicFormat,
    load_pem_public_key,
)
from multiformats import CID

from ..proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)

LGCVF_R_AND_D_TRUST_MODEL: Final = "self_signed_single_user_r_and_d"
LGCVF_R_AND_D_AUTHORITY_SCOPE: Final = "research_and_development_only"
LGCVF_EXTERNAL_RECEIPT_SCHEMA_V2: Final = "lgcvf-external-qualification-receipt@2"
LGCVF_PRODUCTION_RECEIPT_SCHEMA_V2: Final = "lgcvf-production-authorization-receipt@2"
LGCVF_EXTERNAL_R_AND_D_DISPOSITION: Final = "self_verified_r_and_d"
LGCVF_PRODUCTION_DECLINED_DISPOSITION: Final = "production_declined_r_and_d"
LGCVF_ED25519_ALGORITHM: Final = "Ed25519"
LGCVF_BASE64URL_ENCODING: Final = "base64url-no-pad"
LGCVF_R_AND_D_SIGNATURE_DOMAIN: Final = (
    b"ipfs-accelerate/lgcvf-r-and-d-authority-receipt/v2\0"
)
LGCVF_R_AND_D_TRUST_MANIFEST_SCHEMA: Final = "lgcvf-r-and-d-trust-policy@1"
LGCVF_R_AND_D_TRUST_MANIFEST_PATH: Final = "config/lgcvf_r_and_d_authority_trust.json"
LGCVF_R_AND_D_PUBLIC_KEY_PATH: Final = "config/lgcvf_r_and_d_authority_public_key.pem"
LGCVF_R_AND_D_SIGNER_IDENTITY: Final = "Benjamin Barber"
LGCVF_R_AND_D_SIGNER_ROLE: Final = "sole R&D verifier and operator"
LGCVF_R_AND_D_PINNED_KEY_ID: Final = (
    "baguqeeraof5lqknosljjp2d26xqynxi2um53vtfq74dx6apttc3xxsapvslq"
)
LGCVF_R_AND_D_PINNED_PUBLIC_KEY_RAW_SHA256: Final = (
    "sha256:8c3b0a628ca26fde650090269ab3653bc3fdb920536e9585e383a7e47041d0ce"
)

_MAX_TEXT: Final = 4096
_CID_PATTERN: Final = re.compile(r"^b[a-z2-7]+$")
_GIT_OID_PATTERN: Final = re.compile(r"^(?:[0-9a-f]{40}|[0-9a-f]{64})$")
_SHA256_PATTERN: Final = re.compile(r"^sha256:[0-9a-f]{64}$")
_RFC3339_PATTERN: Final = re.compile(
    r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d{1,6})?(?:Z|[+-]\d{2}:\d{2})$"
)
_B64URL_PATTERN: Final = re.compile(r"^[A-Za-z0-9_-]+$")

_SIGNER_FIELDS: Final = frozenset(
    {"identity", "role", "key_id", "public_key_base64url"}
)
_SIGNATURE_FIELDS: Final = frozenset({"algorithm", "encoding", "value"})
_SOURCE_FIELDS: Final = frozenset({"ipfs_accelerate_py", "ipfs_datasets_py"})
_ACCELERATOR_SOURCE_FIELDS: Final = frozenset({"head", "tree"})
_DATASETS_SOURCE_FIELDS: Final = frozenset({"head", "tree", "gitlink"})
_COHORT_FIELDS: Final = frozenset(
    {
        "live_local_model_execution",
        "live_remote_model_execution",
        "production_authoritative_evidence",
    }
)
_COHORT_DISPOSITIONS: Final = frozenset({"passed", "missed", "unavailable"})
_MULTI_WRITER_FIELDS: Final = frozenset({"quack_qualified", "disposition", "notes"})
_TRUST_MANIFEST_FIELDS: Final = frozenset(
    {
        "schema",
        "algorithm",
        "authority_scope",
        "identity",
        "key_id",
        "private_key_committed",
        "public_key_base64url",
        "public_key_path",
        "public_key_raw_sha256",
        "role",
        "third_party_independence_claimed",
        "trust_model",
    }
)

_EXTERNAL_PAYLOAD_FIELDS: Final = frozenset(
    {
        "schema",
        "receipt_kind",
        "trust_model",
        "authority_scope",
        "issuer",
        "third_party_independence_claimed",
        "issued_at",
        "expires_at",
        "plan_cid",
        "qualification_result_cid",
        "qualification_checkout_fingerprint_cid",
        "benchmark_report_cid",
        "source_revisions",
        "cohorts",
        "provider_disclosure_policy",
        "multi_writer",
        "disposition",
        "release_qualified",
        "production_authorized",
        "limitations",
    }
)
_PRODUCTION_PAYLOAD_FIELDS: Final = frozenset(
    {
        "schema",
        "receipt_kind",
        "trust_model",
        "authority_scope",
        "operator",
        "issued_at",
        "expires_at",
        "plan_cid",
        "qualification_result_cid",
        "qualification_checkout_fingerprint_cid",
        "benchmark_report_cid",
        "external_qualification_receipt_cid",
        "external_qualification_payload_cid",
        "release_report_sha256",
        "source_revisions",
        "scope",
        "lgswf_006_reused",
        "depends_on_lgcvf_121",
        "depends_on_lgcvf_122",
        "disposition",
        "release_qualified",
        "production_authorized",
        "limitations",
    }
)
_ENVELOPE_FIELDS: Final = frozenset({"payload_cid", "signature", "receipt_cid"})


class LgcvfRAndDAuthorityError(ValueError):
    """A malformed, stale, untrusted, or authority-raising R&D receipt."""


def _closed_mapping(
    value: Any,
    fields: frozenset[str],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise LgcvfRAndDAuthorityError(f"{label} must be an object")
    if any(type(key) is not str for key in value):
        raise LgcvfRAndDAuthorityError(f"{label} keys must be strings")
    if set(value) != fields:
        raise LgcvfRAndDAuthorityError(f"{label} fields differ from the closed schema")
    return dict(value)


def _text(value: Any, *, label: str) -> str:
    if type(value) is not str or not value or len(value) > _MAX_TEXT:
        raise LgcvfRAndDAuthorityError(f"{label} must be nonempty bounded text")
    if value != value.strip() or unicodedata.normalize("NFC", value) != value:
        raise LgcvfRAndDAuthorityError(f"{label} must be trimmed NFC text")
    if any(not character.isprintable() for character in value):
        raise LgcvfRAndDAuthorityError(f"{label} contains non-printable text")
    return value


def _canonical_cid(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    if _CID_PATTERN.fullmatch(text) is None:
        raise LgcvfRAndDAuthorityError(f"{label} is not a canonical CID")
    try:
        parsed = CID.decode(text)
    except Exception as exc:
        raise LgcvfRAndDAuthorityError(f"{label} is not a valid CID") from exc
    if (
        parsed.version != 1
        or parsed.codec.name != "dag-json"
        or parsed.hashfun.name != "sha2-256"
        or parsed.encode() != text
    ):
        raise LgcvfRAndDAuthorityError(
            f"{label} must be canonical CIDv1/base32/dag-json/sha2-256"
        )
    return text


def _git_oid(value: Any, *, label: str) -> str:
    text = _text(value, label=label)
    if _GIT_OID_PATTERN.fullmatch(text) is None:
        raise LgcvfRAndDAuthorityError(
            f"{label} must be a lowercase SHA-1 or SHA-256 Git object ID"
        )
    return text


def _release_sha256(value: Any) -> str:
    text = _text(value, label="release_report_sha256")
    if _SHA256_PATTERN.fullmatch(text) is None:
        raise LgcvfRAndDAuthorityError(
            "release_report_sha256 must be sha256 plus 64 lowercase hex digits"
        )
    return text


def _decode_base64url(value: Any, *, length: int, label: str) -> bytes:
    text = _text(value, label=label)
    if "=" in text or _B64URL_PATTERN.fullmatch(text) is None:
        raise LgcvfRAndDAuthorityError(f"{label} must be unpadded canonical base64url")
    try:
        decoded = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except Exception as exc:
        raise LgcvfRAndDAuthorityError(f"{label} is malformed") from exc
    canonical = base64.urlsafe_b64encode(decoded).decode("ascii").rstrip("=")
    if len(decoded) != length or canonical != text:
        raise LgcvfRAndDAuthorityError(f"{label} must encode exactly {length} bytes")
    return decoded


def _encode_base64url(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def ed25519_public_key_id(public_key: bytes) -> str:
    """Return the canonical key ID for exact raw Ed25519 public-key bytes."""

    if type(public_key) is not bytes or len(public_key) != 32:
        raise LgcvfRAndDAuthorityError("Ed25519 public key must be exactly 32 bytes")
    material = {
        "algorithm": LGCVF_ED25519_ALGORITHM,
        "encoding": LGCVF_BASE64URL_ENCODING,
        "public_key_base64url": _encode_base64url(public_key),
        "usage": "lgcvf_self_signed_single_user_r_and_d_receipts",
    }
    return content_identity(material)


@dataclass(frozen=True, slots=True)
class LgcvfSourceRevisions:
    """Exact accelerator and datasets source identities judged by a receipt."""

    accelerator_head: str
    accelerator_tree: str
    datasets_head: str
    datasets_tree: str
    datasets_gitlink: str

    def __post_init__(self) -> None:
        for name in (
            "accelerator_head",
            "accelerator_tree",
            "datasets_head",
            "datasets_tree",
            "datasets_gitlink",
        ):
            object.__setattr__(self, name, _git_oid(getattr(self, name), label=name))

    def to_dict(self) -> dict[str, dict[str, str]]:
        return {
            "ipfs_accelerate_py": {
                "head": self.accelerator_head,
                "tree": self.accelerator_tree,
            },
            "ipfs_datasets_py": {
                "head": self.datasets_head,
                "tree": self.datasets_tree,
                "gitlink": self.datasets_gitlink,
            },
        }

    @classmethod
    def from_dict(cls, value: Any) -> LgcvfSourceRevisions:
        source = _closed_mapping(value, _SOURCE_FIELDS, label="source_revisions")
        accelerator = _closed_mapping(
            source["ipfs_accelerate_py"],
            _ACCELERATOR_SOURCE_FIELDS,
            label="source_revisions.ipfs_accelerate_py",
        )
        datasets = _closed_mapping(
            source["ipfs_datasets_py"],
            _DATASETS_SOURCE_FIELDS,
            label="source_revisions.ipfs_datasets_py",
        )
        return cls(
            accelerator_head=accelerator["head"],
            accelerator_tree=accelerator["tree"],
            datasets_head=datasets["head"],
            datasets_tree=datasets["tree"],
            datasets_gitlink=datasets["gitlink"],
        )


@dataclass(frozen=True, slots=True)
class LgcvfAuthorityBindings:
    """Caller-pinned current LGCVF evidence and source identities."""

    plan_cid: str
    qualification_result_cid: str
    qualification_checkout_fingerprint_cid: str
    benchmark_report_cid: str
    release_report_sha256: str
    source_revisions: LgcvfSourceRevisions

    def __post_init__(self) -> None:
        for name in (
            "plan_cid",
            "qualification_result_cid",
            "qualification_checkout_fingerprint_cid",
            "benchmark_report_cid",
        ):
            object.__setattr__(
                self,
                name,
                _canonical_cid(getattr(self, name), label=name),
            )
        object.__setattr__(
            self, "release_report_sha256", _release_sha256(self.release_report_sha256)
        )
        if not isinstance(self.source_revisions, LgcvfSourceRevisions):
            raise LgcvfRAndDAuthorityError(
                "source_revisions must be LgcvfSourceRevisions"
            )


@dataclass(frozen=True, slots=True)
class LgcvfRAndDTrustPolicy:
    """Locally pinned sole-user R&D signer; receipt-embedded keys are untrusted."""

    identity: str
    role: str
    key_id: str
    public_key: bytes
    trust_model: str = LGCVF_R_AND_D_TRUST_MODEL

    def __post_init__(self) -> None:
        object.__setattr__(self, "identity", _text(self.identity, label="identity"))
        object.__setattr__(self, "role", _text(self.role, label="role"))
        if self.role != LGCVF_R_AND_D_SIGNER_ROLE:
            raise LgcvfRAndDAuthorityError(
                "R&D signer role must identify the sole R&D verifier and operator"
            )
        if self.trust_model != LGCVF_R_AND_D_TRUST_MODEL:
            raise LgcvfRAndDAuthorityError("unsupported LGCVF trust model")
        if type(self.public_key) is not bytes or len(self.public_key) != 32:
            raise LgcvfRAndDAuthorityError(
                "trusted Ed25519 public key must be exactly 32 bytes"
            )
        expected_key_id = ed25519_public_key_id(self.public_key)
        object.__setattr__(self, "key_id", _canonical_cid(self.key_id, label="key_id"))
        if self.key_id != expected_key_id:
            raise LgcvfRAndDAuthorityError(
                "trusted key_id does not bind the exact Ed25519 public key"
            )

    @property
    def public_key_base64url(self) -> str:
        return _encode_base64url(self.public_key)

    @classmethod
    def from_public_pem_path(
        cls,
        *,
        identity: str,
        role: str,
        key_id: str,
        public_key_pem_path: str | Path,
        expected_public_key_raw_sha256: str,
    ) -> LgcvfRAndDTrustPolicy:
        """Load one fingerprint-pinned Ed25519 public key from a PEM file."""

        fingerprint = _text(
            expected_public_key_raw_sha256,
            label="expected_public_key_raw_sha256",
        )
        if _SHA256_PATTERN.fullmatch(fingerprint) is None:
            raise LgcvfRAndDAuthorityError(
                "expected_public_key_raw_sha256 must be sha256 plus 64 lowercase hex digits"
            )
        path = Path(public_key_pem_path)
        try:
            pem = path.read_bytes()
        except OSError as exc:
            raise LgcvfRAndDAuthorityError(
                f"cannot read trusted public key PEM: {path}"
            ) from exc
        if not pem or len(pem) > 16_384:
            raise LgcvfRAndDAuthorityError(
                "trusted public key PEM must be nonempty and at most 16384 bytes"
            )
        try:
            parsed = load_pem_public_key(pem)
        except (TypeError, ValueError) as exc:
            raise LgcvfRAndDAuthorityError(
                "trusted public key PEM is malformed"
            ) from exc
        if not isinstance(parsed, Ed25519PublicKey):
            raise LgcvfRAndDAuthorityError(
                "trusted public key PEM must contain one Ed25519 public key"
            )
        canonical_pem = parsed.public_bytes(
            Encoding.PEM,
            PublicFormat.SubjectPublicKeyInfo,
        )
        if not hmac.compare_digest(pem, canonical_pem):
            raise LgcvfRAndDAuthorityError(
                "trusted public key PEM must contain only one canonical public key"
            )
        raw = parsed.public_bytes(Encoding.Raw, PublicFormat.Raw)
        observed_fingerprint = "sha256:" + hashlib.sha256(raw).hexdigest()
        if not hmac.compare_digest(observed_fingerprint, fingerprint):
            raise LgcvfRAndDAuthorityError(
                "trusted public key PEM does not match its pinned raw SHA-256"
            )
        return cls(
            identity=identity,
            role=role,
            key_id=key_id,
            public_key=raw,
        )


@dataclass(frozen=True, slots=True)
class ValidatedLgcvfRAndDReceipt:
    """Non-authorizing projection returned after full semantic verification."""

    receipt_kind: str
    disposition: str
    signer_identity: str
    issued_at: datetime
    expires_at: datetime
    payload_cid: str
    receipt_cid: str
    release_qualified: bool = dataclass_field(default=False, init=False)
    production_authorized: bool = dataclass_field(default=False, init=False)


def _json_object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise LgcvfRAndDAuthorityError(
                f"trust manifest contains duplicate field {key!r}"
            )
        result[key] = value
    return result


def _repository_file(
    repository_root: Path,
    relative_path: str,
    *,
    label: str,
) -> Path:
    text = _text(relative_path, label=label)
    portable = PurePosixPath(text)
    if (
        portable.is_absolute()
        or portable.as_posix() != text
        or any(part in {"", ".", ".."} for part in portable.parts)
    ):
        raise LgcvfRAndDAuthorityError(
            f"{label} must be a normalized repository-relative POSIX path"
        )
    try:
        resolved = repository_root.joinpath(*portable.parts).resolve(strict=True)
        resolved.relative_to(repository_root)
    except (OSError, ValueError) as exc:
        raise LgcvfRAndDAuthorityError(
            f"{label} does not resolve to a file inside the repository"
        ) from exc
    if not resolved.is_file():
        raise LgcvfRAndDAuthorityError(f"{label} must resolve to a regular file")
    return resolved


def load_lgcvf_r_and_d_trust_policy(
    repository_root: str | Path,
    *,
    trust_manifest_path: str = LGCVF_R_AND_D_TRUST_MANIFEST_PATH,
) -> LgcvfRAndDTrustPolicy:
    """Load the closed, repository-pinned sole-user R&D trust manifest and PEM.

    The manifest and its public key are inputs to verification only.  The
    manifest cannot nominate third-party independence or release/production
    authority, and no private-key path is accepted.
    """

    try:
        root = Path(repository_root).resolve(strict=True)
    except OSError as exc:
        raise LgcvfRAndDAuthorityError("repository_root does not exist") from exc
    if not root.is_dir():
        raise LgcvfRAndDAuthorityError("repository_root must be a directory")
    manifest_path = _repository_file(
        root,
        trust_manifest_path,
        label="trust_manifest_path",
    )
    try:
        encoded = manifest_path.read_bytes()
    except OSError as exc:
        raise LgcvfRAndDAuthorityError("cannot read LGCVF trust manifest") from exc
    if not encoded or len(encoded) > 65_536:
        raise LgcvfRAndDAuthorityError(
            "LGCVF trust manifest must be nonempty and at most 65536 bytes"
        )
    try:
        decoded = encoded.decode("utf-8")
        parsed = json.loads(decoded, object_pairs_hook=_json_object_without_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise LgcvfRAndDAuthorityError(
            "LGCVF trust manifest must be strict UTF-8 JSON"
        ) from exc
    manifest = _closed_mapping(
        parsed,
        _TRUST_MANIFEST_FIELDS,
        label="LGCVF R&D trust manifest",
    )
    expected_constants = {
        "schema": LGCVF_R_AND_D_TRUST_MANIFEST_SCHEMA,
        "algorithm": LGCVF_ED25519_ALGORITHM,
        "authority_scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
        "identity": LGCVF_R_AND_D_SIGNER_IDENTITY,
        "key_id": LGCVF_R_AND_D_PINNED_KEY_ID,
        "private_key_committed": False,
        "public_key_path": LGCVF_R_AND_D_PUBLIC_KEY_PATH,
        "public_key_raw_sha256": LGCVF_R_AND_D_PINNED_PUBLIC_KEY_RAW_SHA256,
        "role": LGCVF_R_AND_D_SIGNER_ROLE,
        "third_party_independence_claimed": False,
        "trust_model": LGCVF_R_AND_D_TRUST_MODEL,
    }
    for field, wanted in expected_constants.items():
        observed = manifest[field]
        differs = (
            observed is not wanted if isinstance(wanted, bool) else observed != wanted
        )
        if differs:
            raise LgcvfRAndDAuthorityError(f"LGCVF R&D trust manifest {field} differs")
    public_key_path_text = _text(
        manifest["public_key_path"],
        label="public_key_path",
    )
    public_key_path = _repository_file(
        root,
        public_key_path_text,
        label="public_key_path",
    )
    policy = LgcvfRAndDTrustPolicy.from_public_pem_path(
        identity=manifest["identity"],
        role=manifest["role"],
        key_id=manifest["key_id"],
        public_key_pem_path=public_key_path,
        expected_public_key_raw_sha256=manifest["public_key_raw_sha256"],
    )
    embedded_public_key = _decode_base64url(
        manifest["public_key_base64url"],
        length=32,
        label="public_key_base64url",
    )
    if not hmac.compare_digest(embedded_public_key, policy.public_key):
        raise LgcvfRAndDAuthorityError(
            "trust manifest public key does not match its pinned PEM"
        )
    return policy


def _rfc3339(value: Any, *, label: str) -> datetime:
    text = _text(value, label=label)
    if _RFC3339_PATTERN.fullmatch(text) is None or text.endswith("-00:00"):
        raise LgcvfRAndDAuthorityError(f"{label} must be an RFC3339 timestamp")
    try:
        parsed = datetime.fromisoformat(
            text[:-1] + "+00:00" if text.endswith("Z") else text
        )
    except ValueError as exc:
        raise LgcvfRAndDAuthorityError(f"{label} must be an RFC3339 timestamp") from exc
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise LgcvfRAndDAuthorityError(f"{label} must carry a UTC offset")
    return parsed.astimezone(timezone.utc)


def _current_time(value: datetime | None) -> datetime:
    current = datetime.now(timezone.utc) if value is None else value
    if (
        not isinstance(current, datetime)
        or current.tzinfo is None
        or current.utcoffset() is None
    ):
        raise LgcvfRAndDAuthorityError("now must be a timezone-aware datetime")
    return current.astimezone(timezone.utc)


def _validity_interval(
    receipt: Mapping[str, Any],
    *,
    now: datetime | None,
) -> tuple[datetime, datetime]:
    issued_at = _rfc3339(receipt["issued_at"], label="issued_at")
    expires_at = _rfc3339(receipt["expires_at"], label="expires_at")
    if expires_at <= issued_at:
        raise LgcvfRAndDAuthorityError("receipt validity interval is inverted or empty")
    current = _current_time(now)
    if current < issued_at:
        raise LgcvfRAndDAuthorityError("receipt is not yet valid")
    if current >= expires_at:
        raise LgcvfRAndDAuthorityError("receipt is expired")
    return issued_at, expires_at


def _validate_signer(value: Any, trust: LgcvfRAndDTrustPolicy, *, label: str) -> None:
    signer = _closed_mapping(value, _SIGNER_FIELDS, label=label)
    identity = _text(signer["identity"], label=f"{label}.identity")
    role = _text(signer["role"], label=f"{label}.role")
    key_id = _canonical_cid(signer["key_id"], label=f"{label}.key_id")
    public_key = _decode_base64url(
        signer["public_key_base64url"],
        length=32,
        label=f"{label}.public_key_base64url",
    )
    if (
        identity != trust.identity
        or role != trust.role
        or key_id != trust.key_id
        or public_key != trust.public_key
    ):
        raise LgcvfRAndDAuthorityError(
            f"{label} does not match the locally pinned R&D trust policy"
        )
    if ed25519_public_key_id(public_key) != key_id:
        raise LgcvfRAndDAuthorityError(f"{label}.key_id does not bind its public key")


def _validate_sources(value: Any, expected: LgcvfSourceRevisions) -> None:
    observed = LgcvfSourceRevisions.from_dict(value)
    if observed != expected:
        raise LgcvfRAndDAuthorityError("receipt source revisions are stale or foreign")


def _validate_limitations(value: Any) -> None:
    if not isinstance(value, list) or not value:
        raise LgcvfRAndDAuthorityError("limitations must be a nonempty array")
    limitations = tuple(_text(item, label="limitation") for item in value)
    if len(limitations) != len(set(limitations)):
        raise LgcvfRAndDAuthorityError("limitations must not contain duplicates")


def _validate_external_specific(receipt: Mapping[str, Any]) -> None:
    cohorts = _closed_mapping(receipt["cohorts"], _COHORT_FIELDS, label="cohorts")
    for name, value in cohorts.items():
        if type(value) is not str or value not in _COHORT_DISPOSITIONS:
            raise LgcvfRAndDAuthorityError(f"cohorts.{name} has invalid disposition")
    multi_writer = _closed_mapping(
        receipt["multi_writer"], _MULTI_WRITER_FIELDS, label="multi_writer"
    )
    if type(multi_writer["quack_qualified"]) is not bool:
        raise LgcvfRAndDAuthorityError("multi_writer.quack_qualified must be boolean")
    if (
        type(multi_writer["disposition"]) is not str
        or multi_writer["disposition"] not in _COHORT_DISPOSITIONS
    ):
        raise LgcvfRAndDAuthorityError("multi_writer.disposition is invalid")
    _text(multi_writer["notes"], label="multi_writer.notes")
    _text(receipt["provider_disclosure_policy"], label="provider_disclosure_policy")


def _verify_envelope(
    receipt: Mapping[str, Any],
    *,
    payload_fields: frozenset[str],
    trust: LgcvfRAndDTrustPolicy,
) -> tuple[str, str]:
    payload = {field: receipt[field] for field in payload_fields}
    try:
        payload_bytes = canonical_json_bytes(payload)
        expected_payload_cid = content_identity(payload)
    except Exception as exc:
        raise LgcvfRAndDAuthorityError("receipt payload is not canonical JSON") from exc
    payload_cid = _canonical_cid(receipt["payload_cid"], label="payload_cid")
    if payload_cid != expected_payload_cid:
        raise LgcvfRAndDAuthorityError(
            "payload_cid does not bind the canonical payload"
        )

    signature_block = _closed_mapping(
        receipt["signature"], _SIGNATURE_FIELDS, label="signature"
    )
    if signature_block["algorithm"] != LGCVF_ED25519_ALGORITHM:
        raise LgcvfRAndDAuthorityError("signature algorithm must be Ed25519")
    if signature_block["encoding"] != LGCVF_BASE64URL_ENCODING:
        raise LgcvfRAndDAuthorityError("signature encoding must be base64url-no-pad")
    signature = _decode_base64url(
        signature_block["value"], length=64, label="signature.value"
    )
    try:
        Ed25519PublicKey.from_public_bytes(trust.public_key).verify(
            signature,
            LGCVF_R_AND_D_SIGNATURE_DOMAIN + payload_bytes,
        )
    except (InvalidSignature, TypeError, ValueError) as exc:
        raise LgcvfRAndDAuthorityError("Ed25519 receipt signature is invalid") from exc

    receipt_body = {
        key: value for key, value in receipt.items() if key != "receipt_cid"
    }
    try:
        expected_receipt_cid = content_identity(receipt_body)
    except Exception as exc:
        raise LgcvfRAndDAuthorityError(
            "receipt envelope is not canonical JSON"
        ) from exc
    receipt_cid = _canonical_cid(receipt["receipt_cid"], label="receipt_cid")
    if receipt_cid != expected_receipt_cid:
        raise LgcvfRAndDAuthorityError(
            "receipt_cid does not bind payload CID, signature, and envelope"
        )
    return payload_cid, receipt_cid


def _require_binding(value: Any, expected: str, *, label: str) -> None:
    observed = _canonical_cid(value, label=label)
    if observed != expected:
        raise LgcvfRAndDAuthorityError(f"{label} is stale or foreign")


def validate_lgcvf_external_r_and_d_receipt(
    value: Mapping[str, Any],
    *,
    trust: LgcvfRAndDTrustPolicy,
    expected: LgcvfAuthorityBindings,
    now: datetime | None = None,
) -> ValidatedLgcvfRAndDReceipt:
    """Verify one self-signed R&D observation without granting external authority."""

    if not isinstance(trust, LgcvfRAndDTrustPolicy):
        raise LgcvfRAndDAuthorityError("trust must be LgcvfRAndDTrustPolicy")
    if not isinstance(expected, LgcvfAuthorityBindings):
        raise LgcvfRAndDAuthorityError("expected must be LgcvfAuthorityBindings")
    receipt = _closed_mapping(
        value,
        _EXTERNAL_PAYLOAD_FIELDS | _ENVELOPE_FIELDS,
        label="external R&D receipt",
    )
    expected_constants = {
        "schema": LGCVF_EXTERNAL_RECEIPT_SCHEMA_V2,
        "receipt_kind": "external_qualification_r_and_d",
        "trust_model": LGCVF_R_AND_D_TRUST_MODEL,
        "authority_scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
        "third_party_independence_claimed": False,
        "disposition": LGCVF_EXTERNAL_R_AND_D_DISPOSITION,
        "release_qualified": False,
        "production_authorized": False,
    }
    for field, wanted in expected_constants.items():
        observed = receipt[field]
        differs = (
            observed is not wanted if isinstance(wanted, bool) else observed != wanted
        )
        if differs:
            raise LgcvfRAndDAuthorityError(f"external R&D receipt {field} differs")
    _validate_signer(receipt["issuer"], trust, label="issuer")
    issued_at, expires_at = _validity_interval(receipt, now=now)
    _require_binding(receipt["plan_cid"], expected.plan_cid, label="plan_cid")
    _require_binding(
        receipt["qualification_result_cid"],
        expected.qualification_result_cid,
        label="qualification_result_cid",
    )
    _require_binding(
        receipt["qualification_checkout_fingerprint_cid"],
        expected.qualification_checkout_fingerprint_cid,
        label="qualification_checkout_fingerprint_cid",
    )
    _require_binding(
        receipt["benchmark_report_cid"],
        expected.benchmark_report_cid,
        label="benchmark_report_cid",
    )
    _validate_sources(receipt["source_revisions"], expected.source_revisions)
    _validate_external_specific(receipt)
    _validate_limitations(receipt["limitations"])
    payload_cid, receipt_cid = _verify_envelope(
        receipt, payload_fields=_EXTERNAL_PAYLOAD_FIELDS, trust=trust
    )
    return ValidatedLgcvfRAndDReceipt(
        receipt_kind=receipt["receipt_kind"],
        disposition=receipt["disposition"],
        signer_identity=trust.identity,
        issued_at=issued_at,
        expires_at=expires_at,
        payload_cid=payload_cid,
        receipt_cid=receipt_cid,
    )


def validate_lgcvf_production_declined_r_and_d_receipt(
    value: Mapping[str, Any],
    *,
    external_receipt: Mapping[str, Any],
    trust: LgcvfRAndDTrustPolicy,
    expected: LgcvfAuthorityBindings,
    now: datetime | None = None,
) -> ValidatedLgcvfRAndDReceipt:
    """Verify a sole-user operator decision that explicitly declines production."""

    external = validate_lgcvf_external_r_and_d_receipt(
        external_receipt,
        trust=trust,
        expected=expected,
        now=now,
    )
    receipt = _closed_mapping(
        value,
        _PRODUCTION_PAYLOAD_FIELDS | _ENVELOPE_FIELDS,
        label="production-declined R&D receipt",
    )
    expected_constants = {
        "schema": LGCVF_PRODUCTION_RECEIPT_SCHEMA_V2,
        "receipt_kind": "production_authorization_r_and_d",
        "trust_model": LGCVF_R_AND_D_TRUST_MODEL,
        "authority_scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
        "scope": LGCVF_R_AND_D_AUTHORITY_SCOPE,
        "lgswf_006_reused": False,
        "depends_on_lgcvf_121": True,
        "depends_on_lgcvf_122": True,
        "disposition": LGCVF_PRODUCTION_DECLINED_DISPOSITION,
        "release_qualified": False,
        "production_authorized": False,
    }
    for field, wanted in expected_constants.items():
        observed = receipt[field]
        differs = (
            observed is not wanted if isinstance(wanted, bool) else observed != wanted
        )
        if differs:
            raise LgcvfRAndDAuthorityError(
                f"production-declined R&D receipt {field} differs"
            )
    _validate_signer(receipt["operator"], trust, label="operator")
    issued_at, expires_at = _validity_interval(receipt, now=now)
    if issued_at < external.issued_at or expires_at > external.expires_at:
        raise LgcvfRAndDAuthorityError(
            "operator receipt validity must be contained by the external R&D receipt"
        )
    _require_binding(receipt["plan_cid"], expected.plan_cid, label="plan_cid")
    _require_binding(
        receipt["qualification_result_cid"],
        expected.qualification_result_cid,
        label="qualification_result_cid",
    )
    _require_binding(
        receipt["qualification_checkout_fingerprint_cid"],
        expected.qualification_checkout_fingerprint_cid,
        label="qualification_checkout_fingerprint_cid",
    )
    _require_binding(
        receipt["benchmark_report_cid"],
        expected.benchmark_report_cid,
        label="benchmark_report_cid",
    )
    _require_binding(
        receipt["external_qualification_receipt_cid"],
        external.receipt_cid,
        label="external_qualification_receipt_cid",
    )
    _require_binding(
        receipt["external_qualification_payload_cid"],
        external.payload_cid,
        label="external_qualification_payload_cid",
    )
    if (
        _release_sha256(receipt["release_report_sha256"])
        != expected.release_report_sha256
    ):
        raise LgcvfRAndDAuthorityError("release_report_sha256 is stale or foreign")
    _validate_sources(receipt["source_revisions"], expected.source_revisions)
    _validate_limitations(receipt["limitations"])
    payload_cid, receipt_cid = _verify_envelope(
        receipt, payload_fields=_PRODUCTION_PAYLOAD_FIELDS, trust=trust
    )
    return ValidatedLgcvfRAndDReceipt(
        receipt_kind=receipt["receipt_kind"],
        disposition=receipt["disposition"],
        signer_identity=trust.identity,
        issued_at=issued_at,
        expires_at=expires_at,
        payload_cid=payload_cid,
        receipt_cid=receipt_cid,
    )


__all__ = [
    "LGCVF_BASE64URL_ENCODING",
    "LGCVF_ED25519_ALGORITHM",
    "LGCVF_EXTERNAL_RECEIPT_SCHEMA_V2",
    "LGCVF_EXTERNAL_R_AND_D_DISPOSITION",
    "LGCVF_PRODUCTION_DECLINED_DISPOSITION",
    "LGCVF_PRODUCTION_RECEIPT_SCHEMA_V2",
    "LGCVF_R_AND_D_AUTHORITY_SCOPE",
    "LGCVF_R_AND_D_PINNED_KEY_ID",
    "LGCVF_R_AND_D_PINNED_PUBLIC_KEY_RAW_SHA256",
    "LGCVF_R_AND_D_PUBLIC_KEY_PATH",
    "LGCVF_R_AND_D_SIGNATURE_DOMAIN",
    "LGCVF_R_AND_D_SIGNER_IDENTITY",
    "LGCVF_R_AND_D_SIGNER_ROLE",
    "LGCVF_R_AND_D_TRUST_MANIFEST_PATH",
    "LGCVF_R_AND_D_TRUST_MANIFEST_SCHEMA",
    "LGCVF_R_AND_D_TRUST_MODEL",
    "LgcvfAuthorityBindings",
    "LgcvfRAndDAuthorityError",
    "LgcvfRAndDTrustPolicy",
    "LgcvfSourceRevisions",
    "ValidatedLgcvfRAndDReceipt",
    "ed25519_public_key_id",
    "load_lgcvf_r_and_d_trust_policy",
    "validate_lgcvf_external_r_and_d_receipt",
    "validate_lgcvf_production_declined_r_and_d_receipt",
]
