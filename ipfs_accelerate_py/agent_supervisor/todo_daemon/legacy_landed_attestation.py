"""Signed, non-authoritative evidence for audited legacy landed reviews.

``LegacyLandedReviewAttestation@1`` is deliberately distinct from a
``ProviderExecutionReceipt``.  It records fresh review of bytes which were
already committed before the production provider route existed; it never
claims who historically authored those bytes and cannot itself complete a
task or satisfy a proof gate.

The operator pins both the policy file and Ed25519 key outside task metadata.
This module only handles the public signed envelope.  Repository inspection,
manifest construction, provider invocation, and validation live in
``legacy_landed_review``.
"""

from __future__ import annotations

import base64
import errno
import json
import os
import secrets
import stat
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import Encoding, PublicFormat

from ..proof.formal_verification_contracts import (
    canonical_json_bytes,
    content_identity,
)

LEGACY_LANDED_REVIEW_ATTESTATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/legacy-landed-review-attestation@1"
)
LEGACY_LANDED_REVIEW_ATTESTATION_INTERFACE: Final = (
    "LegacyLandedReviewAttestation@1"
)
LEGACY_LANDED_REVIEW_SIGNATURE_ALGORITHM: Final = "Ed25519"
HISTORICAL_PROVIDER_UNVERIFIED: Final = "unverified"

_ATTESTATION_KEYS: Final = frozenset(
    {
        "schema",
        "interface",
        "attestation_id",
        "signature_algorithm",
        "issuer_key_id",
        "policy_id",
        "task_id",
        "canonical_task_key",
        "canonical_task_cid",
        "historical_provider",
        "baseline_commit",
        "interval_commits",
        "implementation_commit",
        "merge_commit",
        "current_head",
        "current_tree_id",
        "paths",
        "manifest_id",
        "manifest_merkle_root",
        "review_aggregate_id",
        "scope_adjudication_receipt_id",
        "validation_receipt_ids",
        "issued_at_ms",
        "nonce",
        "migration_evidence_only",
        "provider_execution_receipt_synthesized",
        "completion_authoritative",
        "proof_authoritative",
        "signature",
    }
)


def _mapping(value: Any) -> dict[str, Any]:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping or expose to_dict()")
    # Detach custom mappings and reject non-DAG-JSON values before signing.
    return json.loads(canonical_json_bytes(value))


def _text(value: Any) -> str:
    return str(value or "").strip()


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    encoded = _text(value)
    if not encoded:
        raise ValueError("base64url value is empty")
    return base64.b64decode(
        encoded + "=" * (-len(encoded) % 4),
        altchars=b"-_",
        validate=True,
    )


def legacy_landed_review_key_id(public_key: bytes) -> str:
    """Return the stable identifier used by an operator policy."""

    import hashlib

    if not isinstance(public_key, bytes) or len(public_key) != 32:
        raise ValueError("Ed25519 public key must contain exactly 32 bytes")
    return "ed25519:sha256:" + hashlib.sha256(public_key).hexdigest()


def _private_key_open_flags() -> int:
    return (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _read_private_key(path: str | Path) -> bytes:
    key_path = Path(path)
    if not hasattr(os, "O_NOFOLLOW") and key_path.is_symlink():
        raise ValueError("legacy landed review key cannot be a symlink")
    try:
        descriptor = os.open(key_path, _private_key_open_flags())
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError("legacy landed review key path is unsafe") from exc
        raise
    try:
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode):
            raise ValueError("legacy landed review key must be a regular file")
        if info.st_nlink != 1:
            raise ValueError("legacy landed review key cannot be hard-linked")
        if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
            raise ValueError("legacy landed review key owner is invalid")
        mode = stat.S_IMODE(info.st_mode)
        if not mode & stat.S_IRUSR or mode & (stat.S_IRWXG | stat.S_IRWXO):
            raise ValueError("legacy landed review key permissions are invalid")
        if info.st_size != 32:
            raise ValueError("legacy landed review key is invalid")
        raw = b""
        while len(raw) < 33:
            chunk = os.read(descriptor, 33 - len(raw))
            if not chunk:
                break
            raw += chunk
    finally:
        os.close(descriptor)
    if len(raw) != 32:
        raise ValueError("legacy landed review key is invalid")
    return raw


@dataclass(frozen=True, slots=True)
class LegacyLandedReviewAttestation:
    """Public Ed25519 binding for one fresh review of already-landed bytes."""

    attestation_id: str
    issuer_key_id: str
    policy_id: str
    task_id: str
    canonical_task_key: str
    canonical_task_cid: str
    baseline_commit: str
    interval_commits: tuple[str, ...]
    implementation_commit: str
    merge_commit: str
    current_head: str
    current_tree_id: str
    paths: tuple[str, ...]
    manifest_id: str
    manifest_merkle_root: str
    review_aggregate_id: str
    scope_adjudication_receipt_id: str
    validation_receipt_ids: tuple[str, ...]
    issued_at_ms: int
    nonce: str
    signature: str
    signature_algorithm: str = LEGACY_LANDED_REVIEW_SIGNATURE_ALGORITHM
    historical_provider: str = HISTORICAL_PROVIDER_UNVERIFIED

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": LEGACY_LANDED_REVIEW_ATTESTATION_SCHEMA,
            "interface": LEGACY_LANDED_REVIEW_ATTESTATION_INTERFACE,
            "signature_algorithm": self.signature_algorithm,
            "issuer_key_id": self.issuer_key_id,
            "policy_id": self.policy_id,
            "task_id": self.task_id,
            "canonical_task_key": self.canonical_task_key,
            "canonical_task_cid": self.canonical_task_cid,
            "historical_provider": self.historical_provider,
            "baseline_commit": self.baseline_commit,
            "interval_commits": list(self.interval_commits),
            "implementation_commit": self.implementation_commit,
            "merge_commit": self.merge_commit,
            "current_head": self.current_head,
            "current_tree_id": self.current_tree_id,
            "paths": list(self.paths),
            "manifest_id": self.manifest_id,
            "manifest_merkle_root": self.manifest_merkle_root,
            "review_aggregate_id": self.review_aggregate_id,
            "scope_adjudication_receipt_id": (
                self.scope_adjudication_receipt_id
            ),
            "validation_receipt_ids": list(self.validation_receipt_ids),
            "issued_at_ms": self.issued_at_ms,
            "nonce": self.nonce,
            "migration_evidence_only": True,
            "provider_execution_receipt_synthesized": False,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            **self.unsigned_dict(),
            "attestation_id": self.attestation_id,
            "signature": self.signature,
        }

    @classmethod
    def from_dict(
        cls, value: Mapping[str, Any]
    ) -> LegacyLandedReviewAttestation:
        payload = _mapping(value)
        if set(payload) != _ATTESTATION_KEYS:
            raise ValueError("legacy landed review attestation shape is invalid")
        if payload.get("schema") != LEGACY_LANDED_REVIEW_ATTESTATION_SCHEMA:
            raise ValueError("legacy landed review attestation schema is invalid")
        if payload.get("interface") != LEGACY_LANDED_REVIEW_ATTESTATION_INTERFACE:
            raise ValueError("legacy landed review attestation interface is invalid")
        if payload.get("signature_algorithm") != LEGACY_LANDED_REVIEW_SIGNATURE_ALGORITHM:
            raise ValueError("legacy landed review signature algorithm is invalid")
        fixed = {
            "historical_provider": HISTORICAL_PROVIDER_UNVERIFIED,
            "migration_evidence_only": True,
            "provider_execution_receipt_synthesized": False,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }
        if any(payload.get(key) != expected for key, expected in fixed.items()):
            raise ValueError("legacy landed review authority projection is invalid")
        required = (
            "attestation_id",
            "issuer_key_id",
            "policy_id",
            "task_id",
            "canonical_task_key",
            "canonical_task_cid",
            "baseline_commit",
            "implementation_commit",
            "merge_commit",
            "current_head",
            "current_tree_id",
            "manifest_id",
            "manifest_merkle_root",
            "review_aggregate_id",
            "nonce",
            "signature",
        )
        if any(not _text(payload.get(field)) for field in required):
            raise ValueError("legacy landed review attestation is incomplete")
        issued = payload.get("issued_at_ms")
        if isinstance(issued, bool) or not isinstance(issued, int) or issued < 1:
            raise ValueError("legacy landed review issue time is invalid")
        interval = payload.get("interval_commits")
        paths = payload.get("paths")
        validations = payload.get("validation_receipt_ids")
        for field, value in (
            ("interval_commits", interval),
            ("paths", paths),
            ("validation_receipt_ids", validations),
        ):
            if (
                not isinstance(value, Sequence)
                or isinstance(value, (str, bytes, bytearray))
                or not value
                or any(not isinstance(item, str) or not item for item in value)
            ):
                raise ValueError(f"legacy landed review {field} is invalid")
        if list(paths) != sorted(set(paths)):
            raise ValueError("legacy landed review paths are not canonical")
        if len(set(interval)) != len(interval):
            raise ValueError("legacy landed review interval is ambiguous")
        if len(set(validations)) != len(validations):
            raise ValueError("legacy landed review validations are duplicated")
        if len(_text(payload["nonce"])) < 16:
            raise ValueError("legacy landed review nonce is too short")
        return cls(
            attestation_id=_text(payload["attestation_id"]),
            issuer_key_id=_text(payload["issuer_key_id"]),
            policy_id=_text(payload["policy_id"]),
            task_id=_text(payload["task_id"]),
            canonical_task_key=_text(payload["canonical_task_key"]),
            canonical_task_cid=_text(payload["canonical_task_cid"]),
            baseline_commit=_text(payload["baseline_commit"]),
            interval_commits=tuple(interval),
            implementation_commit=_text(payload["implementation_commit"]),
            merge_commit=_text(payload["merge_commit"]),
            current_head=_text(payload["current_head"]),
            current_tree_id=_text(payload["current_tree_id"]),
            paths=tuple(paths),
            manifest_id=_text(payload["manifest_id"]),
            manifest_merkle_root=_text(payload["manifest_merkle_root"]),
            review_aggregate_id=_text(payload["review_aggregate_id"]),
            scope_adjudication_receipt_id=_text(
                payload["scope_adjudication_receipt_id"]
            ),
            validation_receipt_ids=tuple(validations),
            issued_at_ms=issued,
            nonce=_text(payload["nonce"]),
            signature=_text(payload["signature"]),
        )


class LegacyLandedReviewAuthority:
    """Operator-owned Ed25519 signer; private material is never serialized."""

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError("private_key must be an Ed25519PrivateKey")
        self._private_key = private_key

    @classmethod
    def from_private_key_path(
        cls, path: str | Path
    ) -> LegacyLandedReviewAuthority:
        return cls(Ed25519PrivateKey.from_private_bytes(_read_private_key(path)))

    @property
    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            Encoding.Raw, PublicFormat.Raw
        )

    @property
    def public_key_base64url(self) -> str:
        return _b64(self.public_key_bytes)

    @property
    def issuer_key_id(self) -> str:
        return legacy_landed_review_key_id(self.public_key_bytes)

    def issue(
        self,
        *,
        policy_id: str,
        task_id: str,
        canonical_task_key: str,
        canonical_task_cid: str,
        baseline_commit: str,
        interval_commits: Sequence[str],
        implementation_commit: str,
        merge_commit: str,
        current_head: str,
        current_tree_id: str,
        paths: Sequence[str],
        manifest_id: str,
        manifest_merkle_root: str,
        review_aggregate_id: str,
        validation_receipt_ids: Sequence[str],
        scope_adjudication_receipt_id: str = "",
        issued_at_ms: int | None = None,
        nonce: str = "",
    ) -> LegacyLandedReviewAttestation:
        issued = int(time.time() * 1000) if issued_at_ms is None else issued_at_ms
        nonce_value = _text(nonce) or secrets.token_urlsafe(24).rstrip("=")
        provisional = LegacyLandedReviewAttestation(
            attestation_id="pending",
            issuer_key_id=self.issuer_key_id,
            policy_id=_text(policy_id),
            task_id=_text(task_id),
            canonical_task_key=_text(canonical_task_key),
            canonical_task_cid=_text(canonical_task_cid),
            baseline_commit=_text(baseline_commit),
            interval_commits=tuple(_text(item) for item in interval_commits),
            implementation_commit=_text(implementation_commit),
            merge_commit=_text(merge_commit),
            current_head=_text(current_head),
            current_tree_id=_text(current_tree_id),
            paths=tuple(_text(item) for item in paths),
            manifest_id=_text(manifest_id),
            manifest_merkle_root=_text(manifest_merkle_root),
            review_aggregate_id=_text(review_aggregate_id),
            scope_adjudication_receipt_id=_text(
                scope_adjudication_receipt_id
            ),
            validation_receipt_ids=tuple(
                _text(item) for item in validation_receipt_ids
            ),
            issued_at_ms=issued,
            nonce=nonce_value,
            signature="pending",
        )
        # Reuse the strict public parser before signing any attacker-influenced
        # identifiers.  Placeholder signature/id values satisfy shape only.
        LegacyLandedReviewAttestation.from_dict(provisional.to_dict())
        unsigned = provisional.unsigned_dict()
        signature = _b64(self._private_key.sign(canonical_json_bytes(unsigned)))
        attestation_id = content_identity({**unsigned, "signature": signature})
        return LegacyLandedReviewAttestation.from_dict(
            {
                **unsigned,
                "attestation_id": attestation_id,
                "signature": signature,
            }
        )


@dataclass(frozen=True, slots=True)
class LegacyLandedReviewVerification:
    verified: bool
    reason_codes: tuple[str, ...]
    attestation_id: str = ""
    issuer_key_id: str = ""

    @property
    def admitted(self) -> bool:
        return self.verified and not self.reason_codes

    def to_dict(self) -> dict[str, Any]:
        return {
            "verified": self.verified,
            "admitted": self.admitted,
            "reason_codes": list(self.reason_codes),
            "attestation_id": self.attestation_id,
            "issuer_key_id": self.issuer_key_id,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }


def verify_legacy_landed_review_attestation(
    attestation: LegacyLandedReviewAttestation | Mapping[str, Any] | None,
    *,
    trusted_public_keys: Mapping[str, bytes | str],
    expected_policy_id: str,
    expected_task_id: str,
    expected_canonical_task_key: str,
    expected_canonical_task_cid: str,
    expected_current_head: str,
    expected_current_tree_id: str,
    manifest: Mapping[str, Any] | None = None,
    review_aggregate: Mapping[str, Any] | None = None,
    validation_receipts: Sequence[Mapping[str, Any]] = (),
    scope_adjudication_receipt: Mapping[str, Any] | None = None,
) -> LegacyLandedReviewVerification:
    """Verify signer and every supplied evidence identity fail closed."""

    if attestation is None:
        return LegacyLandedReviewVerification(
            False, ("legacy_landed_review_attestation_missing",)
        )
    try:
        parsed = (
            attestation
            if isinstance(attestation, LegacyLandedReviewAttestation)
            else LegacyLandedReviewAttestation.from_dict(attestation)
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return LegacyLandedReviewVerification(
            False, ("legacy_landed_review_attestation_invalid",)
        )
    failures: list[str] = []
    trusted = trusted_public_keys.get(parsed.issuer_key_id)
    public = b""
    if trusted is None:
        failures.append("legacy_landed_review_issuer_untrusted")
    else:
        try:
            public = trusted if isinstance(trusted, bytes) else _unb64(trusted)
        except (TypeError, ValueError):
            public = b""
        if (
            len(public) != 32
            or legacy_landed_review_key_id(public) != parsed.issuer_key_id
        ):
            failures.append("legacy_landed_review_trusted_key_invalid")
    try:
        signature = _unb64(parsed.signature)
        if public:
            Ed25519PublicKey.from_public_bytes(public).verify(
                signature, canonical_json_bytes(parsed.unsigned_dict())
            )
    except (InvalidSignature, TypeError, ValueError):
        failures.append("legacy_landed_review_signature_invalid")
    if parsed.attestation_id != content_identity(
        {**parsed.unsigned_dict(), "signature": parsed.signature}
    ):
        failures.append("legacy_landed_review_attestation_id_mismatch")
    expected = {
        "policy_id": expected_policy_id,
        "task_id": expected_task_id,
        "canonical_task_key": expected_canonical_task_key,
        "canonical_task_cid": expected_canonical_task_cid,
        "current_head": expected_current_head,
        "current_tree_id": expected_current_tree_id,
    }
    for field, value in expected.items():
        if getattr(parsed, field) != _text(value):
            failures.append(f"legacy_landed_review_{field}_mismatch")

    def evidence_id(
        value: Mapping[str, Any] | None, field: str, reason: str
    ) -> str:
        if not isinstance(value, Mapping):
            failures.append(reason + "_missing")
            return ""
        payload = _mapping(value)
        claimed = _text(payload.get(field))
        unsigned = dict(payload)
        unsigned.pop(field, None)
        if not claimed or claimed != content_identity(unsigned):
            failures.append(reason + "_content_id_mismatch")
        if payload.get("completion_authoritative") is not False:
            failures.append(reason + "_completion_authority_claim")
        if payload.get("proof_authoritative") is not False:
            failures.append(reason + "_proof_authority_claim")
        return claimed

    if manifest is not None:
        manifest_id = evidence_id(
            manifest, "manifest_id", "legacy_landed_review_manifest"
        )
        if manifest_id != parsed.manifest_id:
            failures.append("legacy_landed_review_manifest_binding_mismatch")
    else:
        failures.append("legacy_landed_review_manifest_missing")
    if review_aggregate is not None:
        aggregate_id = evidence_id(
            review_aggregate,
            "aggregate_id",
            "legacy_landed_review_aggregate",
        )
        if aggregate_id != parsed.review_aggregate_id:
            failures.append("legacy_landed_review_aggregate_binding_mismatch")
    else:
        failures.append("legacy_landed_review_aggregate_missing")
    observed_validations = tuple(
        evidence_id(item, "receipt_id", "legacy_landed_review_validation")
        for item in validation_receipts
    )
    if observed_validations != parsed.validation_receipt_ids:
        failures.append("legacy_landed_review_validation_binding_mismatch")
    if parsed.scope_adjudication_receipt_id:
        scope_id = evidence_id(
            scope_adjudication_receipt,
            "receipt_id",
            "legacy_landed_review_scope_adjudication",
        )
        if scope_id != parsed.scope_adjudication_receipt_id:
            failures.append("legacy_landed_review_scope_binding_mismatch")
    elif scope_adjudication_receipt is not None:
        failures.append("legacy_landed_review_unexpected_scope_adjudication")
    reasons = tuple(dict.fromkeys(failures))
    return LegacyLandedReviewVerification(
        verified=not reasons,
        reason_codes=reasons,
        attestation_id=parsed.attestation_id,
        issuer_key_id=parsed.issuer_key_id,
    )


__all__ = [
    "HISTORICAL_PROVIDER_UNVERIFIED",
    "LEGACY_LANDED_REVIEW_ATTESTATION_INTERFACE",
    "LEGACY_LANDED_REVIEW_ATTESTATION_SCHEMA",
    "LEGACY_LANDED_REVIEW_SIGNATURE_ALGORITHM",
    "LegacyLandedReviewAttestation",
    "LegacyLandedReviewAuthority",
    "LegacyLandedReviewVerification",
    "legacy_landed_review_key_id",
    "verify_legacy_landed_review_attestation",
]
