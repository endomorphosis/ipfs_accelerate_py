"""Signed trust boundary for production provider-review receipts.

Content identities prove that receipt fields agree with one another; they do
not prove that the supervisor observed the provider executions.  This module
adds that missing issuer boundary.  An operator-pinned Ed25519 authority signs one
exact, independently reviewed provider receipt after its applied result has an
immutable implementation commit and tree.  Completion code accepts only a
signature from an operator-pinned public key and independently reconstructs
the complete receipt and review-chain binding.

The private key is never serialized into task, queue, event, or receipt
metadata.  Public keys can be pinned on another worker for distributed
verification without sharing signing authority.
"""

from __future__ import annotations

import base64
import errno
import fcntl
import hashlib
import json
import os
import secrets
import stat
import time
from collections.abc import Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
    PublicFormat,
)

from .contract_packet_provider_router import (
    PRODUCTION_PROVIDER_ROUTE_INTERFACE,
    PRODUCTION_REVIEW_CHAIN_BINDING_SCHEMA,
    PROVIDER_EXECUTION_RECEIPT_INTERFACE,
    PROVIDER_EXECUTION_RECEIPT_SCHEMA,
    ProductionReviewChainBinding,
    ProviderExecutionReceipt,
    ProviderReason,
    ProviderRole,
    ReviewPresence,
    RouteStatus,
    _packet_content_id,
    review_chain_content_digest,
)
from .production_reviewed_effect import (
    ProductionReviewedEffectBinding,
    verify_finalized_production_reviewed_effect,
)

PRODUCTION_PROVIDER_REVIEW_ATTESTATION_SCHEMA: Final = (
    "ipfs_accelerate_py/agent-supervisor/production-provider-review-attestation@1"
)
PRODUCTION_PROVIDER_REVIEW_ATTESTATION_INTERFACE: Final = (
    "ProductionProviderReviewAttestation@1"
)
PRODUCTION_PROVIDER_REVIEW_SIGNATURE_ALGORITHM: Final = "Ed25519"
DEFAULT_PRODUCTION_PROVIDER_REVIEW_KEY_NAME: Final = (
    ".production-provider-review-authority.ed25519"
)
_PRODUCTION_PROVIDER_REVIEW_KEY_LOCK_SUFFIX: Final = ".lock"

_RECEIPT_KEYS: Final = frozenset(
    {
        "schema",
        "interface",
        "receipt_id",
        "status",
        "reason_code",
        "provider",
        "packet",
        "review_chain",
        "review_presence",
        "admission",
        "attempts",
        "writer_lease_id",
        "write_performed",
        "fallback",
        "selected_proposal_digest",
        "implementation_proposal_digest",
        "review_proposal_digest",
        "proof_authoritative",
        "completion_authoritative",
    }
)
_BINDING_KEYS: Final = frozenset(
    {
        "schema",
        "interface",
        "receipt_id",
        "task_id",
        "packet_id",
        "packet_cid",
        "snapshot_id",
        "review_chain_digest",
        "selected_proposal_digest",
        "implementation_proposal_digest",
        "review_proposal_digest",
        "writer_lease_id",
        "write_performed",
        "review_presence",
        "provider_result_admitted",
        "implementation_commit",
        "merge_commit",
        "disposition",
        "completion_authoritative",
        "proof_authoritative",
    }
)
_ATTESTATION_KEYS: Final = frozenset(
    {
        "schema",
        "interface",
        "attestation_id",
        "signature_algorithm",
        "issuer_key_id",
        "provider_policy_id",
        "provider_receipt_cid",
        "reviewed_effect_binding_cid",
        "task_id",
        "snapshot_id",
        "packet_id",
        "packet_cid",
        "review_chain_digest",
        "selected_proposal_digest",
        "implementation_proposal_digest",
        "review_proposal_digest",
        "writer_lease_id",
        "write_performed",
        "implementation_commit",
        "implementation_tree_id",
        "issued_at_ms",
        "nonce",
        "signature",
        "completion_authoritative",
        "proof_authoritative",
    }
)


def _canonical_json_bytes(value: Mapping[str, Any]) -> bytes:
    return json.dumps(
        dict(value),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _b64(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _unb64(value: str) -> bytes:
    text = str(value or "").strip()
    if not text:
        raise ValueError("base64url value is empty")
    return base64.b64decode(
        text + "=" * (-len(text) % 4),
        altchars=b"-_",
        validate=True,
    )


def _key_id(public_key: bytes) -> str:
    return "ed25519:sha256:" + hashlib.sha256(public_key).hexdigest()


def _private_key_open_flags(base: int) -> int:
    return (
        base
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )


def _validate_private_key_descriptor(
    descriptor: int,
    *,
    require_complete_key: bool,
) -> None:
    info = os.fstat(descriptor)
    if not stat.S_ISREG(info.st_mode):
        raise ValueError("production provider Ed25519 key must be a regular file")
    if info.st_nlink != 1:
        raise ValueError("production provider Ed25519 key cannot be hard-linked")
    if hasattr(os, "geteuid") and info.st_uid != os.geteuid():
        raise ValueError("production provider Ed25519 key owner is invalid")
    mode = stat.S_IMODE(info.st_mode)
    if not mode & stat.S_IRUSR or mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise ValueError("production provider Ed25519 key permissions are invalid")
    if require_complete_key and info.st_size != 32:
        raise ValueError("production provider Ed25519 key is invalid")


@contextmanager
def _private_key_file_lock(path: str | Path, *, exclusive: bool):
    key_path = Path(path)
    lock_path = key_path.with_name(
        key_path.name + _PRODUCTION_PROVIDER_REVIEW_KEY_LOCK_SUFFIX
    )
    if lock_path.parent.is_symlink():
        raise ValueError(
            "production provider Ed25519 key parent cannot be a symlink"
        )
    try:
        descriptor = os.open(
            lock_path,
            _private_key_open_flags(os.O_RDWR | os.O_CREAT),
            0o600,
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                "production provider Ed25519 key lock path is unsafe"
            ) from exc
        raise
    try:
        _validate_private_key_descriptor(
            descriptor,
            require_complete_key=False,
        )
        fcntl.flock(
            descriptor,
            fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH,
        )
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _read_private_key_bytes_unlocked(path: str | Path) -> bytes:
    key_path = Path(path)
    if not hasattr(os, "O_NOFOLLOW") and key_path.is_symlink():
        raise ValueError("production provider Ed25519 key cannot be a symlink")
    try:
        descriptor = os.open(
            key_path,
            _private_key_open_flags(os.O_RDONLY),
        )
    except OSError as exc:
        if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise ValueError(
                "production provider Ed25519 key path is unsafe"
            ) from exc
        raise
    try:
        _validate_private_key_descriptor(
            descriptor,
            require_complete_key=True,
        )
        chunks: list[bytes] = []
        remaining = 33
        while remaining:
            chunk = os.read(descriptor, remaining)
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        raw = b"".join(chunks)
    finally:
        os.close(descriptor)
    if len(raw) != 32:
        raise ValueError("production provider Ed25519 key is invalid")
    return raw


def _read_private_key_bytes(path: str | Path) -> bytes:
    with _private_key_file_lock(path, exclusive=False):
        return _read_private_key_bytes_unlocked(path)


def _write_complete_private_key(descriptor: int, raw: bytes) -> None:
    view = memoryview(raw)
    written = 0
    while written < len(view):
        count = os.write(descriptor, view[written:])
        if count <= 0:
            raise OSError("production provider Ed25519 key write was incomplete")
        written += count
    os.fsync(descriptor)


def _mapping(value: Any) -> dict[str, Any]:
    converter = getattr(value, "to_dict", None)
    if callable(converter):
        value = converter()
    if not isinstance(value, Mapping):
        raise TypeError("value must be a mapping or expose to_dict()")
    # Enforce JSON detachment now so custom mappings and non-finite values do
    # not acquire a different representation after signing.
    return json.loads(_canonical_json_bytes(value))


def _text(value: Any) -> str:
    return str(value or "").strip()


@dataclass(frozen=True, slots=True)
class ProductionProviderReviewVerification:
    """Independent verification result; false results never grant a gate."""

    verified: bool
    reason_codes: tuple[str, ...]
    attestation_id: str = ""
    provider_receipt_cid: str = ""
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
            "provider_receipt_cid": self.provider_receipt_cid,
            "issuer_key_id": self.issuer_key_id,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }


def _receipt_failures(
    value: ProviderExecutionReceipt | Mapping[str, Any],
    *,
    expected_task_id: str,
    expected_snapshot_id: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    """Reconstruct all completion-relevant receipt semantics fail closed."""

    try:
        payload = _mapping(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}, ("provider_receipt_invalid",)
    failures: list[str] = []
    if set(payload) != _RECEIPT_KEYS:
        failures.append("provider_receipt_shape_invalid")
    if payload.get("schema") != PROVIDER_EXECUTION_RECEIPT_SCHEMA:
        failures.append("provider_receipt_schema_invalid")
    if payload.get("interface") != PROVIDER_EXECUTION_RECEIPT_INTERFACE:
        failures.append("provider_receipt_interface_invalid")
    claimed_receipt_id = _text(payload.get("receipt_id"))
    unsigned = dict(payload)
    unsigned.pop("receipt_id", None)
    try:
        computed_receipt_id = _packet_content_id(unsigned)
    except (TypeError, ValueError):
        computed_receipt_id = ""
    if not claimed_receipt_id or claimed_receipt_id != computed_receipt_id:
        failures.append("provider_receipt_content_id_mismatch")
    if payload.get("status") != RouteStatus.SUCCEEDED.value:
        failures.append("provider_route_not_succeeded")
    if payload.get("reason_code") != ProviderReason.ROUTED.value:
        failures.append("provider_route_reason_not_admitted")
    if payload.get("provider") not in {
        ProviderRole.GROK_IMPLEMENT.value,
        ProviderRole.CODEX_REVIEW.value,
    }:
        failures.append("provider_route_provider_invalid")
    if payload.get("fallback") is not False:
        failures.append("provider_route_fallback_forbidden")
    if payload.get("write_performed") is not True:
        failures.append("provider_route_write_missing")
    if not _text(payload.get("writer_lease_id")):
        failures.append("provider_route_writer_lease_missing")
    if payload.get("completion_authoritative") is not False:
        failures.append("provider_receipt_completion_authority_claim")
    if payload.get("proof_authoritative") is not False:
        failures.append("provider_receipt_proof_authority_claim")

    packet = payload.get("packet")
    packet = dict(packet) if isinstance(packet, Mapping) else {}
    if _text(packet.get("task_id")) != _text(expected_task_id):
        failures.append("provider_receipt_task_mismatch")
    if _text(packet.get("snapshot_id")) != _text(expected_snapshot_id):
        failures.append("provider_receipt_snapshot_mismatch")
    if not _text(packet.get("packet_id")) or not _text(packet.get("packet_cid")):
        failures.append("provider_receipt_packet_identity_missing")
    packet_bytes = packet.get("packet_bytes")
    if (
        isinstance(packet_bytes, bool)
        or not isinstance(packet_bytes, int)
        or packet_bytes < 1
    ):
        failures.append("provider_receipt_packet_size_invalid")

    admission = payload.get("admission")
    admission = dict(admission) if isinstance(admission, Mapping) else {}
    expected_admission = {
        "proposal_only": True,
        "repository_write_allowed": True,
        "completion_authoritative": False,
        "proof_authoritative": False,
        "provider_result_admitted": True,
        "independent_review": True,
        "review_presence": ReviewPresence.INDEPENDENT.value,
        "self_review": False,
        "writer_lease_bound": True,
    }
    if admission != expected_admission:
        failures.append("provider_receipt_admission_invalid")
    if payload.get("review_presence") != ReviewPresence.INDEPENDENT.value:
        failures.append("provider_review_not_independent")

    chain = payload.get("review_chain")
    chain = list(chain) if isinstance(chain, Sequence) and not isinstance(
        chain, (str, bytes, bytearray)
    ) else []
    attempts = payload.get("attempts")
    attempts = list(attempts) if isinstance(attempts, Sequence) and not isinstance(
        attempts, (str, bytes, bytearray)
    ) else []
    expected_roles = (
        ProviderRole.GROK_IMPLEMENT.value,
        ProviderRole.CODEX_REVIEW.value,
    )
    if len(chain) != 2 or any(not isinstance(item, Mapping) for item in chain):
        failures.append("provider_review_chain_shape_invalid")
    else:
        for index, role in enumerate(expected_roles):
            step = dict(chain[index])
            if (
                step.get("role") != role
                or step.get("status") != "succeeded"
                or step.get("admitted") is not True
                or not _text(step.get("response_digest"))
            ):
                failures.append(f"provider_review_chain_step_{index}_invalid")
    if len(attempts) != 2 or any(
        not isinstance(item, Mapping) for item in attempts
    ):
        failures.append("provider_attempt_chain_shape_invalid")
    else:
        for index, role in enumerate(expected_roles):
            attempt = dict(attempts[index])
            step = dict(chain[index]) if len(chain) == 2 else {}
            if (
                attempt.get("role") != role
                or attempt.get("status") != "succeeded"
                or not _text(attempt.get("prompt_digest"))
                or attempt.get("response_digest") != step.get("response_digest")
                or attempt.get("prompt_embedded") is not False
                or attempt.get("response_embedded") is not False
            ):
                failures.append(f"provider_attempt_chain_step_{index}_invalid")

    implementation_digest = _text(payload.get("implementation_proposal_digest"))
    review_digest = _text(payload.get("review_proposal_digest"))
    selected_digest = _text(payload.get("selected_proposal_digest"))
    if len(chain) == 2:
        if implementation_digest != _text(chain[0].get("response_digest")):
            failures.append("implementation_proposal_digest_mismatch")
        if review_digest != _text(chain[1].get("response_digest")):
            failures.append("review_proposal_digest_mismatch")
    if not selected_digest or selected_digest != implementation_digest:
        failures.append("selected_proposal_digest_invalid")
    return payload, tuple(dict.fromkeys(failures))


def _binding_failures(
    value: ProductionReviewChainBinding | Mapping[str, Any],
    *,
    receipt: Mapping[str, Any],
    expected_implementation_commit: str,
) -> tuple[dict[str, Any], tuple[str, ...]]:
    try:
        payload = _mapping(value)
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}, ("provider_review_binding_invalid",)
    failures: list[str] = []
    if set(payload) != _BINDING_KEYS:
        failures.append("provider_review_binding_shape_invalid")
    if payload.get("schema") != PRODUCTION_REVIEW_CHAIN_BINDING_SCHEMA:
        failures.append("provider_review_binding_schema_invalid")
    if payload.get("interface") != PRODUCTION_PROVIDER_ROUTE_INTERFACE:
        failures.append("provider_review_binding_interface_invalid")
    packet = receipt.get("packet")
    packet = dict(packet) if isinstance(packet, Mapping) else {}
    expected = {
        "receipt_id": receipt.get("receipt_id"),
        "task_id": packet.get("task_id"),
        "packet_id": packet.get("packet_id"),
        "packet_cid": packet.get("packet_cid"),
        "snapshot_id": packet.get("snapshot_id"),
        "review_chain_digest": review_chain_content_digest(
            receipt.get("review_chain") or ()
        ),
        "selected_proposal_digest": receipt.get("selected_proposal_digest"),
        "implementation_proposal_digest": receipt.get(
            "implementation_proposal_digest"
        ),
        "review_proposal_digest": receipt.get("review_proposal_digest"),
        "writer_lease_id": receipt.get("writer_lease_id"),
        "write_performed": True,
        "review_presence": ReviewPresence.INDEPENDENT.value,
        "provider_result_admitted": True,
        "disposition": "admitted",
        "completion_authoritative": False,
        "proof_authoritative": False,
    }
    for key, expected_value in expected.items():
        if payload.get(key) != expected_value:
            failures.append(f"provider_review_binding_mismatch:{key}")
    binding_commit = _text(payload.get("implementation_commit"))
    expected_commit = _text(expected_implementation_commit)
    if not binding_commit or not expected_commit:
        failures.append("provider_review_implementation_commit_missing")
    elif binding_commit != expected_commit:
        failures.append("provider_review_binding_mismatch:implementation_commit")
    # The attestation is issued before integration.  A merge identity in this
    # pre-merge binding would be an unverified authority claim.
    if _text(payload.get("merge_commit")):
        failures.append("provider_review_binding_premature_merge_claim")
    return payload, tuple(dict.fromkeys(failures))


def _reviewed_effect_failures(
    value: ProductionReviewedEffectBinding | Mapping[str, Any] | None,
    *,
    repo_root: str | Path | None,
    task: Any,
    task_identity: Any,
    receipt: Mapping[str, Any],
    review_binding: Mapping[str, Any],
    expected_provider_policy_id: str,
    expected_implementation_commit: str,
    expected_implementation_tree_id: str,
) -> tuple[ProductionReviewedEffectBinding | None, tuple[str, ...]]:
    """Reconstruct the immutable proposal-to-Git effect, never queue metadata."""

    if value is None:
        return None, ("provider_reviewed_effect_missing",)
    if repo_root is None or task is None or task_identity is None:
        return None, ("provider_reviewed_effect_authoritative_context_missing",)
    try:
        effect = (
            value
            if isinstance(value, ProductionReviewedEffectBinding)
            else ProductionReviewedEffectBinding.from_dict(value)
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return None, ("provider_reviewed_effect_invalid",)
    verification = verify_finalized_production_reviewed_effect(
        effect,
        repo_root=repo_root,
        task=task,
        task_identity=task_identity,
        expected_implementation_commit=expected_implementation_commit,
        expected_implementation_tree_id=expected_implementation_tree_id,
    )
    failures = list(verification.reason_codes)
    expected = {
        "provider_policy_id": expected_provider_policy_id,
        "provider_receipt_cid": receipt.get("receipt_id"),
        "packet_task_id": review_binding.get("task_id"),
        "snapshot_id": review_binding.get("snapshot_id"),
        "packet_id": review_binding.get("packet_id"),
        "packet_cid": review_binding.get("packet_cid"),
        "review_chain_digest": review_binding.get("review_chain_digest"),
        "selected_proposal_digest": review_binding.get(
            "selected_proposal_digest"
        ),
        "implementation_proposal_digest": review_binding.get(
            "implementation_proposal_digest"
        ),
        "review_proposal_digest": review_binding.get("review_proposal_digest"),
        "writer_lease_id": review_binding.get("writer_lease_id"),
        "implementation_commit": expected_implementation_commit,
        "implementation_tree_id": expected_implementation_tree_id,
    }
    for field_name, expected_value in expected.items():
        if getattr(effect, field_name) != expected_value:
            failures.append(f"provider_reviewed_effect_mismatch:{field_name}")
    return effect, tuple(dict.fromkeys(failures))


@dataclass(frozen=True, slots=True)
class ProductionProviderReviewAttestation:
    """Public, signed binding from provider execution to candidate commit."""

    attestation_id: str
    issuer_key_id: str
    provider_policy_id: str
    provider_receipt_cid: str
    reviewed_effect_binding_cid: str
    task_id: str
    snapshot_id: str
    packet_id: str
    packet_cid: str
    review_chain_digest: str
    selected_proposal_digest: str
    implementation_proposal_digest: str
    review_proposal_digest: str
    writer_lease_id: str
    write_performed: bool
    implementation_commit: str
    implementation_tree_id: str
    issued_at_ms: int
    nonce: str
    signature: str
    signature_algorithm: str = PRODUCTION_PROVIDER_REVIEW_SIGNATURE_ALGORITHM

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "schema": PRODUCTION_PROVIDER_REVIEW_ATTESTATION_SCHEMA,
            "interface": PRODUCTION_PROVIDER_REVIEW_ATTESTATION_INTERFACE,
            "signature_algorithm": self.signature_algorithm,
            "issuer_key_id": self.issuer_key_id,
            "provider_policy_id": self.provider_policy_id,
            "provider_receipt_cid": self.provider_receipt_cid,
            "reviewed_effect_binding_cid": self.reviewed_effect_binding_cid,
            "task_id": self.task_id,
            "snapshot_id": self.snapshot_id,
            "packet_id": self.packet_id,
            "packet_cid": self.packet_cid,
            "review_chain_digest": self.review_chain_digest,
            "selected_proposal_digest": self.selected_proposal_digest,
            "implementation_proposal_digest": self.implementation_proposal_digest,
            "review_proposal_digest": self.review_proposal_digest,
            "writer_lease_id": self.writer_lease_id if self.write_performed else "",
            "write_performed": self.write_performed,
            "implementation_commit": self.implementation_commit,
            "implementation_tree_id": self.implementation_tree_id,
            "issued_at_ms": self.issued_at_ms,
            "nonce": self.nonce,
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
    ) -> ProductionProviderReviewAttestation:
        payload = _mapping(value)
        if set(payload) != _ATTESTATION_KEYS:
            raise ValueError("production provider attestation shape is invalid")
        if payload.get("schema") != PRODUCTION_PROVIDER_REVIEW_ATTESTATION_SCHEMA:
            raise ValueError("production provider attestation schema is invalid")
        if payload.get("interface") != PRODUCTION_PROVIDER_REVIEW_ATTESTATION_INTERFACE:
            raise ValueError("production provider attestation interface is invalid")
        if payload.get("signature_algorithm") != (
            PRODUCTION_PROVIDER_REVIEW_SIGNATURE_ALGORITHM
        ):
            raise ValueError("production provider attestation algorithm is invalid")
        if payload.get("completion_authoritative") is not False:
            raise ValueError("provider attestation cannot claim completion authority")
        if payload.get("proof_authoritative") is not False:
            raise ValueError("provider attestation cannot claim proof authority")
        issued = payload.get("issued_at_ms")
        if isinstance(issued, bool) or not isinstance(issued, int) or issued < 1:
            raise ValueError("production provider attestation time is invalid")
        if payload.get("write_performed") is not True:
            raise ValueError("production provider attestation requires an applied write")
        required_text = (
            "attestation_id",
            "issuer_key_id",
            "provider_policy_id",
            "provider_receipt_cid",
            "reviewed_effect_binding_cid",
            "task_id",
            "snapshot_id",
            "packet_id",
            "packet_cid",
            "review_chain_digest",
            "selected_proposal_digest",
            "implementation_proposal_digest",
            "review_proposal_digest",
            "writer_lease_id",
            "implementation_commit",
            "implementation_tree_id",
            "nonce",
            "signature",
        )
        if any(not _text(payload.get(key)) for key in required_text):
            raise ValueError("production provider attestation is incomplete")
        if len(_text(payload.get("nonce"))) < 16:
            raise ValueError("production provider attestation nonce is too short")
        return cls(
            attestation_id=_text(payload["attestation_id"]),
            issuer_key_id=_text(payload["issuer_key_id"]),
            provider_policy_id=_text(payload["provider_policy_id"]),
            provider_receipt_cid=_text(payload["provider_receipt_cid"]),
            reviewed_effect_binding_cid=_text(
                payload["reviewed_effect_binding_cid"]
            ),
            task_id=_text(payload["task_id"]),
            snapshot_id=_text(payload["snapshot_id"]),
            packet_id=_text(payload["packet_id"]),
            packet_cid=_text(payload["packet_cid"]),
            review_chain_digest=_text(payload["review_chain_digest"]),
            selected_proposal_digest=_text(payload["selected_proposal_digest"]),
            implementation_proposal_digest=_text(
                payload["implementation_proposal_digest"]
            ),
            review_proposal_digest=_text(payload["review_proposal_digest"]),
            writer_lease_id=_text(payload["writer_lease_id"]),
            write_performed=True,
            implementation_commit=_text(payload["implementation_commit"]),
            implementation_tree_id=_text(payload["implementation_tree_id"]),
            issued_at_ms=issued,
            nonce=_text(payload["nonce"]),
            signature=_text(payload["signature"]),
            signature_algorithm=PRODUCTION_PROVIDER_REVIEW_SIGNATURE_ALGORITHM,
        )


class ProductionProviderReviewAuthority:
    """Operator-controlled Ed25519 issuer for observed provider routes."""

    def __init__(self, private_key: Ed25519PrivateKey) -> None:
        if not isinstance(private_key, Ed25519PrivateKey):
            raise TypeError("private_key must be an Ed25519PrivateKey")
        self._private_key = private_key

    @classmethod
    def generate(cls) -> ProductionProviderReviewAuthority:
        return cls(Ed25519PrivateKey.generate())

    @classmethod
    def load_or_create(
        cls, path: str | Path
    ) -> ProductionProviderReviewAuthority:
        """Atomically load/create a non-linked mode-0600 development key."""

        key_path = Path(path)
        key_path.parent.mkdir(parents=True, exist_ok=True)
        if key_path.parent.is_symlink():
            raise ValueError(
                "production provider Ed25519 key parent cannot be a symlink"
            )
        with _private_key_file_lock(key_path, exclusive=True):
            try:
                raw = _read_private_key_bytes_unlocked(key_path)
            except FileNotFoundError:
                generated = Ed25519PrivateKey.generate().private_bytes(
                    Encoding.Raw,
                    PrivateFormat.Raw,
                    NoEncryption(),
                )
                temporary_path = key_path.with_name(
                    f".{key_path.name}.{os.getpid()}."
                    f"{secrets.token_hex(12)}.tmp"
                )
                try:
                    descriptor = os.open(
                        temporary_path,
                        _private_key_open_flags(
                            os.O_WRONLY | os.O_CREAT | os.O_EXCL
                        ),
                        0o600,
                    )
                    try:
                        _validate_private_key_descriptor(
                            descriptor,
                            require_complete_key=False,
                        )
                        _write_complete_private_key(descriptor, generated)
                        _validate_private_key_descriptor(
                            descriptor,
                            require_complete_key=True,
                        )
                    finally:
                        os.close(descriptor)
                    try:
                        os.link(
                            temporary_path,
                            key_path,
                            follow_symlinks=False,
                        )
                    except FileExistsError:
                        pass
                    finally:
                        temporary_path.unlink(missing_ok=True)
                except OSError as exc:
                    temporary_path.unlink(missing_ok=True)
                    if exc.errno in {errno.ELOOP, errno.ENOTDIR}:
                        raise ValueError(
                            "production provider Ed25519 key path is unsafe"
                        ) from exc
                    raise
                raw = _read_private_key_bytes_unlocked(key_path)
        if len(raw) != 32:
            raise ValueError("production provider Ed25519 key is invalid")
        return cls(Ed25519PrivateKey.from_private_bytes(raw))

    @property
    def public_key_bytes(self) -> bytes:
        return self._private_key.public_key().public_bytes(
            Encoding.Raw,
            PublicFormat.Raw,
        )

    @property
    def public_key_base64url(self) -> str:
        return _b64(self.public_key_bytes)

    @property
    def issuer_key_id(self) -> str:
        return _key_id(self.public_key_bytes)

    def issue(
        self,
        *,
        provider_receipt: ProviderExecutionReceipt | Mapping[str, Any],
        review_chain_binding: ProductionReviewChainBinding | Mapping[str, Any],
        provider_policy_id: str,
        implementation_commit: str,
        implementation_tree_id: str,
        reviewed_effect_binding: (
            ProductionReviewedEffectBinding | Mapping[str, Any] | None
        ) = None,
        repo_root: str | Path | None = None,
        task: Any = None,
        task_identity: Any = None,
        issued_at_ms: int | None = None,
        nonce: str = "",
    ) -> ProductionProviderReviewAttestation:
        """Sign a fully reconstructed, applied, commit-bound provider route."""

        binding_probe = _mapping(review_chain_binding)
        expected_task_id = _text(binding_probe.get("task_id"))
        expected_snapshot_id = _text(binding_probe.get("snapshot_id"))
        receipt, receipt_failures = _receipt_failures(
            provider_receipt,
            expected_task_id=expected_task_id,
            expected_snapshot_id=expected_snapshot_id,
        )
        binding, binding_failures = _binding_failures(
            binding_probe,
            receipt=receipt,
            expected_implementation_commit=implementation_commit,
        )
        policy_id = _text(provider_policy_id)
        commit = _text(implementation_commit)
        tree_id = _text(implementation_tree_id)
        if not policy_id or not commit or not tree_id:
            raise ValueError("provider policy, implementation commit, and tree are required")
        effect, effect_failures = _reviewed_effect_failures(
            reviewed_effect_binding,
            repo_root=repo_root,
            task=task,
            task_identity=task_identity,
            receipt=receipt,
            review_binding=binding,
            expected_provider_policy_id=policy_id,
            expected_implementation_commit=commit,
            expected_implementation_tree_id=tree_id,
        )
        failures = (*receipt_failures, *binding_failures, *effect_failures)
        if failures or effect is None:
            raise ValueError(
                "provider route cannot be attested: " + ",".join(failures)
            )
        issued = int(time.time() * 1000) if issued_at_ms is None else issued_at_ms
        if isinstance(issued, bool) or not isinstance(issued, int) or issued < 1:
            raise ValueError("issued_at_ms must be a positive integer")
        nonce_value = _text(nonce) or secrets.token_urlsafe(24).rstrip("=")
        if len(nonce_value) < 16:
            raise ValueError("nonce must contain at least 16 characters")
        unsigned = {
            "schema": PRODUCTION_PROVIDER_REVIEW_ATTESTATION_SCHEMA,
            "interface": PRODUCTION_PROVIDER_REVIEW_ATTESTATION_INTERFACE,
            "signature_algorithm": PRODUCTION_PROVIDER_REVIEW_SIGNATURE_ALGORITHM,
            "issuer_key_id": self.issuer_key_id,
            "provider_policy_id": policy_id,
            "provider_receipt_cid": receipt["receipt_id"],
            "reviewed_effect_binding_cid": effect.binding_id,
            "task_id": expected_task_id,
            "snapshot_id": expected_snapshot_id,
            "packet_id": binding["packet_id"],
            "packet_cid": binding["packet_cid"],
            "review_chain_digest": binding["review_chain_digest"],
            "selected_proposal_digest": binding["selected_proposal_digest"],
            "implementation_proposal_digest": binding[
                "implementation_proposal_digest"
            ],
            "review_proposal_digest": binding["review_proposal_digest"],
            "writer_lease_id": binding["writer_lease_id"],
            "write_performed": True,
            "implementation_commit": commit,
            "implementation_tree_id": tree_id,
            "issued_at_ms": issued,
            "nonce": nonce_value,
            "completion_authoritative": False,
            "proof_authoritative": False,
        }
        signature = _b64(self._private_key.sign(_canonical_json_bytes(unsigned)))
        attestation_id = _packet_content_id(
            {**unsigned, "signature": signature}
        )
        return ProductionProviderReviewAttestation.from_dict(
            {
                **unsigned,
                "attestation_id": attestation_id,
                "signature": signature,
            }
        )


def production_provider_review_key_path(state_path: str | Path) -> Path:
    """Return the operator key path adjacent to a daemon state file/dir."""

    path = Path(state_path)
    parent = path if path.suffix == "" and not path.exists() else path.parent
    return parent / DEFAULT_PRODUCTION_PROVIDER_REVIEW_KEY_NAME


def trusted_public_key_from_private_path(path: str | Path) -> tuple[str, bytes]:
    """Load, but never create, the trusted public key for verification."""

    raw = _read_private_key_bytes(path)
    private_key = Ed25519PrivateKey.from_private_bytes(raw)
    public = private_key.public_key().public_bytes(Encoding.Raw, PublicFormat.Raw)
    return _key_id(public), public


def verify_production_provider_review_attestation(
    attestation: ProductionProviderReviewAttestation | Mapping[str, Any] | None,
    *,
    trusted_public_keys: Mapping[str, bytes | str],
    provider_receipt: ProviderExecutionReceipt | Mapping[str, Any] | None,
    review_chain_binding: ProductionReviewChainBinding | Mapping[str, Any] | None,
    expected_task_id: str,
    expected_implementation_commit: str,
    expected_implementation_tree_id: str,
    reviewed_effect_binding: (
        ProductionReviewedEffectBinding | Mapping[str, Any] | None
    ) = None,
    repo_root: str | Path | None = None,
    task: Any = None,
    task_identity: Any = None,
    expected_snapshot_id: str = "",
    expected_provider_policy_id: str = "",
) -> ProductionProviderReviewVerification:
    """Verify issuer, full receipt, binding, and exact candidate identity.

    The attestation's embedded issuer ID is only a lookup key.  Callers must
    provide an independently pinned public-key mapping; metadata cannot install
    its own signer.
    """

    failures: list[str] = []
    if attestation is None:
        return ProductionProviderReviewVerification(
            False, ("provider_review_attestation_missing",)
        )
    try:
        parsed = (
            attestation
            if isinstance(attestation, ProductionProviderReviewAttestation)
            else ProductionProviderReviewAttestation.from_dict(attestation)
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return ProductionProviderReviewVerification(
            False, ("provider_review_attestation_invalid",)
        )

    trusted_value = trusted_public_keys.get(parsed.issuer_key_id)
    if trusted_value is None:
        failures.append("provider_review_issuer_untrusted")
        public_key = b""
    else:
        try:
            public_key = (
                trusted_value
                if isinstance(trusted_value, bytes)
                else _unb64(trusted_value)
            )
        except (TypeError, ValueError):
            public_key = b""
        if len(public_key) != 32 or _key_id(public_key) != parsed.issuer_key_id:
            failures.append("provider_review_trusted_key_invalid")
    signed = parsed.unsigned_dict()
    try:
        signature = _unb64(parsed.signature)
        if public_key:
            Ed25519PublicKey.from_public_bytes(public_key).verify(
                signature,
                _canonical_json_bytes(signed),
            )
    except (InvalidSignature, TypeError, ValueError):
        failures.append("provider_review_signature_invalid")
    expected_attestation_id = _packet_content_id(
        {**signed, "signature": parsed.signature}
    )
    if parsed.attestation_id != expected_attestation_id:
        failures.append("provider_review_attestation_id_mismatch")

    task_id = _text(expected_task_id)
    implementation_commit = _text(expected_implementation_commit)
    tree_id = _text(expected_implementation_tree_id)
    if parsed.task_id != task_id:
        failures.append("provider_review_task_mismatch")
    if parsed.implementation_commit != implementation_commit:
        failures.append("provider_review_implementation_commit_mismatch")
    if parsed.implementation_tree_id != tree_id:
        failures.append("provider_review_implementation_tree_mismatch")
    if expected_snapshot_id and parsed.snapshot_id != _text(expected_snapshot_id):
        failures.append("provider_review_snapshot_mismatch")
    if (
        expected_provider_policy_id
        and parsed.provider_policy_id != _text(expected_provider_policy_id)
    ):
        failures.append("provider_review_policy_mismatch")

    if provider_receipt is None:
        receipt: dict[str, Any] = {}
        failures.append("provider_execution_receipt_missing")
    else:
        receipt, receipt_failures = _receipt_failures(
            provider_receipt,
            expected_task_id=task_id,
            expected_snapshot_id=parsed.snapshot_id,
        )
        failures.extend(receipt_failures)
    if review_chain_binding is None:
        binding: dict[str, Any] = {}
        failures.append("provider_review_binding_missing")
    else:
        binding, binding_failures = _binding_failures(
            review_chain_binding,
            receipt=receipt,
            expected_implementation_commit=implementation_commit,
        )
        failures.extend(binding_failures)
    effect, effect_failures = _reviewed_effect_failures(
        reviewed_effect_binding,
        repo_root=repo_root,
        task=task,
        task_identity=task_identity,
        receipt=receipt,
        review_binding=binding,
        expected_provider_policy_id=parsed.provider_policy_id,
        expected_implementation_commit=implementation_commit,
        expected_implementation_tree_id=tree_id,
    )
    failures.extend(effect_failures)

    attestation_binding = {
        "provider_receipt_cid": receipt.get("receipt_id"),
        "reviewed_effect_binding_cid": effect.binding_id if effect else None,
        "task_id": binding.get("task_id"),
        "snapshot_id": binding.get("snapshot_id"),
        "packet_id": binding.get("packet_id"),
        "packet_cid": binding.get("packet_cid"),
        "review_chain_digest": binding.get("review_chain_digest"),
        "selected_proposal_digest": binding.get("selected_proposal_digest"),
        "implementation_proposal_digest": binding.get(
            "implementation_proposal_digest"
        ),
        "review_proposal_digest": binding.get("review_proposal_digest"),
        "writer_lease_id": binding.get("writer_lease_id"),
        "write_performed": binding.get("write_performed"),
    }
    for key, value in attestation_binding.items():
        if getattr(parsed, key) != value:
            failures.append(f"provider_review_attestation_binding_mismatch:{key}")

    reasons = tuple(dict.fromkeys(failures))
    return ProductionProviderReviewVerification(
        verified=not reasons,
        reason_codes=reasons,
        attestation_id=parsed.attestation_id,
        provider_receipt_cid=parsed.provider_receipt_cid,
        issuer_key_id=parsed.issuer_key_id,
    )


__all__ = [
    "DEFAULT_PRODUCTION_PROVIDER_REVIEW_KEY_NAME",
    "PRODUCTION_PROVIDER_REVIEW_ATTESTATION_INTERFACE",
    "PRODUCTION_PROVIDER_REVIEW_ATTESTATION_SCHEMA",
    "PRODUCTION_PROVIDER_REVIEW_SIGNATURE_ALGORITHM",
    "ProductionProviderReviewAttestation",
    "ProductionProviderReviewAuthority",
    "ProductionProviderReviewVerification",
    "production_provider_review_key_path",
    "trusted_public_key_from_private_path",
    "verify_production_provider_review_attestation",
]
