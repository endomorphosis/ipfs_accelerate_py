"""Locally trusted, content-addressed runner attestations for pytest passes.

This module is intentionally independent of cache and prover providers.  A
cache can retain these immutable bytes, but cannot nominate a key or turn a
signature into skip authority.  The caller must supply the locally pinned
``RunnerTrustPolicy@1`` CID for every verification.
"""

from __future__ import annotations

import base64
import hashlib
import secrets
import time
import unicodedata
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Final

import dag_cbor
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import (
    Ed25519PrivateKey,
    Ed25519PublicKey,
)
from multiformats import CID, multihash

from ...agent_supervisor.proof.test_execution_contracts import (
    SignedTestPassReceiptV2,
    TestPassReceipt,
)


RUNNER_PASS_ATTESTATION_INTERFACE: Final = "RunnerPassAttestation@1"
RUNNER_TRUST_POLICY_INTERFACE: Final = "RunnerTrustPolicy@1"
RUNNER_KEY_INTERFACE: Final = "RunnerEd25519PublicKey@1"
ED25519_PUB_MULTICODEC: Final = 0xED
PYTEST_PASS_ATTESTATION_USAGE: Final = "pytest-pass-attestation"
ATTESTATION_DOMAIN: Final = b"ipfs-test-pass-attestation/v1\0"
MAX_ARTIFACT_BYTES: Final = 64 * 1024
MAX_TEXT: Final = 4096


class RunnerAttestationError(ValueError):
    """A malformed or non-authoritative runner-attestation artifact."""


def _uvarint(value: int) -> bytes:
    if value < 0:
        raise RunnerAttestationError("negative multicodec")
    result = bytearray()
    while value > 0x7F:
        result.append((value & 0x7F) | 0x80)
        value >>= 7
    result.append(value)
    return bytes(result)


def _clean_text(value: Any, field_name: str, *, required: bool = True) -> str:
    if not isinstance(value, str):
        raise RunnerAttestationError("%s must be text" % field_name)
    if value != value.strip() or (required and not value) or len(value) > MAX_TEXT:
        raise RunnerAttestationError("invalid %s" % field_name)
    if unicodedata.normalize("NFC", value) != value:
        raise RunnerAttestationError("%s is not NFC" % field_name)
    return value


def _positive_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise RunnerAttestationError("%s must be a non-negative integer" % field_name)
    return value


def _public_value(value: Any, *, field_name: str = "value") -> Any:
    """Reject non-DAG-CBOR and secret-shaped public values before encoding."""

    if value is None or isinstance(value, (bool, bytes, str)):
        if isinstance(value, str):
            _clean_text(value, field_name, required=False)
        if isinstance(value, bytes) and len(value) > MAX_ARTIFACT_BYTES:
            raise RunnerAttestationError("public bytes exceed bound")
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, (list, tuple)):
        if len(value) > 256:
            raise RunnerAttestationError("public sequence exceeds bound")
        return [_public_value(item, field_name=field_name) for item in value]
    if isinstance(value, Mapping):
        if len(value) > 128:
            raise RunnerAttestationError("public map exceeds bound")
        result: dict[str, Any] = {}
        for raw_key, item in value.items():
            key = _clean_text(raw_key, "public field")
            lowered = key.lower().replace("-", "_")
            if any(marker in lowered for marker in ("secret", "private", "witness", "password", "token")):
                raise RunnerAttestationError("private material is prohibited")
            result[key] = _public_value(item, field_name=key)
        return result
    raise RunnerAttestationError("%s is not a DAG-CBOR public value" % field_name)


def canonical_dag_cbor(value: Mapping[str, Any]) -> bytes:
    """Return the one accepted canonical DAG-CBOR representation.

    Decode-and-reencode catches alternative/noncanonical CBOR encodings when
    callers ingest retained bytes through :func:`decode_canonical_dag_cbor`.
    """

    normalized = _public_value(value)
    assert isinstance(normalized, dict)
    try:
        encoded = dag_cbor.encode(normalized)
    except Exception as exc:  # pragma: no cover - library dependent details
        raise RunnerAttestationError("DAG-CBOR encoding failed") from exc
    if not isinstance(encoded, bytes) or len(encoded) > MAX_ARTIFACT_BYTES:
        raise RunnerAttestationError("DAG-CBOR artifact exceeds bound")
    return encoded


def decode_canonical_dag_cbor(encoded: bytes) -> dict[str, Any]:
    if not isinstance(encoded, bytes) or not encoded or len(encoded) > MAX_ARTIFACT_BYTES:
        raise RunnerAttestationError("invalid DAG-CBOR bytes")
    try:
        decoded = dag_cbor.decode(encoded)
    except Exception as exc:
        raise RunnerAttestationError("malformed DAG-CBOR") from exc
    if not isinstance(decoded, dict) or canonical_dag_cbor(decoded) != encoded:
        raise RunnerAttestationError("DAG-CBOR is not strict canonical form")
    return decoded


def cidv1_for_bytes(data: bytes, codec: str) -> str:
    if not isinstance(data, bytes):
        raise RunnerAttestationError("CID input must be bytes")
    return CID("base32", 1, codec, multihash.digest(data, "sha2-256")).encode()


def dag_cbor_cid(value: Mapping[str, Any]) -> str:
    return cidv1_for_bytes(canonical_dag_cbor(value), "dag-cbor")


def _strict_cid(value: Any, *, codec: str | None = None) -> str:
    text = _clean_text(value, "CID")
    try:
        parsed = CID.decode(text)
    except Exception as exc:
        raise RunnerAttestationError("malformed CID") from exc
    if parsed.version != 1 or parsed.hashfun.name != "sha2-256" or parsed.encode() != text:
        raise RunnerAttestationError("CID must be canonical CIDv1/base32/sha2-256")
    if codec is not None and parsed.codec.name != codec:
        raise RunnerAttestationError("CID has wrong multicodec")
    return text


def _digest_message(unsigned_bytes: bytes) -> bytes:
    return ATTESTATION_DOMAIN + hashlib.sha256(unsigned_bytes).digest()


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).decode("ascii").rstrip("=")


def _b64decode(value: Any) -> bytes:
    text = _clean_text(value, "signature")
    if len(text) > 128:
        raise RunnerAttestationError("signature exceeds bound")
    try:
        decoded = base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
    except Exception as exc:
        raise RunnerAttestationError("malformed signature") from exc
    if len(decoded) != 64:
        raise RunnerAttestationError("Ed25519 signatures are 64 bytes")
    return decoded


@dataclass(frozen=True)
class RunnerPublicKey:
    """Exact Ed25519 multicodec public-key material, never an encoded secret."""

    raw_key: bytes

    def __post_init__(self) -> None:
        if not isinstance(self.raw_key, bytes) or len(self.raw_key) != 32:
            raise RunnerAttestationError("Ed25519 public key must be 32 bytes")

    @property
    def material(self) -> bytes:
        return _uvarint(ED25519_PUB_MULTICODEC) + self.raw_key

    @property
    def cid(self) -> str:
        return cidv1_for_bytes(self.material, "raw")

    @classmethod
    def from_public_key(cls, key: Ed25519PublicKey) -> "RunnerPublicKey":
        if not isinstance(key, Ed25519PublicKey):
            raise RunnerAttestationError("expected an Ed25519 public key")
        return cls(key.public_bytes_raw())

    @classmethod
    def from_material(cls, material: bytes) -> "RunnerPublicKey":
        prefix = _uvarint(ED25519_PUB_MULTICODEC)
        if not isinstance(material, bytes) or len(material) != len(prefix) + 32 or not material.startswith(prefix):
            raise RunnerAttestationError("invalid ed25519-pub multicodec material")
        return cls(material[len(prefix) :])


@dataclass(frozen=True)
class RunnerKeyRecord:
    public_key_cid: str
    public_key_material: bytes
    key_epoch: str
    not_before: int
    not_after: int
    usages: tuple[str, ...] = (PYTEST_PASS_ATTESTATION_USAGE,)
    replaces_key_cid: str = ""
    revoked: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "public_key_cid", _strict_cid(self.public_key_cid, codec="raw"))
        public = RunnerPublicKey.from_material(self.public_key_material)
        if public.cid != self.public_key_cid:
            raise RunnerAttestationError("public key CID does not match exact multicodec material")
        object.__setattr__(self, "key_epoch", _clean_text(self.key_epoch, "key epoch"))
        object.__setattr__(self, "not_before", _positive_int(self.not_before, "not_before"))
        object.__setattr__(self, "not_after", _positive_int(self.not_after, "not_after"))
        if self.not_after < self.not_before:
            raise RunnerAttestationError("key validity interval is inverted")
        usages = tuple(sorted({_clean_text(item, "key usage") for item in self.usages}))
        if usages != (PYTEST_PASS_ATTESTATION_USAGE,):
            raise RunnerAttestationError("runner key is not restricted to pytest-pass attestations")
        object.__setattr__(self, "usages", usages)
        if self.replaces_key_cid:
            object.__setattr__(self, "replaces_key_cid", _strict_cid(self.replaces_key_cid, codec="raw"))
        if not isinstance(self.revoked, bool):
            raise RunnerAttestationError("revoked must be boolean")

    def to_dict(self) -> dict[str, Any]:
        return {
            "public_key_cid": self.public_key_cid, "public_key_material": self.public_key_material, "key_epoch": self.key_epoch,
            "not_before": self.not_before, "not_after": self.not_after,
            "usages": list(self.usages), "replaces_key_cid": self.replaces_key_cid,
            "revoked": self.revoked,
        }

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunnerKeyRecord":
        if set(value) != {"public_key_cid", "public_key_material", "key_epoch", "not_before", "not_after", "usages", "replaces_key_cid", "revoked"}:
            raise RunnerAttestationError("runner key record has unsupported fields")
        usages = value["usages"]
        if not isinstance(usages, list):
            raise RunnerAttestationError("key usages must be a list")
        return cls(value["public_key_cid"], value["public_key_material"], value["key_epoch"], value["not_before"], value["not_after"], tuple(usages), value["replaces_key_cid"], value["revoked"])


@dataclass(frozen=True)
class RunnerTrustPolicy:
    """An immutable local trust root.  Possession is not a trust decision."""

    trust_domain: str
    active_key_epoch: str
    keys: tuple[RunnerKeyRecord, ...]
    policy_epoch: str = "1"
    revoked_key_cids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "trust_domain", _clean_text(self.trust_domain, "trust domain"))
        object.__setattr__(self, "active_key_epoch", _clean_text(self.active_key_epoch, "active key epoch"))
        object.__setattr__(self, "policy_epoch", _clean_text(self.policy_epoch, "policy epoch"))
        if not self.keys or len(self.keys) > 64 or not all(isinstance(key, RunnerKeyRecord) for key in self.keys):
            raise RunnerAttestationError("policy requires bounded runner keys")
        identities = [(key.public_key_cid, key.key_epoch) for key in self.keys]
        if len(set(identities)) != len(identities):
            raise RunnerAttestationError("duplicate policy key epoch")
        if not any(key.key_epoch == self.active_key_epoch for key in self.keys):
            raise RunnerAttestationError("active policy epoch has no key")
        known_key_cids = {key.public_key_cid for key in self.keys}
        for key in self.keys:
            if key.replaces_key_cid:
                if key.replaces_key_cid == key.public_key_cid:
                    raise RunnerAttestationError("a runner key cannot replace itself")
                if key.replaces_key_cid not in known_key_cids:
                    raise RunnerAttestationError("key rotation predecessor is absent from policy")
        revoked = tuple(sorted({_strict_cid(item, codec="raw") for item in self.revoked_key_cids}))
        object.__setattr__(self, "revoked_key_cids", revoked)

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "interface": RUNNER_TRUST_POLICY_INTERFACE, "trust_domain": self.trust_domain,
            "active_key_epoch": self.active_key_epoch, "policy_epoch": self.policy_epoch,
            "keys": [key.to_dict() for key in sorted(self.keys, key=lambda item: (item.key_epoch, item.public_key_cid))],
            "revoked_key_cids": list(self.revoked_key_cids),
        }

    def canonical_bytes(self) -> bytes:
        return canonical_dag_cbor(self.unsigned_dict())

    @property
    def cid(self) -> str:
        return cidv1_for_bytes(self.canonical_bytes(), "dag-cbor")

    @classmethod
    def from_bytes(cls, encoded: bytes, *, expected_cid: str | None = None) -> "RunnerTrustPolicy":
        if expected_cid is not None and _strict_cid(expected_cid, codec="dag-cbor") != cidv1_for_bytes(encoded, "dag-cbor"):
            raise RunnerAttestationError("trust policy CID does not match bytes")
        value = decode_canonical_dag_cbor(encoded)
        if set(value) != {"interface", "trust_domain", "active_key_epoch", "policy_epoch", "keys", "revoked_key_cids"} or value.get("interface") != RUNNER_TRUST_POLICY_INTERFACE:
            raise RunnerAttestationError("unsupported runner trust policy")
        if not isinstance(value["keys"], list) or not isinstance(value["revoked_key_cids"], list):
            raise RunnerAttestationError("malformed runner trust policy")
        policy = cls(value["trust_domain"], value["active_key_epoch"], tuple(RunnerKeyRecord.from_dict(item) for item in value["keys"]), value["policy_epoch"], tuple(value["revoked_key_cids"]))
        if policy.canonical_bytes() != encoded:
            raise RunnerAttestationError("runner trust policy is noncanonical")
        return policy

    def key_for(self, key_cid: str, key_epoch: str, now: int) -> RunnerKeyRecord:
        key_cid = _strict_cid(key_cid, codec="raw")
        key_epoch = _clean_text(key_epoch, "key epoch")
        if key_epoch != self.active_key_epoch:
            raise RunnerAttestationError("attestation key epoch is not active")
        matches = [key for key in self.keys if key.public_key_cid == key_cid and key.key_epoch == key_epoch]
        if len(matches) != 1:
            raise RunnerAttestationError("signer key is not trusted by pinned policy")
        key = matches[0]
        if (key.revoked or key_cid in self.revoked_key_cids or PYTEST_PASS_ATTESTATION_USAGE not in key.usages or now < key.not_before or now > key.not_after):
            raise RunnerAttestationError("signer key is revoked, expired, or not permitted for pytest passes")
        return key


def phase_commitment(receipt: TestPassReceipt) -> str:
    return dag_cbor_cid({"setup": receipt.setup_outcome.value, "call": receipt.call_outcome.value, "teardown": receipt.teardown_outcome.value})


def trace_commitment(receipt: TestPassReceipt) -> str:
    return dag_cbor_cid({"static_trace_root_cid": receipt.static_trace_root_cid, "runtime_trace_root_cid": receipt.runtime_trace_root_cid, "completeness_receipt_cid": receipt.completeness_receipt_cid})


@dataclass(frozen=True)
class RunnerPassAttestation:
    receipt_cid: str
    execution_key_cid: str
    candidate_context_cid: str
    phase_root_cid: str
    trace_root_cid: str
    policy_cid: str
    trust_domain: str
    signer_key_cid: str
    key_epoch: str
    issuance_nonce: str
    issued_at: int
    signature: bytes = field(repr=False)

    def __post_init__(self) -> None:
        for name in ("receipt_cid", "execution_key_cid", "candidate_context_cid", "phase_root_cid", "trace_root_cid", "policy_cid", "signer_key_cid"):
            object.__setattr__(self, name, _strict_cid(getattr(self, name), codec="dag-cbor" if name == "policy_cid" else None))
        object.__setattr__(self, "trust_domain", _clean_text(self.trust_domain, "trust domain"))
        object.__setattr__(self, "key_epoch", _clean_text(self.key_epoch, "key epoch"))
        object.__setattr__(self, "issuance_nonce", _clean_text(self.issuance_nonce, "issuance nonce"))
        object.__setattr__(self, "issued_at", _positive_int(self.issued_at, "issued_at"))
        if not isinstance(self.signature, bytes) or len(self.signature) != 64:
            raise RunnerAttestationError("Ed25519 signatures are 64 bytes")

    def unsigned_dict(self) -> dict[str, Any]:
        return {
            "interface": RUNNER_PASS_ATTESTATION_INTERFACE, "receipt_cid": self.receipt_cid,
            "execution_key_cid": self.execution_key_cid, "candidate_context_cid": self.candidate_context_cid,
            "phase_root_cid": self.phase_root_cid, "trace_root_cid": self.trace_root_cid,
            "policy_cid": self.policy_cid, "trust_domain": self.trust_domain,
            "signer_key_cid": self.signer_key_cid, "key_epoch": self.key_epoch,
            "issuance_nonce": self.issuance_nonce, "issued_at": self.issued_at,
        }

    def unsigned_bytes(self) -> bytes:
        return canonical_dag_cbor(self.unsigned_dict())

    @property
    def unsigned_cid(self) -> str:
        return cidv1_for_bytes(self.unsigned_bytes(), "dag-cbor")

    def to_dict(self) -> dict[str, Any]:
        return {**self.unsigned_dict(), "signature": _b64encode(self.signature)}

    def canonical_bytes(self) -> bytes:
        return canonical_dag_cbor(self.to_dict())

    @property
    def cid(self) -> str:
        return cidv1_for_bytes(self.canonical_bytes(), "dag-cbor")

    @classmethod
    def from_bytes(cls, encoded: bytes, *, expected_cid: str | None = None) -> "RunnerPassAttestation":
        if expected_cid is not None and _strict_cid(expected_cid, codec="dag-cbor") != cidv1_for_bytes(encoded, "dag-cbor"):
            raise RunnerAttestationError("runner attestation CID does not match bytes")
        value = decode_canonical_dag_cbor(encoded)
        expected = {"interface", "receipt_cid", "execution_key_cid", "candidate_context_cid", "phase_root_cid", "trace_root_cid", "policy_cid", "trust_domain", "signer_key_cid", "key_epoch", "issuance_nonce", "issued_at", "signature"}
        if set(value) != expected or value.get("interface") != RUNNER_PASS_ATTESTATION_INTERFACE:
            raise RunnerAttestationError("unsupported runner attestation")
        attestation = cls(*(value[name] for name in ("receipt_cid", "execution_key_cid", "candidate_context_cid", "phase_root_cid", "trace_root_cid", "policy_cid", "trust_domain", "signer_key_cid", "key_epoch", "issuance_nonce", "issued_at")), _b64decode(value["signature"]))
        if attestation.canonical_bytes() != encoded:
            raise RunnerAttestationError("runner attestation is noncanonical")
        return attestation


class AttestationNonceRegistry:
    """Local issuance registry; verification never mutates or consumes it."""

    def __init__(self, bindings: Mapping[str, str] | None = None) -> None:
        self._bindings = dict(bindings or {})

    def register_issuance(self, attestation: RunnerPassAttestation) -> None:
        previous = self._bindings.get(attestation.issuance_nonce)
        if previous is not None and previous != attestation.cid:
            raise RunnerAttestationError("issuance nonce is already bound to another attestation")
        self._bindings[attestation.issuance_nonce] = attestation.cid

    def matches(self, attestation: RunnerPassAttestation) -> bool:
        return self._bindings.get(attestation.issuance_nonce) == attestation.cid

    def snapshot(self) -> dict[str, str]:
        return dict(self._bindings)


@dataclass(frozen=True)
class AttestationVerification:
    valid: bool
    reason: str
    signed_receipt: SignedTestPassReceiptV2 | None = None


def attest_test_pass_receipt(
    receipt: TestPassReceipt,
    *,
    private_key: Ed25519PrivateKey,
    policy: RunnerTrustPolicy,
    candidate_context_cid: str,
    issuance_nonce: str | None = None,
    issued_at: int | None = None,
    nonce_registry: AttestationNonceRegistry | None = None,
) -> RunnerPassAttestation:
    """Sign a complete pass; issuance may register once, reads never do."""

    if not isinstance(receipt, TestPassReceipt) or not receipt.admitted or not receipt.all_phases_pass:
        raise RunnerAttestationError("only admitted complete pytest passes may be attested")
    if not isinstance(private_key, Ed25519PrivateKey):
        raise RunnerAttestationError("expected an Ed25519 private key")
    now = int(time.time()) if issued_at is None else _positive_int(issued_at, "issued_at")
    public = RunnerPublicKey.from_public_key(private_key.public_key())
    policy.key_for(public.cid, policy.active_key_epoch, now)
    if receipt.trust_domain != policy.trust_domain or receipt.policy_cid != policy.cid:
        raise RunnerAttestationError("receipt does not bind the active local trust policy")
    candidate_context_cid = _strict_cid(candidate_context_cid)
    nonce = issuance_nonce or secrets.token_urlsafe(32)
    unsigned = {
        "interface": RUNNER_PASS_ATTESTATION_INTERFACE, "receipt_cid": _strict_cid(receipt.receipt_id),
        "execution_key_cid": _strict_cid(receipt.execution_key_cid), "candidate_context_cid": candidate_context_cid,
        "phase_root_cid": phase_commitment(receipt), "trace_root_cid": trace_commitment(receipt),
        "policy_cid": policy.cid, "trust_domain": policy.trust_domain, "signer_key_cid": public.cid,
        "key_epoch": policy.active_key_epoch, "issuance_nonce": _clean_text(nonce, "issuance nonce"), "issued_at": now,
    }
    signature = private_key.sign(_digest_message(canonical_dag_cbor(unsigned)))
    attestation = RunnerPassAttestation(
        **{key: value for key, value in unsigned.items() if key != "interface"},
        signature=signature,
    )
    if nonce_registry is not None:
        nonce_registry.register_issuance(attestation)
    return attestation


def verify_runner_pass_attestation(
    attestation: RunnerPassAttestation | bytes,
    *,
    receipt: TestPassReceipt,
    policy: RunnerTrustPolicy,
    pinned_policy_cid: str,
    current_execution_key_cid: str,
    current_candidate_context_cid: str,
    now: int | None = None,
    nonce_registry: AttestationNonceRegistry | None = None,
) -> AttestationVerification:
    """Fail closed: validate local policy and signature before any proof work."""

    try:
        pinned = _strict_cid(pinned_policy_cid, codec="dag-cbor")
        if pinned != policy.cid:
            raise RunnerAttestationError("policy is not the explicitly pinned local policy")
        candidate = RunnerPassAttestation.from_bytes(attestation) if isinstance(attestation, bytes) else attestation
        if not isinstance(candidate, RunnerPassAttestation):
            raise RunnerAttestationError("invalid attestation type")
        current_time = int(time.time()) if now is None else _positive_int(now, "now")
        if candidate.policy_cid != pinned or candidate.trust_domain != policy.trust_domain:
            raise RunnerAttestationError("policy or trust domain mismatch")
        if candidate.receipt_cid != _strict_cid(receipt.receipt_id) or candidate.execution_key_cid != _strict_cid(current_execution_key_cid) or candidate.execution_key_cid != _strict_cid(receipt.execution_key_cid):
            raise RunnerAttestationError("receipt or execution context mismatch")
        if candidate.candidate_context_cid != _strict_cid(current_candidate_context_cid):
            raise RunnerAttestationError("candidate context mismatch")
        if not receipt.admitted or not receipt.all_phases_pass or receipt.policy_cid != pinned or receipt.trust_domain != policy.trust_domain:
            raise RunnerAttestationError("receipt is not a policy-bound complete pass")
        if candidate.phase_root_cid != phase_commitment(receipt) or candidate.trace_root_cid != trace_commitment(receipt):
            raise RunnerAttestationError("phase or trace commitment mismatch")
        policy.key_for(candidate.signer_key_cid, candidate.key_epoch, current_time)
        if nonce_registry is not None and not nonce_registry.matches(candidate):
            raise RunnerAttestationError("issuance nonce is not bound to this attestation")
        key_record = next(key for key in policy.keys if key.public_key_cid == candidate.signer_key_cid and key.key_epoch == candidate.key_epoch)
        public = RunnerPublicKey.from_material(key_record.public_key_material)
        if public.cid != candidate.signer_key_cid:
            raise RunnerAttestationError("pinned key record has inconsistent key material")
        Ed25519PublicKey.from_public_bytes(public.raw_key).verify(candidate.signature, _digest_message(candidate.unsigned_bytes()))
        return AttestationVerification(True, "verified", SignedTestPassReceiptV2(candidate.receipt_cid, candidate.execution_key_cid, candidate.candidate_context_cid, candidate.cid, candidate.policy_cid, candidate.signer_key_cid, candidate.trust_domain, candidate.key_epoch))
    except RunnerAttestationError as exc:
        return AttestationVerification(False, str(exc))
    except Exception:
        return AttestationVerification(False, "attestation verification failed")


def verify_runner_pass_attestation_with_key(
    attestation: RunnerPassAttestation | bytes,
    *,
    receipt: TestPassReceipt,
    policy: RunnerTrustPolicy,
    pinned_policy_cid: str,
    current_execution_key_cid: str,
    current_candidate_context_cid: str,
    pinned_public_key_material: bytes,
    now: int | None = None,
    nonce_registry: AttestationNonceRegistry | None = None,
) -> AttestationVerification:
    """Verification entry point requiring locally provisioned exact key bytes."""

    try:
        candidate = RunnerPassAttestation.from_bytes(attestation) if isinstance(attestation, bytes) else attestation
        if not isinstance(candidate, RunnerPassAttestation):
            raise RunnerAttestationError("invalid attestation type")
        public = RunnerPublicKey.from_material(pinned_public_key_material)
        if public.cid != candidate.signer_key_cid:
            raise RunnerAttestationError("local pinned key material does not match attestation")
        # Do all policy/context and cryptographic checks against the locally
        # pinned policy.  The supplied local material is an additional pin.
        precheck = verify_runner_pass_attestation(candidate, receipt=receipt, policy=policy, pinned_policy_cid=pinned_policy_cid, current_execution_key_cid=current_execution_key_cid, current_candidate_context_cid=current_candidate_context_cid, now=now, nonce_registry=nonce_registry)
        if not precheck.valid:
            return precheck
        return precheck
    except InvalidSignature:
        return AttestationVerification(False, "invalid Ed25519 signature")
    except RunnerAttestationError as exc:
        return AttestationVerification(False, str(exc))
    except Exception:
        return AttestationVerification(False, "attestation verification failed")


__all__ = [
    "ATTESTATION_DOMAIN", "AttestationNonceRegistry", "AttestationVerification",
    "ED25519_PUB_MULTICODEC", "PYTEST_PASS_ATTESTATION_USAGE",
    "RUNNER_PASS_ATTESTATION_INTERFACE", "RUNNER_TRUST_POLICY_INTERFACE",
    "RunnerAttestationError", "RunnerKeyRecord", "RunnerPassAttestation",
    "RunnerPublicKey", "RunnerTrustPolicy", "attest_test_pass_receipt",
    "canonical_dag_cbor", "cidv1_for_bytes", "dag_cbor_cid",
    "decode_canonical_dag_cbor", "phase_commitment", "trace_commitment",
    "verify_runner_pass_attestation", "verify_runner_pass_attestation_with_key",
]
