"""Retain a scoped header proof's complete key beside its native sealed proof.

This is custody in the existing Doctor SQLite owner, not conversion to a generic
proof receipt or admission to a positive DuckDB cache. Replay requires the
independently rebuilt preparation and actually reruns the native pinned kernel.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import stat
from types import MappingProxyType
from typing import ClassVar

from ..proof.deterministic_doctor_hammer import (
    DeterministicDoctorHammer, DoctorAuthoritativeProofReceipt,
    DoctorExactLoweringReceipt, DoctorHammerBounds, DoctorKernelReplayReceipt,
    DoctorNativeExecutionReceipt, DoctorPinnedExecutable,
    DoctorProofAuthorityDisposition, DoctorReviewedTheorem,
)
from ..proof.doctor_proof_cache import (
    DoctorProofCacheKey, DoctorSealedReceiptRef, DoctorSealedReceiptStore,
)
from ..proof.formal_verification_contracts import CanonicalContract, canonical_json
from .formal_status_census import _keys
from .supervisor_meta_index import public_replay_without_metadata

SCHEMA = "ipfs_accelerate_py/agent-supervisor/header-contract-proof-key-custody@1"
PREPARATION_SCHEMA = "supervisor-header-contract-proof-preparation@1"
MAX_PREPARATION_BYTES = 1024 * 1024
MAX_CUSTODY_BYTES = MAX_PREPARATION_BYTES + 32 * 1024
MAX_REFERENCE_BYTES = 16 * 1024
_FLAGS = (
    "proof_authority", "mutation_authority", "execution_authority",
    "completion_authority", "publication_authority", "generic_cache_admission_supported",
)
_PREPARATION_FLAGS = ("proof_authority", "execution_authority",
                      "generic_cache_admission_supported", "resource_enforcement_verified")
_PREPARATION_FIELDS = frozenset((
    "schema", "task_cid", "analysis", "contract", "candidate_cid", "theorem", "lowering",
    "solver_pin", "kernel_pin", "bounds", "lean", "z3", "translator_preimage",
    "toolchain_preimage", "profiles", "key", "planned", *_PREPARATION_FLAGS,
))


class HeaderContractProofCustodyError(ValueError):
    """Native proof custody differs from independently admitted current inputs."""


def _require(condition, message):
    if not condition:
        raise HeaderContractProofCustodyError(message)


def _json(value, maximum):
    rendered = canonical_json(value)
    _require(len(rendered.encode("utf-8")) <= maximum, "bounded canonical custody required")
    return rendered


def _reference(value, schema=None):
    _require(type(value) is DoctorSealedReceiptRef, "exact sealed native reference required")
    body = _json(value.to_dict(), MAX_REFERENCE_BYTES)
    restored = DoctorSealedReceiptRef.from_dict(json.loads(body))
    _require(_json(restored.to_dict(), MAX_REFERENCE_BYTES) == body
             and (schema is None or restored.receipt_schema == schema),
             "sealed native reference differs from its concrete schema")
    return restored


def _preparation_body(value):
    _require(type(value) is dict, "exact canonical preparation body required")
    body = _json(value, MAX_PREPARATION_BYTES)
    _require(set(value) == _PREPARATION_FIELDS and value.get("schema") == PREPARATION_SCHEMA
             and type(value.get("planned")) is bool
             and all(value.get(name) is False for name in _PREPARATION_FLAGS),
             "closed versioned preparation body without authority required")
    raw_key = value.get("key")
    _require(type(raw_key) is dict, "complete native key custody required")
    key = DoctorProofCacheKey.from_dict(raw_key)
    restored = _keys((key,), None)[0]
    _require(_json(restored.to_dict(), MAX_PREPARATION_BYTES)
             == _json(raw_key, MAX_PREPARATION_BYTES), "native key custody did not roundtrip")
    return body, restored


@dataclass(frozen=True, slots=True)
class HeaderContractProofKeyCustody(CanonicalContract):
    """Bounded canonical key preparation plus the exact sealed proof reference.

    Canonical bytes and seals establish retained custody. All authority claims
    remain false; only the existing native proof replay may return its own proof.
    """

    SCHEMA: ClassVar[str] = SCHEMA
    preparation_json: str
    proof_ref: DoctorSealedReceiptRef

    def __post_init__(self):
        _require(type(self.preparation_json) is str
                 and len(self.preparation_json.encode("utf-8")) <= MAX_PREPARATION_BYTES,
                 "bounded canonical preparation required")
        body, _key = _preparation_body(json.loads(self.preparation_json))
        _require(body == self.preparation_json, "preparation must retain exact canonical JSON")
        object.__setattr__(self, "proof_ref", _reference(
            self.proof_ref, DoctorAuthoritativeProofReceipt.SCHEMA))
        _json(self._payload(), MAX_CUSTODY_BYTES)

    def _payload(self):
        return {"contract_version": 1, "preparation": json.loads(self.preparation_json),
                "proof_ref": self.proof_ref.to_dict(), **{name: False for name in _FLAGS}}

    @classmethod
    def from_dict(cls, value):
        _require(type(value) is dict, "exact custody record required")
        _json(value, MAX_CUSTODY_BYTES)
        allowed = {"schema", "contract_version", "preparation", "proof_ref", "content_id", *_FLAGS}
        _require(not set(value) - allowed and value.get("schema") == cls.SCHEMA
                 and type(value.get("contract_version")) is int and value["contract_version"] == 1
                 and all(value.get(name) is False for name in _FLAGS),
                 "closed custody schema and absent authority required")
        _require(type(value.get("proof_ref")) is dict, "exact retained native proof reference required")
        proof_ref = DoctorSealedReceiptRef.from_dict(value["proof_ref"])
        _require(_json(proof_ref.to_dict(), MAX_REFERENCE_BYTES)
                 == _json(value["proof_ref"], MAX_REFERENCE_BYTES), "complete native reference required")
        body, _key = _preparation_body(value.get("preparation"))
        result = cls(preparation_json=body, proof_ref=proof_ref)
        _require(value.get("content_id", result.content_id) == result.content_id,
                 "custody content identity differs")
        return result


def _file_identity(path, *, private=False):
    _require(type(path) is type(Path()) and path.is_absolute()
             and path.resolve(strict=True) == path
             and not any(part.is_symlink() for part in (path, *path.parents)),
             "canonical existing native owner required")
    info = path.stat(follow_symlinks=False)
    _require(stat.S_ISREG(info.st_mode) and info.st_uid == os.geteuid()
             and not stat.S_IMODE(info.st_mode) & (0o077 if private else 0o022),
             "owner-controlled native custody file required")
    if private:
        _require(info.st_size == 32, "original native seal key required")
    return info.st_dev, info.st_ino


class _OwnerFence:
    """Keep strong references and file continuity across native operations."""

    def __init__(self, hammer, store):
        self.hammer, self.store = hammer, store
        self.store_id, self.authority_id = store.store_id, store.authority_id
        self.path, self.key_path = store.path, store._key_path
        self.owner = _file_identity(self.path)
        self.key_owner = _file_identity(self.key_path, private=True)
        self.key_digest = self._key_digest()
        self.check()

    def _key_digest(self):
        descriptor = os.open(self.key_path, os.O_RDONLY | os.O_NOFOLLOW)
        try:
            info = os.fstat(descriptor)
            _require((info.st_dev, info.st_ino) == self.key_owner
                     and stat.S_ISREG(info.st_mode) and info.st_size == 32,
                     "original native seal-key owner required")
            raw = os.read(descriptor, 33)
        finally:
            os.close(descriptor)
        _require(type(self.store._seal_key) is bytes and len(raw) == 32
                 and raw == self.store._seal_key, "native in-memory and retained seal keys differ")
        return hashlib.sha256(raw).digest()

    def check(self):
        _require(self.hammer._authoritative_store is self.store
                 and self.store.store_id == self.store_id and self.store.authority_id == self.authority_id
                 and self.store.path == self.path and self.store._key_path == self.key_path
                 and _file_identity(self.path) == self.owner
                 and _file_identity(self.key_path, private=True) == self.key_owner
                 and self._key_digest() == self.key_digest,
                 "native custody owner changed")


def _prepared_inputs(prepared, hammer, expected_key_id=None):
    # Local import preserves the existing proof factory's import direction.
    from .doctor_contract_proof import HeaderContractProofPreparation
    from .doctor_contract_key_custody import HeaderContractProofKeyProfiles

    _require(type(prepared) is HeaderContractProofPreparation
             and type(hammer) is DeterministicDoctorHammer,
             "exact native preparation and existing hammer required")
    _require(type(prepared.key) is DoctorProofCacheKey
             and type(prepared.proof_key_profiles) is HeaderContractProofKeyProfiles,
             "complete independently rebuilt key profiles required")
    key = _keys((prepared.key,), None)[0]
    _require(expected_key_id is None or (type(expected_key_id) is str
             and expected_key_id == key.key_id), "independently expected full key identity differs")
    _require(type(prepared.bounds) is DoctorHammerBounds and type(hammer.bounds) is DoctorHammerBounds
             and canonical_json(hammer.bounds.to_dict()) == canonical_json(prepared.bounds.to_dict()),
             "existing native hammer bounds differ from preparation")
    trusted = hammer._trusted_executable_pins
    _require(type(trusted) is MappingProxyType
             and all(type(trusted.get(pin.content_id)) is type(pin)
                     and trusted.get(pin.content_id) == pin
                     for pin in (prepared.solver_pin, prepared.kernel_pin)),
             "both exact executable profiles must remain independently admitted")
    _require(type(hammer._authoritative_store) is DoctorSealedReceiptStore,
             "existing exact native sealed owner required")
    prepared.assert_current()
    body, retained_key = _preparation_body(prepared.custody_payload())
    _require(canonical_json(retained_key.to_dict()) == canonical_json(key.to_dict()),
             "preparation custody differs from the full native key")
    return body, hammer._authoritative_store


def _proof_binding(prepared, proof, store, fence):
    _require(type(proof) is DoctorAuthoritativeProofReceipt
             and proof.disposition is DoctorProofAuthorityDisposition.VERIFIED
             and proof.authority_store is store
             and all(type(value) is expected for value, expected in (
                 (proof.theorem, DoctorReviewedTheorem), (proof.lowering, DoctorExactLoweringReceipt),
                 (proof.solver_pin, DoctorPinnedExecutable), (proof.kernel_pin, DoctorPinnedExecutable),
                 (proof.native_execution, DoctorNativeExecutionReceipt),
                 (proof.kernel_replay, DoctorKernelReplayReceipt))),
             "exact successful native proof and its existing owner required")
    ref = _reference(proof.authority_store_ref, DoctorAuthoritativeProofReceipt.SCHEMA)
    _require(all(canonical_json(actual.to_dict()) == canonical_json(expected.to_dict())
                 for actual, expected in ((proof.theorem, prepared.theorem),
                                          (proof.lowering, prepared.lowering),
                                          (proof.solver_pin, prepared.solver_pin),
                                          (proof.kernel_pin, prepared.kernel_pin)))
             and canonical_json(proof.roots.to_dict()) == canonical_json(prepared.theorem.roots.to_dict())
             and all(canonical_json(receipt.roots.to_dict())
                     == canonical_json(prepared.theorem.roots.to_dict())
                     and receipt.executable_sha256 == pin.executable_sha256
                     and receipt.command_id == pin.command_id
                     for receipt, pin in ((proof.native_execution, prepared.solver_pin),
                                          (proof.kernel_replay, prepared.kernel_pin)))
             and ref.receipt_cid == proof.content_id,
             "sealed proof differs from independently rebuilt native inputs")
    loaded = store.reload(ref, DoctorAuthoritativeProofReceipt)
    fence.check()
    _require(canonical_json(loaded.to_dict()) == canonical_json(proof.to_dict()),
             "native authoritative proof envelope differs")
    for retained_ref, receipt in ((proof.native_store_ref, proof.native_execution),
                                  (proof.kernel_store_ref, proof.kernel_replay)):
        retained_ref = _reference(retained_ref, type(receipt).SCHEMA)
        retained = store.reload(retained_ref, type(receipt))
        fence.check()
        _require(canonical_json(retained.to_dict()) == canonical_json(receipt.to_dict()),
                 "native proof evidence differs from its original sealed lineage")
    return ref


def _still_current(prepared, hammer, body, store, fence):
    fence.check()
    current_body, current_store = _prepared_inputs(prepared, hammer)
    _require(current_store is store and current_body == body,
             "current independent preparation changed during native custody")
    fence.check()


def persist_header_proof_key_custody(*, prepared, proof, hammer) -> DoctorSealedReceiptRef:
    """Seal complete key preparation beside a successful proof in the same owner."""
    with public_replay_without_metadata():
        body, store = _prepared_inputs(prepared, hammer)
        fence = _OwnerFence(hammer, store)
        proof_ref = _proof_binding(prepared, proof, store, fence)
        _still_current(prepared, hammer, body, store, fence)
        custody = HeaderContractProofKeyCustody(preparation_json=body, proof_ref=proof_ref)
        reference = store.append(custody)
        fence.check()
        reference = _reference(reference, HeaderContractProofKeyCustody.SCHEMA)
        loaded = store.reload(reference, HeaderContractProofKeyCustody)
        fence.check()
        _require(loaded.to_dict() == custody.to_dict(), "sealed key custody did not replay exactly")
        _still_current(prepared, hammer, body, store, fence)
        return reference


def replay_header_proof_key_custody(*, prepared, hammer, custody_ref,
                                   expected_key_id) -> DoctorAuthoritativeProofReceipt:
    """Replay retained native custody and rerun the actually pinned kernel.

    The caller independently admits an existing store and rebuilds current
    preparation. This operation never discovers a locator or imports a generic
    positive cache entry. Sequential fencing is not an atomic filesystem lease.
    """
    _require(type(expected_key_id) is str and expected_key_id,
             "independently expected full key identity required")
    reference = _reference(custody_ref, HeaderContractProofKeyCustody.SCHEMA)
    with public_replay_without_metadata():
        body, store = _prepared_inputs(prepared, hammer, expected_key_id)
        fence = _OwnerFence(hammer, store)
        custody = store.reload(reference, HeaderContractProofKeyCustody)
        fence.check()
        _require(custody.preparation_json == body,
                 "retained custody differs from current independent preparation")
        proof_ref = _reference(custody.proof_ref, DoctorAuthoritativeProofReceipt.SCHEMA)
        loaded = store.reload(proof_ref, DoctorAuthoritativeProofReceipt)
        fence.check()
        proof = replace(loaded, authority_store_ref=proof_ref, authority_store=store)
        _proof_binding(prepared, proof, store, fence)
        _still_current(prepared, hammer, body, store, fence)
        replayed = hammer.reverify_authoritative(proof, current_roots=prepared.theorem.roots)
        fence.check()
        _proof_binding(prepared, replayed, store, fence)
        _still_current(prepared, hammer, body, store, fence)
        return replayed
