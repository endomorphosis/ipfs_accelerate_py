"""Retain an independently admitted header-proof population in its native owner.

Inventory custody grants no proof or task authority. Restart requires an
independently pinned inventory CID/reference, both original owner paths and
independently rebuilt current preparations. Every entry is preflighted before
fresh kernel imports; no expected population is discovered from a cache. Checks
are sequential, not an atomic lease or rollback. Earlier valid imports and
native executable effects can remain after a later failure.
"""
from __future__ import annotations

import base64
from dataclasses import dataclass, replace
import hashlib
import json
from pathlib import Path
import re
from types import MappingProxyType
from typing import ClassVar

from ..proof.deterministic_doctor_hammer import (
    DeterministicDoctorHammer, DoctorAuthoritativeProofReceipt, DoctorHammerBounds,
)
from ..proof.doctor_proof_cache import DoctorSealedReceiptRef, DoctorSealedReceiptStore
from ..proof.formal_verification_contracts import CanonicalContract, canonical_json
from . import doctor_contract_proof_custody as custody
from . import doctor_header_cache_bridge as bridge
from .doctor_contract_proof import HeaderContractProofPreparation
from .doctor_scoped_analysis import ScopedDoctorAnalysis
from .formal_status_census import MAX_KEYS, _keys
from .supervisor_meta_index import public_replay_without_metadata

SCHEMA = "ipfs_accelerate_py/agent-supervisor/header-proof-cache-inventory@1"
ENTRY_SCHEMA = "ipfs_accelerate_py/agent-supervisor/header-proof-cache-inventory-entry@1"
MAX_INVENTORY_BYTES = 256 * 1024
_DIGEST = re.compile(r"[0-9a-f]{64}\Z")
_KEY_ID = re.compile(r"doctor-proof-cache-key:sha256:[0-9a-f]{64}\Z")
_FIELDS = ("task_cid", "repository_id", "scope_id", "store_id", "authority_id",
    "native_owner_path_sha256", "cache_owner_path_sha256", "bridge_sha256",
    "receipt_contracts_sha256", "inventory_implementation_sha256", "entries")
_FLAGS = ("proof_authority", "execution_authority", "mutation_authority", "completion_authority",
    "publication_authority", "replacement_authority", "omission_authority", "whole_program_proved",
    "resource_enforcement_verified", "generic_cache_admission_supported")


class HeaderProofInventoryError(ValueError):
    """Inventory custody or current independently admitted population refused."""


def _require(condition, message):
    if not condition:
        raise HeaderProofInventoryError(message)


def _json(value):
    body = canonical_json(value)
    _require(len(body.encode("utf-8")) <= MAX_INVENTORY_BYTES, "bounded canonical inventory required")
    return body


def _text(value):
    _require(type(value) is str and value and value == value.strip()
             and len(value.encode()) <= 1024 and not any(ord(c) < 32 for c in value),
             "bounded exact inventory identity required")
    return value


def _cid(value):
    _text(value)
    _require(value.startswith("b") and value == value.lower(), "native DAG-JSON CID required")
    try:
        raw = base64.b32decode(value[1:].upper() + "=" * (-len(value[1:]) % 8))
    except (ValueError, TypeError) as exc:
        raise HeaderProofInventoryError("native inventory CID is malformed") from exc
    _require(len(raw) == 37 and raw[:5] == b"\x01\xa9\x02\x12\x20"
             and "b" + base64.b32encode(raw).decode().rstrip("=").lower() == value,
             "exact CIDv1 DAG-JSON SHA256 identity required")
    return value


def _digest(value):
    _require(type(value) is str and _DIGEST.fullmatch(value), "independently admitted SHA256 required")
    return value


def _implementations():
    body = bridge.header_proof_implementation_bindings()
    _require(type(body) is dict and set(body) == {"bridge_sha256", "receipt_contracts_sha256"},
             "exact current header emitter and codec provenance required")
    return {**{name: _digest(value) for name, value in body.items()},
        "inventory_implementation_sha256": hashlib.sha256(Path(__file__).resolve(strict=True).read_bytes()).hexdigest()}


@dataclass(frozen=True, slots=True)
class HeaderProofCacheInventoryEntry(CanonicalContract):
    SCHEMA: ClassVar[str] = ENTRY_SCHEMA
    key_id: str
    generic_receipt_id: str
    custody_ref: DoctorSealedReceiptRef

    def __post_init__(self):
        _require(type(self.key_id) is str and _KEY_ID.fullmatch(self.key_id), "complete native key identity required")
        _cid(self.generic_receipt_id)
        object.__setattr__(self, "custody_ref", custody._reference(self.custody_ref, custody.HeaderContractProofKeyCustody.SCHEMA))
        _json(self.to_dict())

    def _payload(self):
        return {"contract_version": 1, "key_id": self.key_id, "generic_receipt_id": self.generic_receipt_id,
                "custody_ref": self.custody_ref.to_dict()}

    @classmethod
    def from_dict(cls, value):
        _require(type(value) is dict and set(value) == {"schema", "contract_version", "key_id", "generic_receipt_id", "custody_ref"}
                 and value["schema"] == cls.SCHEMA and type(value["contract_version"]) is int
                 and value["contract_version"] == 1 and type(value["custody_ref"]) is dict,
                 "closed versioned inventory entry required")
        result = cls(value["key_id"], value["generic_receipt_id"], DoctorSealedReceiptRef.from_dict(value["custody_ref"]))
        _require(_json(result.to_dict()) == _json(value), "exact inventory entry roundtrip required")
        return result


@dataclass(frozen=True, slots=True)
class HeaderProofCacheInventory(CanonicalContract):
    """A bounded immutable population; its seal authenticates retained custody."""

    SCHEMA: ClassVar[str] = SCHEMA
    task_cid: str
    repository_id: str
    scope_id: str
    store_id: str
    authority_id: str
    native_owner_path_sha256: str
    cache_owner_path_sha256: str
    bridge_sha256: str
    receipt_contracts_sha256: str
    inventory_implementation_sha256: str
    entries: tuple[HeaderProofCacheInventoryEntry, ...]

    def __post_init__(self):
        for name in ("task_cid", "repository_id", "scope_id", "store_id"):
            _cid(getattr(self, name))
        _text(self.authority_id)
        for name in ("native_owner_path_sha256", "cache_owner_path_sha256", "bridge_sha256",
                     "receipt_contracts_sha256", "inventory_implementation_sha256"):
            _digest(getattr(self, name))
        _require(type(self.entries) is tuple and len(self.entries) <= MAX_KEYS
                 and all(type(item) is HeaderProofCacheInventoryEntry for item in self.entries),
                 "bounded exact immutable inventory entries required")
        entries = tuple(sorted(self.entries, key=lambda item: item.key_id))
        _require(len({item.key_id for item in entries}) == len(entries)
                 and len({item.generic_receipt_id for item in entries}) == len(entries)
                 and all(item.custody_ref.store_id == self.store_id and item.custody_ref.authority_id == self.authority_id
                         for item in entries), "unique population in the same original native owner required")
        object.__setattr__(self, "entries", entries)
        _json(self.to_dict())

    proof_authority = property(lambda self: False)
    completion_authority = property(lambda self: False)
    execution_authority = property(lambda self: False)
    mutation_authority = property(lambda self: False)
    publication_authority = property(lambda self: False)
    replacement_authority = property(lambda self: False)
    omission_authority = property(lambda self: False)
    whole_program_proved = property(lambda self: False)
    resource_enforcement_verified = property(lambda self: False)
    generic_cache_admission_supported = property(lambda self: False)

    def _payload(self):
        return {"contract_version": 1, **{name: getattr(self, name) for name in _FIELDS if name != "entries"},
                "entries": [item.to_dict() for item in self.entries], **{name: False for name in _FLAGS}}

    @classmethod
    def from_dict(cls, value):
        _require(type(value) is dict, "exact inventory object required")
        _json(value)
        fields = {"schema", "contract_version", *_FIELDS, *_FLAGS}
        _require(set(value) in (fields, fields | {"content_id"}) and value["schema"] == cls.SCHEMA
                 and type(value["contract_version"]) is int and value["contract_version"] == 1
                 and all(value[name] is False for name in _FLAGS) and type(value["entries"]) is list,
                 "closed inventory schema and absent authority required")
        result = cls(**{name: value[name] for name in _FIELDS if name != "entries"},
                     entries=tuple(HeaderProofCacheInventoryEntry.from_dict(item) for item in value["entries"]))
        _require(value.get("content_id", result.content_id) == result.content_id
                 and _json(result.to_dict()) == _json({name: body for name, body in value.items() if name != "content_id"}),
                 "inventory canonical identity or population order differs")
        return result


class _Population:
    def __init__(self, *, scoped_analysis, hammer, gate, native_digest, cache_digest):
        _require(type(scoped_analysis) is ScopedDoctorAnalysis and type(hammer) is DeterministicDoctorHammer
                 and type(hammer._authoritative_store) is DoctorSealedReceiptStore
                 and type(hammer.bounds) is DoctorHammerBounds and type(hammer._trusted_executable_pins) is MappingProxyType,
                 "exact admitted scope, existing native hammer, bounds, pins and store required")
        _digest(native_digest); _digest(cache_digest)
        self.scoped, self.hammer, self.gate, self.store = scoped_analysis, hammer, gate, hammer._authoritative_store
        _require(hashlib.sha256(str(self.store.path).encode()).hexdigest() == native_digest,
                 "native owner differs from independently expected path")
        self.native_fence = custody._OwnerFence(hammer, self.store)
        self.cache_fence = bridge._CacheFence(gate, cache_digest)
        self.native_digest, self.cache_digest = native_digest, cache_digest
        self.scope_json = canonical_json(scoped_analysis.report)
        self.roots_json = canonical_json(scoped_analysis.snapshot.roots.to_dict())
        self.bounds_json = canonical_json(hammer.bounds.to_dict())
        _require(canonical_json(DoctorHammerBounds.from_dict(hammer.bounds.to_dict()).to_dict()) == self.bounds_json,
                 "current native bounds must roundtrip exactly")
        self.trusted_pins = hammer._trusted_executable_pins
        self.implementations = _implementations()
        self.inventory, self.reference = None, None
        self.preflight = ()
        self.current()

    def current(self):
        self.native_fence.check(); self.cache_fence.check()
        self.scoped.assert_current()
        _require(canonical_json(self.scoped.report) == self.scope_json
                 and canonical_json(self.scoped.snapshot.roots.to_dict()) == self.roots_json
                 and canonical_json(self.hammer.bounds.to_dict()) == self.bounds_json
                 and self.hammer._trusted_executable_pins is self.trusted_pins
                 and _implementations() == self.implementations,
                 "current signed scope or inventory/emitter/codec implementation changed")
        if self.inventory is not None:
            loaded = self.store.reload(self.reference, HeaderProofCacheInventory)
            _require(loaded.to_dict() == self.inventory.to_dict(), "original pinned inventory changed")
        for prepared, reference, proof, body in self.preflight:
            current_body, store = custody._prepared_inputs(prepared, self.hammer, prepared.key.key_id)
            _require(store is self.store and prepared.scoped_analysis is self.scoped and current_body == body,
                     "whole prepared population or original native owner changed")
            retained = self.store.reload(reference, custody.HeaderContractProofKeyCustody)
            _require(retained.preparation_json == body and retained.proof_ref == proof.authority_store_ref,
                     "original key custody or native proof lineage changed")
            custody._proof_binding(prepared, proof, self.store, self.native_fence)
        # Storage callbacks have finished. Recheck every source/profile/pin
        # preparation, then compare all retained bodies without owner callbacks.
        for prepared, _reference, _proof, body in self.preflight:
            current_body, store = custody._prepared_inputs(prepared, self.hammer, prepared.key.key_id)
            _require(store is self.store and current_body == body, "population changed during lineage consultation")
        self.scoped.assert_current()
        _require(_implementations() == self.implementations
                 and canonical_json(self.scoped.report) == self.scope_json
                 and canonical_json(self.scoped.snapshot.roots.to_dict()) == self.roots_json
                 and canonical_json(self.hammer.bounds.to_dict()) == self.bounds_json
                 and self.hammer._trusted_executable_pins is self.trusted_pins
                 and all(canonical_json(prepared.custody_payload()) == body
                         for prepared, _reference, _proof, body in self.preflight),
                 "final inventory implementation or prepared population changed")
        self.native_fence.check(); self.cache_fence.check()


def _preflight_entry(population, prepared, reference, key_id):
    _require(type(prepared) is HeaderContractProofPreparation and prepared.scoped_analysis is population.scoped,
             "original independently admitted scoped preparation required")
    body, store = custody._prepared_inputs(prepared, population.hammer, key_id)
    _require(store is population.store, "original native population owner required")
    retained = store.reload(reference, custody.HeaderContractProofKeyCustody)
    _require(retained.preparation_json == body, "retained key custody differs from independently rebuilt current preparation")
    loaded = store.reload(retained.proof_ref, DoctorAuthoritativeProofReceipt)
    proof = replace(loaded, authority_store_ref=retained.proof_ref, authority_store=store)
    custody._proof_binding(prepared, proof, store, population.native_fence)
    return prepared, reference, proof, body


def _expected_population(key_ids, receipt_ids):
    _require(type(key_ids) is tuple and len(key_ids) <= MAX_KEYS
             and all(type(value) is str and _KEY_ID.fullmatch(value) for value in key_ids)
             and len(set(key_ids)) == len(key_ids) and type(receipt_ids) is tuple
             and len(receipt_ids) == len(key_ids), "independently expected complete inventory population required")
    _require(all(type(pair) is tuple and len(pair) == 2 and type(pair[0]) is str
                 and _KEY_ID.fullmatch(pair[0]) for pair in receipt_ids), "exact independently expected receipt pairs required")
    for pair in receipt_ids:
        _cid(pair[1])
    expected = dict(receipt_ids)
    _require(len(expected) == len(receipt_ids) and set(expected) == set(key_ids)
             and len({pair[1] for pair in receipt_ids}) == len(receipt_ids), "unique exact key and receipt population required")
    return expected


def persist_header_proof_inventory(*, scoped_analysis, hammer, gate, admissions, expected_key_ids,
                                   expected_receipt_ids, expected_native_owner_path_sha256,
                                   expected_cache_owner_path_sha256) -> DoctorSealedReceiptRef:
    """Seal an explicit current population in its original native SQLite owner."""
    try:
        with public_replay_without_metadata():
            expected = _expected_population(expected_key_ids, expected_receipt_ids)
            _require(type(admissions) is tuple and len(admissions) == len(expected)
                     and all(type(value) is bridge.HeaderProofCacheAdmission for value in admissions),
                     "exact complete native admission handles required")
            keys = _keys(tuple(value.key for value in admissions), None)
            _require({key.key_id for key in keys} == set(expected), "admissions differ from independently expected population")
            population = _Population(scoped_analysis=scoped_analysis, hammer=hammer, gate=gate,
                native_digest=expected_native_owner_path_sha256, cache_digest=expected_cache_owner_path_sha256)
            entries = []
            for admission in admissions:
                key = admission.key
                binding = admission.inventory_binding(hammer=hammer, gate=gate, scoped_analysis=scoped_analysis,
                    expected_key_id=key.key_id, expected_receipt_id=expected[key.key_id],
                    expected_native_owner_path_sha256=expected_native_owner_path_sha256,
                    expected_cache_owner_path_sha256=expected_cache_owner_path_sha256)
                _require(type(binding) is dict and set(binding) == {"custody_ref", "key_id", "generic_receipt_id",
                    "task_cid", "repository_id", "scope_id", "native_owner_path_sha256", "cache_owner_path_sha256",
                    "generic_receipt_implementations"} and binding["key_id"] == key.key_id
                    and binding["generic_receipt_id"] == expected[key.key_id]
                    and binding["task_cid"] == scoped_analysis.task_cid
                    and binding["repository_id"] == scoped_analysis.snapshot.roots.repository_id
                    and binding["scope_id"] == scoped_analysis.report["scope_id"]
                    and binding["native_owner_path_sha256"] == population.native_digest
                    and binding["cache_owner_path_sha256"] == population.cache_digest
                    and binding["generic_receipt_implementations"] == {name: population.implementations[name]
                        for name in ("bridge_sha256", "receipt_contracts_sha256")}, "current native admission binding differs")
                reference = DoctorSealedReceiptRef.from_dict(binding["custody_ref"])
                _require(canonical_json(reference.to_dict()) == canonical_json(binding["custody_ref"]), "exact original custody reference required")
                entries.append(HeaderProofCacheInventoryEntry(key.key_id, expected[key.key_id], reference))
                guard = admission._guard
                _require(type(guard) is bridge._AdmissionGuard and guard.hammer is hammer
                         and guard.store is population.store and guard.gate is gate and guard.reference == reference,
                         "exact original native admission custody required")
                population.preflight += (_preflight_entry(population, guard.prepared, reference, key.key_id),)
                population.current()
            inventory = HeaderProofCacheInventory(task_cid=scoped_analysis.task_cid,
                repository_id=scoped_analysis.snapshot.roots.repository_id, scope_id=scoped_analysis.report["scope_id"],
                store_id=population.store.store_id, authority_id=population.store.authority_id,
                native_owner_path_sha256=population.native_digest, cache_owner_path_sha256=population.cache_digest,
                entries=tuple(entries), **population.implementations)
            population.current()
            for admission in admissions:
                admission.assert_current()
                population.current()
            population.current()
            reference = population.store.append(inventory)
            population.inventory, population.reference = inventory, custody._reference(reference, HeaderProofCacheInventory.SCHEMA)
            population.current()
            for admission in admissions:
                admission.assert_current()
                population.current()
            population.current()
            return population.reference
    except HeaderProofInventoryError:
        raise
    except Exception as exc:
        raise HeaderProofInventoryError("native header inventory persistence refused; existing owners may have been written") from exc


def load_header_proof_inventory(*, scoped_analysis, hammer, gate, preparations, inventory_ref,
                                expected_inventory_cid, expected_task_cid, expected_scope_id,
                                expected_native_owner_path_sha256, expected_cache_owner_path_sha256) -> tuple:
    """Reload a pinned population, preflight all entries, then freshly import it.

    The host supplies independently rebuilt preparations and existing owners.
    A retained inventory is neither current source admission nor positive proof.
    Empty inventories are explicit and run no kernel. A later failure returns no
    accepted tuple, but earlier valid imports may already have written the cache.
    """
    try:
        with public_replay_without_metadata():
            _cid(expected_inventory_cid); _cid(expected_task_cid); _cid(expected_scope_id)
            reference = custody._reference(inventory_ref, HeaderProofCacheInventory.SCHEMA)
            _require(reference.receipt_cid == expected_inventory_cid, "inventory reference differs from independently expected CID")
            _require(type(preparations) is tuple and len(preparations) <= MAX_KEYS
                     and all(type(item) is HeaderContractProofPreparation for item in preparations),
                     "bounded independently rebuilt preparation population required")
            keys = _keys(tuple(item.key for item in preparations), None)
            population = _Population(scoped_analysis=scoped_analysis, hammer=hammer, gate=gate,
                native_digest=expected_native_owner_path_sha256, cache_digest=expected_cache_owner_path_sha256)
            inventory = population.store.reload(reference, HeaderProofCacheInventory)
            _require(inventory.content_id == expected_inventory_cid and inventory.store_id == population.store.store_id
                     and inventory.authority_id == population.store.authority_id
                     and inventory.task_cid == expected_task_cid == scoped_analysis.task_cid
                     and inventory.scope_id == expected_scope_id == scoped_analysis.report["scope_id"]
                     and inventory.repository_id == scoped_analysis.snapshot.roots.repository_id
                     and inventory.native_owner_path_sha256 == population.native_digest
                     and inventory.cache_owner_path_sha256 == population.cache_digest
                     and all(getattr(inventory, name) == value for name, value in population.implementations.items()),
                     "pinned inventory differs from current independently admitted owners, source or implementation")
            population.inventory, population.reference = inventory, reference
            supplied = {item.key.key_id: item for item in preparations}
            _require(len(supplied) == len(preparations) == len(inventory.entries)
                     and set(supplied) == {item.key_id for item in inventory.entries}, "complete independently rebuilt population differs")
            preflight = []
            for entry in inventory.entries:
                prepared = supplied[entry.key_id]
                preflight.append(_preflight_entry(population, prepared, entry.custody_ref, entry.key_id))
            population.preflight = tuple(preflight)
            population.current()
            admissions = []
            for entry in inventory.entries:
                population.current()
                admission = bridge.import_header_proof_to_cache(prepared=supplied[entry.key_id], hammer=hammer,
                    custody_ref=entry.custody_ref, expected_key_id=entry.key_id, expected_receipt_id=entry.generic_receipt_id,
                    gate=gate, expected_owner_path_sha256=population.cache_digest)
                admission.assert_bound(gate=gate, scoped_analysis=scoped_analysis, expected_key_id=entry.key_id,
                    expected_receipt_id=entry.generic_receipt_id, expected_owner_path_sha256=population.cache_digest)
                binding = admission.inventory_binding(hammer=hammer, gate=gate, scoped_analysis=scoped_analysis,
                    expected_key_id=entry.key_id, expected_receipt_id=entry.generic_receipt_id,
                    expected_native_owner_path_sha256=population.native_digest,
                    expected_cache_owner_path_sha256=population.cache_digest)
                _require(binding["custody_ref"] == entry.custody_ref.to_dict(), "fresh import substituted original native custody")
                population.current()
                admissions.append(admission)
            for admission in admissions:
                admission.assert_current()
                population.current()
            population.current()
            return tuple(admissions)
    except HeaderProofInventoryError:
        raise
    except Exception as exc:
        raise HeaderProofInventoryError("native header inventory restart refused; earlier valid imports may have written existing owners") from exc


__all__ = ["HeaderProofInventoryError", "HeaderProofCacheInventoryEntry", "HeaderProofCacheInventory",
           "persist_header_proof_inventory", "load_header_proof_inventory"]
