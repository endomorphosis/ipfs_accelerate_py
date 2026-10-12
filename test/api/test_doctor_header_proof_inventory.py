"""Explicit sealed inventory restart under independently rebuilt host inputs.

Authored cases use the qualified bridge fixture's pinned transcripts and real
SQLite/DuckDB owners. The separate real-prover case delegates to actual Lean/Z3.
Inventory custody carries no proof, execution, mutation or completion authority.
"""

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_doctor_contract_proof import _scoped
from test.api.test_doctor_contract_proof_custody import _profiles
from test.api.test_doctor_header_cache_bridge import authored, _assert_local_receipt, _forbid_owners, _import, _prepare
from test.api.test_doctor_header_contracts import PROGRAM, analyze
from test.api.test_doctor_task_workflow import _provers
from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import identify_strict_artifact
from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import DeterministicDoctorHammer, DoctorExecutableRole
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import (
    DoctorIdentityBinding, DoctorProofCacheGate, DoctorSealedReceiptRef, DoctorSealedReceiptStore,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import canonical_json, content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof as factory
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof_custody as custody
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_header_cache_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_header_proof_inventory as inventory
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


def _path_pin(owner):
    return hashlib.sha256(str(owner.path if type(owner) is DoctorSealedReceiptStore else owner.db_path).encode()).hexdigest()


def _persist(values, **changes):
    admissions = tuple(sorted(values["admissions"], key=lambda item: item.key.key_id))
    options = {"scoped_analysis": values["scoped"], "hammer": values["hammer"], "gate": values["gate"],
        "admissions": admissions, "expected_key_ids": tuple(item.key.key_id for item in admissions),
        "expected_receipt_ids": tuple((item.key.key_id, item.receipt.receipt_id) for item in admissions),
        "expected_native_owner_path_sha256": values["native_pin"],
        "expected_cache_owner_path_sha256": values["owner_pin"]}
    options.update(changes)
    return inventory.persist_header_proof_inventory(**options)


def _load(values, **changes):
    options = {"scoped_analysis": values["scoped"], "hammer": values["hammer"], "gate": values["gate"],
        "preparations": values["preparations"], "inventory_ref": values["inventory_ref"],
        "expected_inventory_cid": values["inventory_cid"], "expected_task_cid": values["scoped"].task_cid,
        "expected_scope_id": values["scoped"].report["scope_id"],
        "expected_native_owner_path_sha256": values["native_pin"],
        "expected_cache_owner_path_sha256": values["owner_pin"]}
    options.update(changes)
    return inventory.load_header_proof_inventory(**options)


@pytest.fixture
def retained(authored):
    admission = _import(authored)
    values = {**authored, "admissions": (admission,), "preparations": (authored["prepared"],),
              "native_pin": _path_pin(authored["store"])}
    reference = _persist(values)
    values.update(inventory_ref=reference, inventory_cid=reference.receipt_cid)
    return values


def _forbid_generic_put(values, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("invalid inventory/receipt must refuse before a generic positive write")
    monkeypatch.setattr(values["gate"], "put", forbidden)


def _second_preparation(values):
    policy = DoctorIdentityBinding.from_identity(identify_strict_artifact({
        "schema": "authored-header-inventory-policy@2", "profile_version": 2,
        "native_kernel_required": True, "isolation_verified": False}),
        logical_id="authored-header-inventory-policy:v2")
    profiles = replace(values["profiles"], policy=policy)
    prepared = _prepare(values, proof_key_profiles=profiles)
    assert prepared.key.key_id != values["prepared"].key.key_id
    assert prepared.solver_pin == values["prepared"].solver_pin
    assert prepared.kernel_pin == values["prepared"].kernel_pin
    assert prepared.bounds == values["prepared"].bounds
    return prepared


def _add_second_entry(values):
    second = _second_preparation(values)
    hammer = values["hammer"]
    with public_replay_without_metadata():
        proof = hammer.verify_authoritative(second.theorem, second.lowering,
            solver_pin=second.solver_pin, kernel_pin=second.kernel_pin, current_roots=second.theorem.roots,
            eligible_consequence_refs=(second.theorem.consequence_ref,))
        reference = custody.persist_header_proof_key_custody(prepared=second, proof=proof, hammer=hammer)
        admission = _import(values, prepared=second, custody_ref=reference, expected_key_id=second.key.key_id)
    result = {**values, "preparations": (values["prepared"], second),
              "admissions": (*values["admissions"], admission)}
    manifest = _persist(result)
    result.update(inventory_ref=manifest, inventory_cid=manifest.receipt_cid)
    return result


def test_persisted_inventory_uses_same_sealed_owner_without_kernel_or_generic_put(retained, monkeypatch):
    before = list(retained["calls"])
    _forbid_owners(monkeypatch)
    _forbid_generic_put(retained, monkeypatch)
    reference = _persist(retained)
    assert retained["calls"] == before
    assert reference.store_id == retained["store"].store_id
    assert reference.receipt_cid == retained["inventory_cid"]
    assert reference.entry_cid != retained["inventory_ref"].entry_cid
    typed = retained["store"].reload(reference, inventory.HeaderProofCacheInventory)
    body = typed.to_dict()
    assert body["task_cid"] == retained["scoped"].task_cid
    assert body["scope_id"] == retained["scoped"].report["scope_id"]
    assert body["repository_id"] == retained["prepared"].theorem.roots.repository_id
    assert body["native_owner_path_sha256"] == retained["native_pin"]
    assert body["cache_owner_path_sha256"] == retained["owner_pin"]
    entry = body["entries"][0]
    assert entry["key_id"] == retained["prepared"].key.key_id
    assert entry["generic_receipt_id"] == retained["admissions"][0].receipt.receipt_id
    assert entry["custody_ref"] == retained["custody_ref"].to_dict()
    for name in ("proof_authority", "completion_authority", "execution_authority", "mutation_authority"):
        assert body[name] is False
    assert inventory.HeaderProofCacheInventory.from_dict(typed.to_record()).to_dict() == body
    detached = typed.to_dict()
    detached["entries"][0]["generic_receipt_id"] = "authored-corrupted-copy"
    assert typed.to_dict() == body


def test_cold_restart_rebuilds_host_inputs_and_replays_one_kernel_before_stable_local_cache_admission(retained, monkeypatch):
    prepared = _prepare(retained)
    store = DoctorSealedReceiptStore(retained["store"].path, authority_id=retained["store"].authority_id)
    hammer = DeterministicDoctorHammer(bounds=prepared.bounds, authoritative_store=store,
        trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
    gate = DoctorProofCacheGate(cache=FormalVerificationCache(retained["gate"].db_path, exact_path=True))
    before = len(retained["calls"])
    _forbid_owners(monkeypatch)
    restored = _load(retained, preparations=(prepared,), hammer=hammer, gate=gate)
    assert type(restored) is tuple and len(restored) == 1
    assert retained["calls"][before:] == [DoctorExecutableRole.KERNEL]
    current = restored[0]
    assert current is not retained["admissions"][0]
    assert current.receipt.receipt_id == retained["admissions"][0].receipt.receipt_id
    _assert_local_receipt(prepared, current.receipt)
    assert current.revalidate_for_render().hit and current.revalidate_for_commit().hit
    assert retained["calls"][before:] == [DoctorExecutableRole.KERNEL]
    assert not gate.is_quarantined(current.key)


@pytest.mark.parametrize("poison", ["unknown", "authority", "version", "duplicate-entry"])
def test_inventory_decoder_rejects_closed_schema_and_population_poison(retained, poison):
    body = retained["store"].reload(retained["inventory_ref"], inventory.HeaderProofCacheInventory).to_dict()
    body.pop("content_id", None)
    if poison == "unknown":
        body["unreviewed_profile_loader"] = "must never become a locator"
    elif poison == "authority":
        body["completion_authority"] = True
    elif poison == "version":
        body["contract_version"] = 2
    else:
        body["entries"].append(deepcopy(body["entries"][0]))
    with pytest.raises(ValueError):
        inventory.HeaderProofCacheInventory.from_dict(body)


def test_inert_inventory_declarations_enforce_entry_count_and_aggregate_byte_bounds(retained):
    typed = retained["store"].reload(retained["inventory_ref"], inventory.HeaderProofCacheInventory)
    # These synthetic canonical entries are declarations only, never persisted
    # or admitted as a proof. They isolate the two manifest limits.
    entries = tuple(inventory.HeaderProofCacheInventoryEntry(
        key_id="doctor-proof-cache-key:sha256:" + format(index + 1, "064x"),
        generic_receipt_id=content_identity({"authored-declaration": index}),
        custody_ref=retained["custody_ref"]) for index in range(inventory.MAX_KEYS + 1))
    bounded = replace(typed, entries=entries[:-1])
    assert len(bounded.entries) == inventory.MAX_KEYS
    with pytest.raises(ValueError):
        replace(typed, entries=entries)
    large_reference = replace(retained["custody_ref"], entry_id="e" * 10_000)
    large_entries = tuple(replace(entry, custody_ref=large_reference) for entry in entries[:-1])
    assert len(canonical_json(large_entries[0].to_dict()).encode()) < custody.MAX_REFERENCE_BYTES
    with pytest.raises(ValueError):
        replace(typed, entries=large_entries)


@pytest.mark.parametrize("wrong", ["inventory-cid", "task", "scope", "native-path", "cache-path"])
def test_independent_inventory_and_host_pins_refuse_before_kernel_or_generic_write(retained, monkeypatch, wrong):
    changes = {
        "inventory-cid": {"expected_inventory_cid": content_identity({"authored": "foreign-inventory"})},
        "task": {"expected_task_cid": "FOREIGN_TASK"},
        "scope": {"expected_scope_id": content_identity({"authored": "foreign-scope"})},
        "native-path": {"expected_native_owner_path_sha256": "0" * 64},
        "cache-path": {"expected_cache_owner_path_sha256": "0" * 64},
    }[wrong]
    before = list(retained["calls"])
    _forbid_owners(monkeypatch)
    _forbid_generic_put(retained, monkeypatch)
    with pytest.raises(ValueError):
        _load(retained, **changes)
    assert retained["calls"] == before


def test_inventory_sql_preimage_corruption_refuses_before_any_kernel(retained, monkeypatch):
    with sqlite3.connect(retained["store"].path) as connection:
        row = connection.execute("SELECT receipt_json FROM doctor_sealed_receipts WHERE sequence=?",
                                 (retained["inventory_ref"].sequence,)).fetchone()
        body = json.loads(row[0])
        body["scope_id"] = content_identity({"authored": "tampered-scope"})
        connection.execute("UPDATE doctor_sealed_receipts SET receipt_json=? WHERE sequence=?",
            (canonical_json(body), retained["inventory_ref"].sequence))
    before = list(retained["calls"])
    _forbid_owners(monkeypatch)
    _forbid_generic_put(retained, monkeypatch)
    with pytest.raises(ValueError):
        _load(retained)
    assert retained["calls"] == before


@pytest.mark.parametrize("population", ["missing", "extra"])
def test_restore_requires_the_complete_independently_rebuilt_preparation_population(retained, monkeypatch, population):
    preparations = () if population == "missing" else (retained["prepared"], _second_preparation(retained))
    before = list(retained["calls"])
    _forbid_owners(monkeypatch)
    _forbid_generic_put(retained, monkeypatch)
    with pytest.raises(ValueError):
        _load(retained, preparations=preparations)
    assert retained["calls"] == before


def test_second_valid_entry_profile_poison_preflights_the_whole_inventory_before_first_kernel(retained, monkeypatch):
    values = _add_second_entry(retained)
    typed = retained["store"].reload(values["inventory_ref"], inventory.HeaderProofCacheInventory)
    assert len(typed.entries) == 2
    ordered = sorted(values["preparations"], key=lambda item: item.key.key_id)
    assert [entry.key_id for entry in typed.entries] == [item.key.key_id for item in ordered]
    object.__setattr__(ordered[-1].proof_key_profiles, "policy", retained["profiles"].environment)
    before = list(retained["calls"])
    _forbid_owners(monkeypatch)
    _forbid_generic_put(retained, monkeypatch)
    with pytest.raises(ValueError):
        _load(values)
    assert retained["calls"] == before


def test_later_valid_but_wrong_receipt_id_refuses_before_its_put_without_rolling_back_first_import(retained, monkeypatch):
    values = _add_second_entry(retained)
    typed = values["store"].reload(values["inventory_ref"], inventory.HeaderProofCacheInventory)
    last = typed.entries[-1]
    original_receipt = values["gate"].lookup(next(
        prepared.key for prepared in values["preparations"] if prepared.key.key_id == last.key_id)).receipt
    poisoned = replace(typed, entries=(*typed.entries[:-1], replace(last,
        generic_receipt_id=content_identity({"authored": "wrong-second-valid-receipt-cid"}))))
    reference = values["store"].append(poisoned)
    values.update(inventory_ref=reference, inventory_cid=reference.receipt_cid)
    puts = []
    original_put = values["gate"].put
    def count_put(key, receipt, **kwargs):
        puts.append((key.key_id, receipt.receipt_id))
        return original_put(key, receipt, **kwargs)
    monkeypatch.setattr(values["gate"], "put", count_put)
    before = len(values["calls"])
    _forbid_owners(monkeypatch)
    with pytest.raises(ValueError):
        _load(values)
    assert values["calls"][before:] == [DoctorExecutableRole.KERNEL, DoctorExecutableRole.KERNEL]
    assert puts == [(typed.entries[0].key_id, typed.entries[0].generic_receipt_id)]
    key = next(prepared.key for prepared in values["preparations"] if prepared.key.key_id == last.key_id)
    assert not values["gate"].is_quarantined(key)
    assert values["gate"].lookup(key).receipt.receipt_id == original_receipt.receipt_id


def test_duplicate_admissions_cannot_append_a_new_inventory_row(retained, monkeypatch):
    original_sequence = retained["inventory_ref"].sequence
    before = list(retained["calls"])
    _forbid_owners(monkeypatch)
    _forbid_generic_put(retained, monkeypatch)
    def forbidden(*args, **kwargs):
        pytest.fail("malformed admission population must refuse before sealed append")
    monkeypatch.setattr(retained["store"], "append", forbidden)
    with pytest.raises(ValueError):
        _persist(retained, admissions=retained["admissions"] * 2)
    assert retained["calls"] == before
    with sqlite3.connect(retained["store"].path) as connection:
        assert connection.execute("SELECT MAX(sequence) FROM doctor_sealed_receipts").fetchone()[0] == original_sequence


@pytest.mark.parametrize("implementation", ["emitter", "codec", "inventory"])
def test_historical_inventory_implementation_drift_refuses_before_fresh_kernel_or_cache_put(retained, monkeypatch, implementation):
    module = {"emitter": bridge, "codec": bridge.receipt_contracts, "inventory": inventory}[implementation]
    copied = retained["tmp_path"] / ("authored-inventory-copy-" + implementation + ".py")
    copied.write_bytes(Path(module.__file__).read_bytes() + b"\n# authored implementation drift\n")
    monkeypatch.setattr(module, "__file__", str(copied))
    before = list(retained["calls"])
    _forbid_owners(monkeypatch)
    _forbid_generic_put(retained, monkeypatch)
    with pytest.raises(ValueError):
        _load(retained)
    assert retained["calls"] == before
    assert not retained["gate"].is_quarantined(retained["prepared"].key)


def test_explicit_empty_inventory_persists_and_restores_without_kernel_or_generic_put(retained, monkeypatch):
    before = list(retained["calls"])
    _forbid_owners(monkeypatch)
    _forbid_generic_put(retained, monkeypatch)
    reference = _persist(retained, admissions=(), expected_key_ids=(), expected_receipt_ids=())
    values = {**retained, "inventory_ref": reference, "inventory_cid": reference.receipt_cid}
    typed = retained["store"].reload(reference, inventory.HeaderProofCacheInventory)
    assert typed.entries == ()
    assert _load(values, preparations=()) == ()
    assert retained["calls"] == before


def test_invalid_receipt_pin_type_refuses_before_cache_path_stat_or_kernel(authored, monkeypatch):
    before = list(authored["calls"])
    original_stat = Path.stat
    def stat(path, *args, **kwargs):
        if path == authored["gate"].db_path:
            pytest.fail("invalid expected receipt identity must refuse before cache filesystem observation")
        return original_stat(path, *args, **kwargs)
    monkeypatch.setattr(Path, "stat", stat)
    _forbid_owners(monkeypatch)
    _forbid_generic_put(authored, monkeypatch)
    with pytest.raises(ValueError):
        _import(authored, expected_receipt_id=True)
    assert authored["calls"] == before


def test_wrong_bounded_expected_receipt_cid_replays_kernel_but_never_puts_or_quarantines(authored, monkeypatch):
    before = len(authored["calls"])
    _forbid_owners(monkeypatch)
    _forbid_generic_put(authored, monkeypatch)
    with pytest.raises(ValueError):
        _import(authored, expected_receipt_id=content_identity({"authored": "wrong-generic-receipt"}))
    assert authored["calls"][before:] == [DoctorExecutableRole.KERNEL]
    assert not authored["gate"].is_quarantined(authored["prepared"].key)


def test_real_lean_z3_inventory_cold_restore_runs_one_actual_kernel_and_preserves_local_scope(scenario, tmp_path, monkeypatch):
    inputs, scoped = _scoped(scenario, tmp_path)
    inputs = _provers(inputs)
    candidate = analyze().candidate
    values = {"scoped": scoped, "contract": candidate.contract,
        "candidate_cid": content_identity(candidate.to_dict()), "profiles": _profiles(),
        "lean": inputs["kernel_executable"], "z3": inputs["solver_executable"]}
    report, proof = factory.prove_header_contract(**{
        name: values[name] for name in ("scoped", "contract", "candidate_cid", "lean", "z3")},
        proof_key_profiles=values["profiles"], state=tmp_path / "actual-inventory-proof")
    assert proof is not None and proof.mutation_capable, report
    prepared = _prepare(values)
    store = proof.authority_store
    hammer = DeterministicDoctorHammer(bounds=prepared.bounds, authoritative_store=store,
        trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
    gate = DoctorProofCacheGate(tmp_path / "actual-inventory-existing-cache.duckdb")
    values.update(prepared=prepared, preparations=(prepared,), store=store, hammer=hammer, gate=gate,
        custody_ref=DoctorSealedReceiptRef.from_dict(report["native_key_custody"]["ref"]),
        native_pin=_path_pin(store), owner_pin=_path_pin(gate))
    admission = _import(values)
    values["admissions"] = (admission,)
    reference = _persist(values)
    values.update(inventory_ref=reference, inventory_cid=reference.receipt_cid)
    # Reconnect existing owners explicitly before every constructor is banned.
    current = _prepare(values)
    reopened = DoctorSealedReceiptStore(store.path, authority_id=store.authority_id)
    restarted_hammer = DeterministicDoctorHammer(bounds=current.bounds, authoritative_store=reopened,
        trusted_executable_pins=(current.solver_pin, current.kernel_pin))
    restarted_gate = DoctorProofCacheGate(cache=FormalVerificationCache(gate.db_path, exact_path=True))
    actual_run = DeterministicDoctorHammer._run_pinned
    calls = []
    def spy(self, pin, request):
        calls.append(pin.role)
        return actual_run(self, pin, request)
    monkeypatch.setattr(DeterministicDoctorHammer, "_run_pinned", spy)
    _forbid_owners(monkeypatch)
    restored = _load(values, preparations=(current,), hammer=restarted_hammer, gate=restarted_gate)
    assert calls == [DoctorExecutableRole.KERNEL]
    assert len(restored) == 1 and restored[0].receipt.receipt_id == admission.receipt.receipt_id
    _assert_local_receipt(current, restored[0].receipt)
    assert restored[0].revalidate_for_render().hit and restored[0].revalidate_for_commit().hit
    assert calls == [DoctorExecutableRole.KERNEL]
