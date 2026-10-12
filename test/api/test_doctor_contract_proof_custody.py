"""Authored kernel transcripts exercise custody; no theorem prover is run.

The source admission, preparation, complete key producer, sealed SQLite lineage
and restart replay are native. Pinned subprocess output is explicitly mocked;
these tests neither prove the header theorem nor qualify production coverage.
"""
from dataclasses import replace
import json
from pathlib import Path
import sqlite3

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_doctor_contract_proof import _scoped
from test.api.test_doctor_header_contracts import PROGRAM, analyze
from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import identify_strict_artifact
from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import (
    DeterministicDoctorHammer, DoctorAuthoritativeProofReceipt, DoctorExecutableRole,
)
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import (
    DoctorIdentityBinding, DoctorProofCacheGate, DoctorSealedReceiptError,
    DoctorSealedReceiptRef, DoctorSealedReceiptStore,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    ContractValidationError, ProofReceipt, canonical_json, content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof as factory
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_contract_key_custody import HeaderContractProofKeyProfiles
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof_custody as custody
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import public_replay_without_metadata


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


def _profiles():
    return HeaderContractProofKeyProfiles(**{
        name: DoctorIdentityBinding.from_identity(identify_strict_artifact({
            "schema": "authored-header-custody-" + name + "@1",
            "profile": "authored-no-real-prover", "isolation_verified": False}),
            logical_id="authored-header-custody:" + name)
        for name in HeaderContractProofKeyProfiles.__dataclass_fields__})


def _prepare(authored, **changes):
    values = {"scoped": authored["scoped"], "contract": authored["contract"],
              "candidate_cid": authored["candidate_cid"], "lean": authored["lean"],
              "z3": authored["z3"], "proof_key_profiles": authored["profiles"]}
    values.update(changes)
    return factory.prepare_header_contract_proof(**values)


@pytest.fixture
def authored(scenario, tmp_path, monkeypatch):
    _inputs, scoped = _scoped(scenario, tmp_path)
    candidate = analyze().candidate
    values = {"scoped": scoped, "contract": candidate.contract,
              "candidate_cid": content_identity(candidate.to_dict()), "profiles": _profiles(),
              "repository": scenario["repository"], "calls": []}
    for name in ("lean", "z3"):
        path = tmp_path / ("authored-custody-" + name)
        path.write_text("#!/bin/sh\n# inert authored executable; never run\nexit 99\n")
        path.chmod(0o755)
        values[name] = path
    prepared = _prepare(values)

    def authored_pinned_output(self, pin, request):
        # This replaces process execution, not native sealed-store decoding.
        values["calls"].append(pin.role)
        theorem, lowering = request["theorem"], request["lowering"]
        output = {"theorem_cid": content_identity(theorem), "lowering_cid": content_identity(lowering),
                  "property_id": theorem["property_id"], "consequence_ref": theorem["consequence_ref"],
                  "premise_ids": theorem["premise_ids"]}
        if pin.role is DoctorExecutableRole.SOLVER:
            output.update(status="proved", proof_object=theorem["theorem_body"])
        else:
            native = request["native_receipt"]
            assert native["proof_object"] == theorem["theorem_body"]
            output.update(status="kernel_verified", native_receipt_cid=content_identity(native),
                          proof_object_cid=content_identity({"proof_object": native["proof_object"]}),
                          kernel_id="authored-header-custody-kernel")
        transcript = {"executable_sha256": pin.executable_sha256, "command_id": pin.command_id,
            "stdin_cid": content_identity({"bytes_hex": canonical_json(request).encode().hex()}),
            "stdout_cid": content_identity({"bytes_hex": canonical_json(output).encode().hex()}),
            "stderr_cid": content_identity({"bytes_hex": ""}), "exit_code": 0}
        transcript["transcript_cid"] = content_identity(transcript)
        return output, transcript

    monkeypatch.setattr(DeterministicDoctorHammer, "_run_pinned", authored_pinned_output)
    with public_replay_without_metadata():
        store = DoctorSealedReceiptStore(tmp_path / "authored-proof.sqlite",
                                        authority_id="authored-header-custody-owner")
        hammer = DeterministicDoctorHammer(bounds=prepared.bounds, authoritative_store=store,
            trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
        proof = hammer.verify_authoritative(prepared.theorem, prepared.lowering,
            solver_pin=prepared.solver_pin, kernel_pin=prepared.kernel_pin,
            current_roots=prepared.theorem.roots,
            eligible_consequence_refs=(prepared.theorem.consequence_ref,))
    assert proof.mutation_capable
    values.update(prepared=prepared, store=store, hammer=hammer, proof=proof)
    return values


def _persist(authored):
    return custody.persist_header_proof_key_custody(prepared=authored["prepared"],
        proof=authored["proof"], hammer=authored["hammer"])


def _replay(authored, reference, **changes):
    values = {"prepared": authored["prepared"], "hammer": authored["hammer"],
              "custody_ref": reference, "expected_key_id": authored["prepared"].key.key_id}
    values.update(changes)
    return custody.replay_header_proof_key_custody(**values)


def _forbid_owners(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("custody API must reuse its existing native owner")
    for cls in (DoctorSealedReceiptStore, DoctorProofCacheGate, FormalVerificationCache):
        monkeypatch.setattr(cls, "__init__", forbidden)


def test_same_owner_retains_complete_key_without_generic_proof_admission(authored, monkeypatch):
    _forbid_owners(monkeypatch)
    before = list(authored["calls"])
    reference = _persist(authored)
    assert authored["calls"] == before  # Persistence never reexecutes a prover.
    retained = authored["store"].reload(reference, custody.HeaderContractProofKeyCustody)
    body = retained.to_dict()
    assert body["preparation"] == authored["prepared"].custody_payload()
    assert body["proof_ref"] == authored["proof"].authority_store_ref.to_dict()
    assert reference.store_id == authored["proof"].authority_store_ref.store_id
    assert reference.sequence == authored["proof"].authority_store_ref.sequence + 1
    assert all(body[name] is False for name in custody._FLAGS)
    assert custody.HeaderContractProofKeyCustody.from_dict(retained.to_record()) == retained
    for foreign_schema in (body, authored["proof"].to_record()):
        with pytest.raises(ContractValidationError):
            ProofReceipt.from_dict(foreign_schema)
    replayed = _replay(authored, reference)
    assert type(replayed) is DoctorAuthoritativeProofReceipt and replayed.mutation_capable
    assert replayed.content_id == authored["proof"].content_id
    assert authored["calls"] == [*before, DoctorExecutableRole.KERNEL]


def test_restart_requires_independently_rebuilt_preparation_and_existing_sealed_lineage(authored, monkeypatch):
    reference = _persist(authored)
    original = authored["store"]
    # Explicit test host reconnects its admitted original SQLite/key owner.
    reopened = DoctorSealedReceiptStore(original.path, authority_id=original.authority_id)
    prepared = _prepare(authored)
    hammer = DeterministicDoctorHammer(bounds=prepared.bounds, authoritative_store=reopened,
        trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
    decoded_report = DoctorAuthoritativeProofReceipt.from_dict(authored["proof"].to_dict())
    assert decoded_report.authority_store_ref is None and not decoded_report.mutation_capable
    _forbid_owners(monkeypatch)
    before = len(authored["calls"])
    replayed = _replay(authored, reference, prepared=prepared, hammer=hammer,
                        expected_key_id=prepared.key.key_id)
    assert replayed.authority_store is reopened and replayed.mutation_capable
    assert replayed.authority_store_ref == authored["proof"].authority_store_ref
    assert authored["calls"][before:] == [DoctorExecutableRole.KERNEL]


def test_production_header_workflow_retains_restart_record_and_same_owner_key_custody(authored, tmp_path, monkeypatch):
    state = tmp_path / "authored-production-proof"
    report, proof = factory.prove_header_contract(scoped=authored["scoped"],
        contract=authored["contract"], candidate_cid=authored["candidate_cid"], state=state,
        lean=authored["lean"], z3=authored["z3"], proof_key_profiles=authored["profiles"])
    assert proof is not None and proof.mutation_capable
    retained_report = json.loads((state / "proof-report.json").read_bytes())
    assert retained_report["proof"]["authority_store_ref"] == proof.authority_store_ref.to_dict()
    entry = retained_report["native_key_custody"]
    reference = DoctorSealedReceiptRef.from_dict(entry["ref"])
    assert entry["key_id"] == authored["prepared"].key.key_id
    assert entry["generic_cache_admission_supported"] is False
    assert reference.store_id == proof.authority_store_ref.store_id
    prepared = _prepare(authored)
    hammer = DeterministicDoctorHammer(bounds=prepared.bounds, authoritative_store=proof.authority_store,
        trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
    _forbid_owners(monkeypatch)
    replayed = _replay(authored, reference, prepared=prepared, hammer=hammer,
                        expected_key_id=entry["key_id"])
    assert replayed.content_id == report["proof_receipt_id"]


@pytest.mark.parametrize("ref_drift", ["seal", "store", "authority", "schema"])
def test_forged_native_custody_reference_never_replays_kernel(authored, ref_drift):
    reference = _persist(authored)
    changes = {"seal": "hmac-sha256:" + "0" * 64, "store": "foreign-store",
               "authority": "foreign-authority", "schema": DoctorAuthoritativeProofReceipt.SCHEMA}
    field = {"store": "store_id", "authority": "authority_id", "schema": "receipt_schema"}.get(ref_drift, ref_drift)
    forged = replace(reference, **{field: changes[ref_drift]})
    before = list(authored["calls"])
    with pytest.raises((ValueError, DoctorSealedReceiptError)):
        _replay(authored, forged)
    assert authored["calls"] == before


def test_durable_custody_bytes_are_checked_before_kernel_replay(authored):
    reference = _persist(authored)
    with sqlite3.connect(authored["store"].path) as connection:
        original = connection.execute("SELECT receipt_json FROM doctor_sealed_receipts WHERE sequence=?",
                                      (reference.sequence,)).fetchone()[0]
        marker = "authored-header-custody:registry"
        assert marker in original
        connection.execute("UPDATE doctor_sealed_receipts SET receipt_json=replace(receipt_json, ?, ?) "
                           "WHERE sequence=?", (marker, "forged-registry", reference.sequence))
        changed = connection.execute("SELECT receipt_json FROM doctor_sealed_receipts WHERE sequence=?",
                                     (reference.sequence,)).fetchone()[0]
        assert changed != original and "forged-registry" in changed
    before = list(authored["calls"])
    with pytest.raises(DoctorSealedReceiptError):
        _replay(authored, reference)
    assert authored["calls"] == before


@pytest.mark.parametrize("change", ["source", "profile", "bounds", "registry-missing", "registry-wrong", "expected-key", "missing-expected-key"])
def test_current_input_drift_refuses_before_kernel_replay(authored, change):
    reference = _persist(authored)
    kwargs = {}
    if change == "source":
        (authored["repository"] / "answer.py").write_text(PROGRAM + "\n# source changed\n")
    elif change == "profile":
        profiles = replace(authored["profiles"], registry=DoctorIdentityBinding.from_identity(
            identify_strict_artifact({"schema": "authored-next-registry@2", "operators": []}),
            logical_id="authored-next-registry"))
        prepared = _prepare(authored, proof_key_profiles=profiles)
        kwargs.update(prepared=prepared, expected_key_id=prepared.key.key_id)
    elif change == "bounds":
        authored["hammer"]._bounds = replace(authored["prepared"].bounds, wall_time_ms=60_001)
    elif change.startswith("registry"):
        pins = () if change == "registry-missing" else (
            replace(authored["prepared"].kernel_pin, verifier_id="foreign-kernel"),)
        kwargs["hammer"] = DeterministicDoctorHammer(bounds=authored["prepared"].bounds,
            authoritative_store=authored["store"], trusted_executable_pins=pins)
    elif change == "expected-key":
        kwargs["expected_key_id"] = "doctor-proof-cache-key:sha256:" + "0" * 64
    else:
        kwargs["expected_key_id"] = None
    before = list(authored["calls"])
    with pytest.raises(ValueError):
        _replay(authored, reference, **kwargs)
    assert authored["calls"] == before


@pytest.mark.parametrize("missing", ["key", "profiles", "store"])
def test_incomplete_host_inputs_cannot_create_key_custody(authored, missing):
    if missing == "store":
        authored["hammer"]._authoritative_store = None
    else:
        authored["prepared"] = replace(authored["prepared"],
            **{"key" if missing == "key" else "proof_key_profiles": None})
    before = list(authored["calls"])
    with pytest.raises(ValueError):
        _persist(authored)
    assert authored["calls"] == before


@pytest.mark.parametrize("during", ["source", "database", "seal-key"])
def test_drift_during_fresh_kernel_replay_cannot_return_authoritative_proof(authored, monkeypatch, during):
    reference = _persist(authored)
    original = authored["hammer"]._execute_kernel
    def drift(*args, **kwargs):
        result = original(*args, **kwargs)
        if during == "source":
            (authored["repository"] / "answer.py").write_text(PROGRAM + "\n# replay drift\n")
        else:
            path = authored["store"].path if during == "database" else authored["store"]._key_path
            prior = path.with_name(path.name + ".fixture-original")
            path.rename(prior)
            path.write_bytes(prior.read_bytes())
            path.chmod(0o600)
            assert path.stat().st_ino != prior.stat().st_ino
        return result
    monkeypatch.setattr(authored["hammer"], "_execute_kernel", drift)
    with pytest.raises(ValueError):
        _replay(authored, reference)


@pytest.mark.parametrize("invalid", ["unknown", "authority", "incomplete-key", "proof-ref", "identity", "oversized",
                                     "nested-unknown", "nested-authority", "missing-preparation"])
def test_custody_decoder_is_closed_and_bounded(authored, invalid):
    reference = _persist(authored)
    retained = authored["store"].reload(reference, custody.HeaderContractProofKeyCustody)
    body = retained.to_record()
    if invalid == "unknown":
        body["unreviewed"] = True
    elif invalid == "authority":
        body["proof_authority"] = True
    elif invalid == "incomplete-key":
        body["preparation"]["key"].pop("premises")
    elif invalid == "proof-ref":
        body["proof_ref"].pop("schema")
    elif invalid == "identity":
        body["content_id"] = "forged"
    elif invalid == "nested-unknown":
        body["preparation"]["unreviewed"] = True
    elif invalid == "nested-authority":
        body["preparation"]["proof_authority"] = True
    elif invalid == "missing-preparation":
        body["preparation"].pop("profiles")
    else:
        body["preparation"]["authored_extra"] = "x" * custody.MAX_PREPARATION_BYTES
    with pytest.raises(ValueError):
        custody.HeaderContractProofKeyCustody.from_dict(body)
