"""Local scoped proof admission into existing native owners only.

Most cases use explicitly authored pinned transcripts and real temporary SQLite
and DuckDB owners. The separate real-prover case runs actual Lean and Z3. Neither
lane establishes whole-program proof or qualifies Terminal prompt changes.
"""

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_doctor_contract_proof import _scoped
from test.api.test_doctor_contract_proof_custody import _profiles
from test.api.test_doctor_header_contracts import PROGRAM, analyze
from test.api.test_doctor_task_workflow import _provers
from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import (
    DeterministicDoctorHammer, DoctorExecutableRole,
)
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import (
    DoctorCacheLookupResult, DoctorCacheStage, DoctorProofCacheGate,
    DoctorSealedReceiptRef, DoctorSealedReceiptStore,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel, ProofReceipt, ProofVerdict, ScopedProofReceipt, canonical_json, content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof as factory
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof_custody as custody
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_header_cache_bridge as bridge
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import (
    SupervisorMetaIndex, public_replay_without_metadata,
)


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


def _path_pin(gate):
    return hashlib.sha256(str(gate.db_path).encode()).hexdigest()


def _prepare(values, **changes):
    options = {"scoped": values["scoped"], "contract": values["contract"],
        "candidate_cid": values["candidate_cid"], "lean": values["lean"], "z3": values["z3"],
        "proof_key_profiles": values["profiles"]}
    options.update(changes)
    return factory.prepare_header_contract_proof(**options)


def _import(values, **changes):
    options = {"prepared": values["prepared"], "hammer": values["hammer"],
        "custody_ref": values["custody_ref"], "expected_key_id": values["prepared"].key.key_id,
        "gate": values["gate"], "expected_owner_path_sha256": values["owner_pin"]}
    options.update(changes)
    return bridge.import_header_proof_to_cache(**options)


def _forbid_owners(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("bridge must reuse explicitly admitted existing native owners")
    for cls in (DoctorSealedReceiptStore, DoctorProofCacheGate, FormalVerificationCache, SupervisorMetaIndex):
        monkeypatch.setattr(cls, "__init__", forbidden)


@pytest.fixture
def authored(scenario, tmp_path, monkeypatch):
    _inputs, scoped = _scoped(scenario, tmp_path)
    candidate = analyze().candidate
    values = {"scoped": scoped, "contract": candidate.contract,
        "candidate_cid": content_identity(candidate.to_dict()), "profiles": _profiles(),
        "repository": scenario["repository"], "calls": [], "tmp_path": tmp_path}
    for name in ("lean", "z3"):
        path = tmp_path / ("authored-bridge-" + name)
        path.write_text("#!/bin/sh\n# authored inert executable; never run\nexit 99\n")
        path.chmod(0o755)
        values[name] = path
    prepared = _prepare(values)
    cfg = json.loads(prepared.kernel_pin.argv[-2])

    def authored_pinned_output(self, pin, request):
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
                kernel_id=cfg["kernel_id"])
        transcript = {"executable_sha256": pin.executable_sha256, "command_id": pin.command_id,
            "stdin_cid": content_identity({"bytes_hex": canonical_json(request).encode().hex()}),
            "stdout_cid": content_identity({"bytes_hex": canonical_json(output).encode().hex()}),
            "stderr_cid": content_identity({"bytes_hex": ""}), "exit_code": 0}
        transcript["transcript_cid"] = content_identity(transcript)
        return output, transcript

    monkeypatch.setattr(DeterministicDoctorHammer, "_run_pinned", authored_pinned_output)
    with public_replay_without_metadata():
        store = DoctorSealedReceiptStore(tmp_path / "authored-native-proof.sqlite",
                                        authority_id="authored-header-bridge-owner")
        hammer = DeterministicDoctorHammer(bounds=prepared.bounds, authoritative_store=store,
            trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
        proof = hammer.verify_authoritative(prepared.theorem, prepared.lowering,
            solver_pin=prepared.solver_pin, kernel_pin=prepared.kernel_pin,
            current_roots=prepared.theorem.roots,
            eligible_consequence_refs=(prepared.theorem.consequence_ref,))
        assert proof.mutation_capable
        reference = custody.persist_header_proof_key_custody(prepared=prepared, proof=proof, hammer=hammer)
        gate = DoctorProofCacheGate(tmp_path / "authored-existing-cache.duckdb")
        replacement_cache = FormalVerificationCache(gate.db_path, exact_path=True)
        replacement_store = DoctorSealedReceiptStore(store.path, authority_id=store.authority_id)
    values.update(prepared=prepared, hammer=hammer, proof=proof, store=store, custody_ref=reference,
        gate=gate, owner_pin=_path_pin(gate), replacement_cache=replacement_cache,
        replacement_store=replacement_store)
    return values


def _assert_local_receipt(prepared, receipt):
    assert type(receipt) is ScopedProofReceipt
    assert receipt.authoritative_verdict is ProofVerdict.PROVED
    assert receipt.authoritative_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert receipt.obligation_id == prepared.theorem.theorem_id
    assert receipt.repository_id == prepared.theorem.roots.repository_id
    assert receipt.repository_tree_id == prepared.key.tree.logical_id
    assert receipt.premise_ids == prepared.theorem.premise_ids
    assert receipt.premise_ids == tuple(item.logical_id for item in prepared.key.premises)
    assert receipt.resource_budget == prepared.key.budget
    for receipt_name, key_name in (("solver_id", "solver"), ("kernel_id", "kernel"),
            ("translator_id", "translation"), ("toolchain_id", "toolchain"),
            ("theorem_registry_id", "registry"), ("policy_id", "policy")):
        assert getattr(receipt, receipt_name) == getattr(prepared.key, key_name).logical_id
    scope = receipt.scope
    assert scope.obligation_id == prepared.theorem.theorem_id
    assert scope.theorem_cid == prepared.theorem.content_id
    assert scope.property_id == prepared.theorem.property_id
    assert scope.consequence_ref == prepared.theorem.consequence_ref
    assert scope.premise_ids == prepared.theorem.premise_ids
    assert scope.assumption_ids == prepared.theorem.assumption_ids
    assert scope.proof_scope == factory.PROOF_SCOPE
    if "security_ir_declaration_cid" in prepared.contract:
        assert scope.residual_obligation_ids
    else:
        assert scope.residual_obligation_ids == ()
    for name in ("whole_program_proved", "completion_authority", "mutation_authority", "execution_authority"):
        assert getattr(scope, name) is False
    decoded = ProofReceipt.from_dict(receipt.to_record())
    assert type(decoded) is ScopedProofReceipt and decoded.to_dict() == receipt.to_dict()
    for current in (receipt, decoded):
        assert current.is_kernel_verified
        assert current.satisfies_completion() is False
        completion = current.completion_reference()
        assert completion["assurance_satisfied"] is False
        assert "conditional_local_proof_cannot_complete_task" in completion["reason_codes"]


def test_native_custody_replay_admits_only_local_receipt_and_native_cache_preserves_scope(authored, monkeypatch):
    _forbid_owners(monkeypatch)
    before = len(authored["calls"])
    admission = _import(authored)
    assert authored["calls"][before:] == [DoctorExecutableRole.KERNEL]
    assert admission.accepted and admission.stored
    assert admission.key == authored["prepared"].key
    _assert_local_receipt(authored["prepared"], admission.receipt)
    for name in ("completion_authority", "execution_authority", "mutation_authority", "whole_program_proved"):
        assert getattr(admission, name) is False
    for method, stage in ((admission.revalidate_for_render, DoctorCacheStage.RENDER),
                          (admission.revalidate_for_commit, DoctorCacheStage.COMMIT)):
        result = method()
        assert type(result) is DoctorCacheLookupResult and result.hit
        assert result.audit.stage is stage
        assert result.receipt.content_id == admission.receipt.content_id
        _assert_local_receipt(authored["prepared"], result.receipt)


def test_repeat_native_replay_keeps_original_receipt_identity_without_quarantining(authored, monkeypatch):
    _forbid_owners(monkeypatch)
    first = _import(authored)
    second = _import(authored)
    assert first.receipt.content_id == second.receipt.content_id
    assert first.receipt.to_dict() == second.receipt.to_dict()
    assert second.revalidate_for_render().hit
    assert second.revalidate_for_commit().hit
    assert not authored["gate"].is_quarantined(first.key)
    assert not authored["gate"].is_tombstoned(first.key)


def test_independently_compiled_security_ir_residuals_remain_open_in_local_cached_proof(authored, monkeypatch):
    analysis = analyze()
    analyses = [{"path": "answer.py", "status": analysis.status,
        "reason_codes": list(analysis.reason_codes), "source_sha256": analysis.source_sha256,
        "contracts": list(analysis.contracts), "open_frontiers": list(analysis.open_frontiers)}]
    compiled = factory.security_ir_bridge.compile_header_security_ir(scoped=authored["scoped"], analyses=analyses)
    contract = {**authored["contract"],
        "security_ir_declaration_cid": compiled.report["declaration_cid"],
        "security_ir_formalization_cid": compiled.report["artifact_cid"]}
    prepared = _prepare(authored, contract=contract)
    hammer = DeterministicDoctorHammer(bounds=prepared.bounds, authoritative_store=authored["store"],
        trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
    with public_replay_without_metadata():
        proof = hammer.verify_authoritative(prepared.theorem, prepared.lowering,
            solver_pin=prepared.solver_pin, kernel_pin=prepared.kernel_pin,
            current_roots=prepared.theorem.roots,
            eligible_consequence_refs=(prepared.theorem.consequence_ref,))
        reference = custody.persist_header_proof_key_custody(prepared=prepared, proof=proof, hammer=hammer)
    _forbid_owners(monkeypatch)
    admission = _import(authored, prepared=prepared, hammer=hammer, custody_ref=reference,
                        expected_key_id=prepared.key.key_id)
    expected = tuple(sorted(item.obligation_id for item in compiled.artifact.proof_obligations))
    assert admission.receipt.scope.residual_obligation_ids == expected
    assert expected
    _assert_local_receipt(prepared, admission.revalidate_for_render().receipt)


@pytest.mark.parametrize("wrong", ["key", "owner-pin", "custody-seal"])
def test_independent_admission_pin_errors_refuse_before_kernel_or_cache_put(authored, monkeypatch, wrong):
    changes = {}
    if wrong == "key":
        changes["expected_key_id"] = "doctor-proof-cache-key:sha256:" + "0" * 64
    elif wrong == "owner-pin":
        changes["expected_owner_path_sha256"] = "0" * 64
    else:
        changes["custody_ref"] = replace(authored["custody_ref"], seal="hmac-sha256:" + "0" * 64)
    before = list(authored["calls"])
    def forbidden(*args, **kwargs):
        pytest.fail("invalid host bindings must refuse before putting a generic receipt")
    monkeypatch.setattr(authored["gate"], "put", forbidden)
    _forbid_owners(monkeypatch)
    with pytest.raises(ValueError):
        _import(authored, **changes)
    assert authored["calls"] == before


@pytest.mark.parametrize("drift", ["source", "executable", "profile", "pins", "bounds",
                                   "seal-key", "native-inode", "native-object"])
def test_render_revalidation_refuses_changed_native_owner_or_recipe(authored, monkeypatch, drift):
    _forbid_owners(monkeypatch)
    admission = _import(authored)
    if drift == "source":
        (authored["repository"] / "answer.py").write_text(PROGRAM + "\n# changed\n")
    elif drift == "executable":
        authored["lean"].write_text("#!/bin/sh\n# changed bytes\nexit 98\n")
    elif drift == "profile":
        object.__setattr__(authored["prepared"].proof_key_profiles, "policy",
                           authored["profiles"].environment)
    elif drift == "pins":
        authored["hammer"]._trusted_executable_pins = {}
    elif drift == "bounds":
        authored["hammer"]._bounds = replace(authored["prepared"].bounds, wall_time_ms=60_001)
    elif drift == "seal-key":
        authored["store"]._key_path.write_bytes(b"x" * 32)
    elif drift == "native-inode":
        path = authored["store"].path
        original = path.with_name("fixture-original-native.sqlite")
        path.rename(original)
        path.write_bytes(original.read_bytes())
    else:
        authored["hammer"]._authoritative_store = authored["replacement_store"]
    with pytest.raises(ValueError):
        admission.revalidate_for_render()


@pytest.mark.parametrize("drift", ["cache-object", "state-cache-object", "cache-inode"])
def test_commit_revalidation_refuses_replaced_generic_owner(authored, monkeypatch, drift):
    _forbid_owners(monkeypatch)
    admission = _import(authored)
    if drift == "cache-object":
        authored["gate"]._cache = authored["replacement_cache"]
    elif drift == "state-cache-object":
        authored["gate"]._state._cache = authored["replacement_cache"]
    else:
        path = authored["gate"].db_path
        original = path.with_name("fixture-original-generic.duckdb")
        path.rename(original)
        path.write_bytes(original.read_bytes())
    with pytest.raises(ValueError):
        admission.revalidate_for_commit()


@pytest.mark.parametrize("denial", ["quarantine", "tombstone"])
@pytest.mark.parametrize("when", ["before-import", "after-import"])
def test_native_generic_denial_state_never_returns_accepted_local_admission(authored, monkeypatch, denial, when):
    _forbid_owners(monkeypatch)
    admission = _import(authored) if when == "after-import" else None
    if denial == "quarantine":
        authored["gate"].quarantine(authored["prepared"].key)
    else:
        authored["gate"].invalidate_semantic_root(root_field="tree",
            root_cid=authored["prepared"].key.tree.cid)
    with pytest.raises(ValueError):
        _import(authored) if admission is None else admission.revalidate_for_render()


@pytest.mark.parametrize("during", ["put-source", "lookup-quarantine"])
def test_drift_during_generic_operations_cannot_escape_as_a_positive_result(authored, monkeypatch, during):
    _forbid_owners(monkeypatch)
    if during == "put-source":
        original = authored["gate"].put
        def drift(*args, **kwargs):
            result = original(*args, **kwargs)
            (authored["repository"] / "answer.py").write_text(PROGRAM + "\n# changed during put\n")
            return result
        monkeypatch.setattr(authored["gate"], "put", drift)
        action = lambda: _import(authored)
    else:
        admission = _import(authored)
        original = authored["gate"].lookup
        def drift(key, **kwargs):
            result = original(key, **kwargs)
            authored["gate"].quarantine(key)
            return result
        monkeypatch.setattr(authored["gate"], "lookup", drift)
        action = admission.revalidate_for_render
    with pytest.raises(ValueError):
        action()


def test_duplicate_custody_row_keeps_same_scoped_cache_receipt_without_quarantine(authored, monkeypatch):
    duplicate = custody.persist_header_proof_key_custody(prepared=authored["prepared"],
        proof=authored["proof"], hammer=authored["hammer"])
    assert duplicate.entry_cid != authored["custody_ref"].entry_cid
    assert duplicate.receipt_cid == authored["custody_ref"].receipt_cid
    _forbid_owners(monkeypatch)
    first = _import(authored)
    second = _import(authored, custody_ref=duplicate)
    assert second.receipt.content_id == first.receipt.content_id
    assert second.receipt.to_dict() == first.receipt.to_dict()
    assert second.revalidate_for_commit().hit
    assert not authored["gate"].is_quarantined(second.key)


def test_public_admission_receipt_is_deeply_immutable_and_detached(authored, monkeypatch):
    _forbid_owners(monkeypatch)
    admission = _import(authored)
    receipt = admission.receipt
    original = receipt.to_dict()
    with pytest.raises(TypeError):
        receipt.metadata["completion_authority"] = True
    with pytest.raises(TypeError):
        receipt.metadata["generic_receipt_implementations"]["bridge_sha256"] = "0" * 64
    with pytest.raises(TypeError):
        receipt.evidence[0].metadata["native_kernel_id"] = "authored-corrupted-kernel"
    with pytest.raises(TypeError):
        receipt.resource_usage["invented"] = 1
    assert type(receipt.metadata["open_frontiers"]) is tuple
    detached = receipt.to_dict()
    detached["metadata"]["completion_authority"] = True
    detached["metadata"]["generic_receipt_implementations"]["bridge_sha256"] = "0" * 64
    detached["evidence"][0]["metadata"]["native_kernel_id"] = "authored-corrupted-kernel"
    assert admission.receipt.to_dict() == original
    assert admission.revalidate_for_render().receipt.to_dict() == original


@pytest.mark.parametrize("implementation", ["emitter", "codec"])
def test_receipt_implementation_byte_drift_refuses_using_only_temporary_source_copies(authored, monkeypatch, implementation):
    module = bridge if implementation == "emitter" else bridge.receipt_contracts
    copied = authored["tmp_path"] / ("copied-" + implementation + ".py")
    copied.write_bytes(Path(module.__file__).read_bytes())
    initial_digest = hashlib.sha256(copied.read_bytes()).hexdigest()
    monkeypatch.setattr(module, "__file__", str(copied))
    _forbid_owners(monkeypatch)
    admission = _import(authored)
    field = "bridge_sha256" if implementation == "emitter" else "receipt_contracts_sha256"
    assert admission.receipt.metadata["generic_receipt_implementations"][field] == initial_digest
    copied.write_bytes(copied.read_bytes() + b"\n# authored implementation-byte drift\n")
    with pytest.raises(bridge.HeaderProofCacheBridgeError):
        admission.revalidate_for_render()


@pytest.mark.parametrize("drift", ["source", "executable", "quarantine"])
def test_final_tombstone_consultation_drift_cannot_return_a_current_positive(authored, monkeypatch, drift):
    _forbid_owners(monkeypatch)
    admission = _import(authored)
    gate = authored["gate"]
    original_lookup, original_tombstone = gate.lookup, gate.is_tombstoned
    state = {"positive_returned": False, "drifted": False}
    source = authored["repository"] / "answer.py"
    original_source = source.read_bytes()
    before = list(authored["calls"])

    def observe_lookup(key, **kwargs):
        result = original_lookup(key, **kwargs)
        assert result.hit
        state["positive_returned"] = True
        return result

    def change_after_tombstone_read(key):
        result = original_tombstone(key)
        if state["positive_returned"] and not state["drifted"]:
            assert result is False
            state["drifted"] = True
            if drift == "source":
                source.write_bytes(original_source + b"\n# changed during final denial consultation\n")
            elif drift == "executable":
                authored["lean"].write_text("#!/bin/sh\n# changed during final denial consultation\nexit 98\n")
            else:
                gate.quarantine(key)
        return result

    monkeypatch.setattr(gate, "lookup", observe_lookup)
    monkeypatch.setattr(gate, "is_tombstoned", change_after_tombstone_read)
    with pytest.raises(bridge.HeaderProofCacheBridgeError):
        admission.revalidate_for_render()
    assert state == {"positive_returned": True, "drifted": True}
    assert authored["calls"] == before
    if drift == "quarantine":
        assert source.read_bytes() == original_source
        assert gate.is_quarantined(admission.key)


def test_real_lean_z3_proof_import_render_and_explicit_restart_preserve_local_scope(scenario, tmp_path, monkeypatch):
    inputs, scoped = _scoped(scenario, tmp_path)
    inputs = _provers(inputs)
    candidate = analyze().candidate
    values = {"scoped": scoped, "contract": candidate.contract,
        "candidate_cid": content_identity(candidate.to_dict()), "profiles": _profiles(),
        "lean": inputs["kernel_executable"], "z3": inputs["solver_executable"]}
    report, proof = factory.prove_header_contract(**{
        name: values[name] for name in ("scoped", "contract", "candidate_cid", "lean", "z3")},
        proof_key_profiles=values["profiles"], state=tmp_path / "actual-proof")
    assert proof is not None and proof.mutation_capable, report
    prepared = _prepare(values)
    reference = DoctorSealedReceiptRef.from_dict(report["native_key_custody"]["ref"])
    gate = DoctorProofCacheGate(tmp_path / "actual-existing-cache.duckdb")
    # All reconnection is explicit fixture-host work, before constructor bans.
    reopened = DoctorSealedReceiptStore(proof.authority_store.path,
                                        authority_id=proof.authority_store.authority_id)
    first_hammer = DeterministicDoctorHammer(bounds=prepared.bounds,
        authoritative_store=proof.authority_store,
        trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
    second_hammer = DeterministicDoctorHammer(bounds=prepared.bounds, authoritative_store=reopened,
        trusted_executable_pins=(prepared.solver_pin, prepared.kernel_pin))
    reopened_gate = DoctorProofCacheGate(cache=FormalVerificationCache(gate.db_path, exact_path=True))
    restarted_preparation = _prepare(values)
    values.update(prepared=prepared, hammer=first_hammer, custody_ref=reference,
                  gate=gate, owner_pin=_path_pin(gate))
    # Delegate to the real execution method; no authored prover output is used.
    actual_run_pinned = DeterministicDoctorHammer._run_pinned
    calls = []

    def observe_actual_run(self, pin, request):
        calls.append(pin.role)
        return actual_run_pinned(self, pin, request)

    monkeypatch.setattr(DeterministicDoctorHammer, "_run_pinned", observe_actual_run)
    _forbid_owners(monkeypatch)
    first = _import(values)
    assert calls == [DoctorExecutableRole.KERNEL]
    assert first.revalidate_for_render().hit
    assert calls == [DoctorExecutableRole.KERNEL]
    second = _import(values, prepared=restarted_preparation, hammer=second_hammer,
                     gate=reopened_gate, expected_owner_path_sha256=_path_pin(reopened_gate))
    assert calls == [DoctorExecutableRole.KERNEL, DoctorExecutableRole.KERNEL]
    assert second.receipt.content_id == first.receipt.content_id
    assert second.revalidate_for_commit().hit
    assert calls == [DoctorExecutableRole.KERNEL, DoctorExecutableRole.KERNEL]
    _assert_local_receipt(restarted_preparation, second.receipt)
