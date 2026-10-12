"""Pure source-bound header key preparation grants no accepted proof.

Executable files and external profiles are authored fixture inputs. They are
hashed and joined to real signed source/AST replay, and are never executed.
"""

from dataclasses import replace
from copy import deepcopy
import hashlib
import json

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_doctor_contract_proof import _scoped
from test.api.test_doctor_header_contracts import PROGRAM, analyze
from ipfs_accelerate_py.agent_supervisor.analysis.content_identity_bridge import identify_strict_artifact
from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import DeterministicDoctorHammer
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import (
    DoctorIdentityBinding, DoctorProofCacheGate, DoctorProofCacheKey, DoctorSealedReceiptStore,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_cache import FormalVerificationCache
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
    AssuranceLevel, canonical_json, content_identity,
)
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_proof as proof_module
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_key_custody as custody
from ipfs_accelerate_py.agent_supervisor.runtime.supervisor_meta_index import SupervisorMetaIndex


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


def _identity(name, body):
    return DoctorIdentityBinding.from_identity(identify_strict_artifact(body),
                                               logical_id="authored-header-key:" + name)


def _profiles():
    bodies = {
        "corpus": {"schema": "authored-header-corpus@1", "inputs": "signed-public-task-scope"},
        "registry": {"schema": "authored-header-registry@1", "operators": [proof_module.OPERATOR]},
        "policy": {"schema": "authored-header-policy@1", "native_kernel_required": True,
                   "network_allowed": False},
        "sandbox": {"schema": "authored-header-sandbox@1", "isolation_required": True,
                    "source_writes_allowed": False},
        "environment": {"schema": "authored-header-environment@1", "platform": "linux-cpu",
                        "locale": "C"},
    }
    return custody.HeaderContractProofKeyProfiles(**{
        name: _identity(name, body) for name, body in bodies.items()})


@pytest.fixture
def authored(scenario, tmp_path, monkeypatch):
    inputs, scoped = _scoped(scenario, tmp_path)
    executables = {}
    for name in ("lean", "z3"):
        executable = tmp_path / ("authored-" + name)
        executable.write_text("#!/bin/sh\n# authored inert " + name + " identity\nexit 99\n")
        executable.chmod(0o755)
        executables[name] = executable
    candidate = analyze().candidate
    def forbidden(*args, **kwargs):
        pytest.fail("pure preparation must not create proof storage or dispatch a prover")
    for cls in (DoctorProofCacheGate, FormalVerificationCache, DoctorSealedReceiptStore, SupervisorMetaIndex):
        monkeypatch.setattr(cls, "__init__", forbidden)
    for name in ("_run_pinned", "verify_authoritative", "reverify_authoritative"):
        monkeypatch.setattr(DeterministicDoctorHammer, name, forbidden)
    return {"inputs": inputs, "scoped": scoped, "contract": candidate.contract,
            "candidate_cid": content_identity(candidate.to_dict()), "profiles": _profiles(),
            "executables": executables, "scenario": scenario, "tmp_path": tmp_path}


def _prepare(authored, **changes):
    values = {"scoped": authored["scoped"], "contract": authored["contract"],
              "candidate_cid": authored["candidate_cid"], **authored["executables"],
              "proof_key_profiles": authored["profiles"]}
    values.update(changes)
    return proof_module.prepare_header_contract_proof(**values)


def test_complete_key_reuses_actual_native_scope_pins_bounds_and_explicit_host_profiles(authored):
    prepared = _prepare(authored)
    key, roots = prepared.key, authored["scoped"].snapshot.roots
    assert type(key) is DoctorProofCacheKey
    assert DoctorProofCacheKey.from_dict(key.to_dict()) == key
    for identity in key.identities:
        assert DoctorIdentityBinding.from_dict(identity.to_dict()) == identity
    for name in ("forest", "tree", "overlay", "candidate_tree"):
        assert getattr(key, name).logical_id == getattr(roots, "tree_id" if name == "candidate_tree" else name + "_id")
    assert key.ast.logical_id == roots.ast_root_id
    assert key.graph.logical_id == roots.graph_id
    assert key.goal.logical_id == prepared.theorem.theorem_id
    assert key.goal.canonical_bytes == canonical_json(prepared.theorem.to_dict()).encode()
    assert len(key.premises) == 1
    assert key.premises[0].logical_id == content_identity(prepared.contract)
    projected = json.loads(key.premises[0].canonical_bytes)
    assert projected["schema"] == "reviewed-header-source-premise@1"
    assert projected["field_correspondence"] == "witnesses->ast_source_observations@1"
    assert projected["original_contract_cid"] == content_identity(prepared.contract)
    assert projected["contract_fields"] == {
        name: value for name, value in prepared.contract.items() if name != "witnesses"}
    assert projected["ast_source_observations"] == prepared.contract["witnesses"]
    restored = custody.restore_header_source_premise(projected, expected_contract=prepared.contract)
    assert canonical_json(restored) == canonical_json(prepared.contract)
    assert key.premises[0].cid == identify_strict_artifact(projected).cid
    assert key.premises[0].cid != content_identity(prepared.contract)
    # Native confidentiality rules remain intact. This reviewed public source
    # schema preserves observations under its own identity and exact inverse.
    assert _identity("original-contract", prepared.contract).contains_private_material is True
    scope_body = json.loads(key.tree.canonical_bytes)
    assert scope_body["task_cid"] == authored["scoped"].task_cid
    assert scope_body["source_tree_id"] == authored["scoped"].report["source_tree_id"]
    for path, source in authored["scoped"].sources.items():
        digest = hashlib.sha256(source).hexdigest()
        assert scope_body["sources"][path]["sha256"] == digest
        assert digest in key.ast.canonical_bytes.decode()
    for name in ("corpus", "registry", "policy", "sandbox", "environment"):
        assert getattr(key, name) == getattr(authored["profiles"], name)
    assert key.budget == prepared.bounds.to_resource_budget()
    assert key.budget.network_allowed is False
    assert key.required_assurance is AssuranceLevel.KERNEL_VERIFIED
    assert prepared.theorem.roots.corpus_id == key.corpus.logical_id
    assert prepared.theorem.roots.policy_id == key.policy.logical_id
    assert prepared.theorem.roots.environment_id == key.environment.logical_id
    canonical = canonical_json(key.to_dict())
    assert "PRIVATE" not in canonical
    assert key.contains_private_material is False
    # Both the actual external executable bytes and the immutable native pin
    # records must participate in the producer's retained canonical bodies.
    toolchain_body = prepared.key.toolchain.canonical_bytes.decode()
    for path in authored["executables"].values():
        assert hashlib.sha256(path.read_bytes()).hexdigest() in toolchain_body
    for pin, binding in ((prepared.solver_pin, key.solver), (prepared.kernel_pin, key.kernel)):
        body = binding.canonical_bytes.decode()
        assert pin.command_id in body
        assert pin.executable_sha256 in body
    translator_body = key.translation.canonical_bytes.decode()
    for digest in prepared.implementation_bindings.values():
        assert digest in translator_body
    assert prepared.contract == authored["contract"]
    assert prepared.candidate_cid == authored["candidate_cid"]
    prepared.assert_current()


def test_absent_external_profiles_leave_key_unavailable_without_inventing_roots(authored):
    prepared = _prepare(authored, proof_key_profiles=None)
    assert prepared is not None
    assert prepared.key is None
    assert prepared.theorem.roots.corpus_id == authored["scoped"].snapshot.roots.corpus_id
    assert prepared.theorem.roots.policy_id == authored["scoped"].snapshot.roots.policy_id
    prepared.assert_current()


def test_unavailable_provers_do_not_create_an_owner_or_execute_anything(authored):
    unavailable = authored["tmp_path"] / "unavailable-prover"
    assert _prepare(authored, lean=unavailable, z3=unavailable) is None
    assert not unavailable.exists()
    assert not list(authored["tmp_path"].glob("**/proof.sqlite"))


@pytest.mark.parametrize("invalid", ["missing", "foreign", "private", "oversized"])
@pytest.mark.parametrize("provers_available", [True, False])
def test_malformed_external_profiles_refuse_before_state_or_prover_dispatch(authored, invalid, provers_available):
    profiles = authored["profiles"]
    with pytest.raises((TypeError, ValueError)):
        if invalid == "missing":
            profiles = replace(profiles, corpus=None)
        elif invalid == "foreign":
            profiles = {"corpus": profiles.corpus}
        elif invalid == "private":
            profiles = replace(profiles, policy=_identity("private-policy", {
                "schema": "authored-header-policy@1", "password": "authored_dummy"}))
        else:
            profiles = replace(profiles, corpus=_identity("oversized-corpus", {
                "schema": "authored-header-corpus@1", "text": "x" * (300 * 1024)}))
        changes = {} if provers_available else {
            "lean": authored["tmp_path"] / "unavailable-lean",
            "z3": authored["tmp_path"] / "unavailable-z3"}
        _prepare(authored, proof_key_profiles=profiles, **changes)
    assert not list(authored["tmp_path"].glob("**/proof.sqlite"))


@pytest.mark.parametrize("drift", ["source", "executable", "implementation", "profile", "review"])
def test_current_replay_rejects_changed_native_preparation_inputs(authored, monkeypatch, drift):
    prepared = _prepare(authored)
    if drift == "source":
        (authored["scenario"]["repository"] / "answer.py").write_text(PROGRAM + "\n# changed\n")
    elif drift == "executable":
        authored["executables"]["lean"].write_text("#!/bin/sh\n# changed authored bytes\nexit 98\n")
    elif drift == "implementation":
        bindings = proof_module._implementation_bindings()
        bindings["header_ast_translator_sha256"] = hashlib.sha256(b"changed reviewed translator").hexdigest()
        monkeypatch.setattr(proof_module, "_implementation_bindings", lambda: bindings)
    elif drift == "profile":
        # A caller can circumvent frozen Python fields; replay still compares
        # the resulting complete key with the original captured native inputs.
        object.__setattr__(prepared.proof_key_profiles, "environment",
            _identity("changed-environment", {"schema": "authored-header-environment@2", "locale": "C"}))
    else:
        prepared = replace(prepared, theorem=replace(prepared.theorem,
            review_receipt_id=content_identity({"review": "foreign-reviewed-body"})))
    with pytest.raises(ValueError):
        prepared.assert_current()
    assert not list(authored["tmp_path"].glob("**/proof.sqlite"))


def test_report_is_detached_and_prepared_key_does_not_create_accepted_proof(authored):
    prepared = _prepare(authored)
    report = prepared.report
    original = json.loads(json.dumps(report))
    report["proof_receipt_id"] = "caller-asserted-proof"
    report["status"] = "proved_local_contract"
    assert prepared.report == original
    assert prepared.report.get("proof_receipt_id") is None
    assert prepared.report["provider_calls"] == 0
    assert prepared.report["whole_program_proved"] is False
    assert prepared.report["completion_authority"] is False
    captured = prepared.custody_payload()
    for name in ("proof_authority", "execution_authority", "generic_cache_admission_supported",
                 "resource_enforcement_verified"):
        assert captured[name] is False
    assert prepared.key is not None
    assert not list(authored["tmp_path"].glob("**/proof.sqlite"))
    prepared.assert_current()


@pytest.mark.parametrize("corruption", ["original-cid", "source-row", "extra-field",
                                        "correspondence", "private-field"])
def test_public_source_projection_inverse_refuses_corruption_or_private_extensions(authored, corruption):
    prepared = _prepare(authored)
    projected = json.loads(prepared.key.premises[0].canonical_bytes)
    expected = deepcopy(authored["contract"])
    if corruption == "original-cid":
        projected["original_contract_cid"] = content_identity({"foreign": "contract"})
    elif corruption == "source-row":
        projected["ast_source_observations"][0]["property_line"] += 1
    elif corruption == "extra-field":
        projected["unreviewed_extension"] = True
    elif corruption == "correspondence":
        projected["field_correspondence"] = "arbitrary-fields->hidden-body@1"
    else:
        expected["password"] = "authored_dummy"
        projected["contract_fields"]["password"] = expected["password"]
        projected["original_contract_cid"] = content_identity(expected)
    with pytest.raises(ValueError):
        custody.restore_header_source_premise(projected, expected_contract=expected)


def test_source_projection_cannot_accept_caller_forged_ast_observations(authored):
    foreign = deepcopy(authored["contract"])
    foreign["witnesses"][0]["callback_parameter"] = "caller_forged_callback"
    with pytest.raises(ValueError):
        custody.project_header_source_premise(scoped=authored["scoped"], contract=foreign,
                                             candidate_cid=authored["candidate_cid"])
