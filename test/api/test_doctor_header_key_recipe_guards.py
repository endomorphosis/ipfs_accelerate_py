"""Native-valid foreign proof recipes fail before any owner or prover opens.

All source, executable and profile inputs are authored fixtures. These tests
exercise the production key producer directly and never execute the leaves.
"""
from dataclasses import replace
import json
from pathlib import Path
import sys

import pytest

from test.api.test_agent_supervisor_local_planning_admission import scenario
from test.api.test_doctor_contract_key_custody import authored, _identity, _prepare
from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import DoctorExactLoweringReceipt
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.runtime import doctor_contract_key_custody as custody


@pytest.fixture(autouse=True)
def isolated_orchestration(tmp_path, monkeypatch):
    monkeypatch.setenv("IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR", str(tmp_path / "orchestration"))


@pytest.fixture
def prepared(authored):
    result = _prepare(authored)
    assert result is not None and result.key is not None
    return result


def _build(prepared, **changes):
    values = {"scoped": prepared.scoped_analysis, "contract": prepared.contract,
        "candidate_cid": prepared.candidate_cid, "theorem": prepared.theorem,
        "lowering": prepared.lowering, "solver_pin": prepared.solver_pin,
        "kernel_pin": prepared.kernel_pin, "bounds": prepared.bounds,
        "translator_preimage": prepared.translator_preimage,
        "toolchain_preimage": prepared.toolchain_preimage,
        "profiles": prepared.proof_key_profiles}
    values.update(changes)
    return custody.build_header_contract_proof_key(**values)


def _foreign_pin(pin, mutation):
    if mutation == "environment":
        return replace(pin, environment_id="authored:foreign-environment")
    if mutation == "toolchain":
        return replace(pin, toolchain_id=content_identity({"toolchain": "foreign"}))
    argv = list(pin.argv)
    if mutation == "adapter":
        argv[2] += "\n# foreign adapter bytes\n"
    else:
        cfg = json.loads(argv[4])
        if mutation == "source-sha":
            cfg["source_sha256"] = "0" * 64
        elif mutation == "smt":
            cfg["smt"] = "(check-sat)\n"
        elif mutation == "axioms":
            cfg["expected_axioms"] = []
        else:
            assert mutation == "leaf-sha"
            cfg["lean_sha256"] = "0" * 64
        argv[4] = json.dumps(cfg)
    return replace(pin, argv=tuple(argv))


@pytest.mark.parametrize("mutation", [
    "adapter", "source-sha", "smt", "axioms", "leaf-sha", "environment", "toolchain",
])
def test_native_valid_alternate_commands_cannot_claim_the_current_header_recipe(prepared, mutation):
    # Change both pins so agreement between the pins alone cannot validate the
    # foreign recipe. Their native canonical identities remain well formed.
    solver = _foreign_pin(prepared.solver_pin, mutation)
    kernel = _foreign_pin(prepared.kernel_pin, mutation)
    assert solver.command_id != kernel.command_id
    with pytest.raises(ValueError):
        _build(prepared, solver_pin=solver, kernel_pin=kernel)


def test_swapped_native_roles_are_not_repaired_into_a_header_key(prepared):
    with pytest.raises(ValueError):
        _build(prepared, solver_pin=prepared.kernel_pin, kernel_pin=prepared.solver_pin)


@pytest.mark.parametrize("mutation", ["review", "consequence"])
def test_canonical_theorem_and_matching_lowering_still_need_original_source_recipe(prepared, mutation):
    theorem = prepared.theorem
    if mutation == "review":
        theorem = replace(theorem, review_receipt_id=content_identity({"review": "foreign"}))
    else:
        consequence = content_identity({"consequence": "foreign"})
        theorem = replace(theorem, consequence_ref=consequence,
            theorem_body=theorem.theorem_body.replace(theorem.consequence_ref, consequence))
    lowering = DoctorExactLoweringReceipt.create(theorem,
        logic_ir_statement=theorem.theorem_body, native_statement=theorem.theorem_body)
    with pytest.raises(ValueError, match="reviewed theorem differs"):
        _build(prepared, theorem=theorem, lowering=lowering)


@pytest.mark.parametrize("body", [
    {}, "opaque profile label", {"schema": "unversioned-profile"},
    {"repository_id": "authored:repository", "tree_id": "authored:tree", "prefix": "corpus"},
])
def test_hashed_empty_or_diagnostic_profiles_do_not_supply_concrete_proof_profiles(authored, body):
    # These bindings have valid native CID/preimage identities. A digest and
    # caller label still cannot replace a concrete declared proof profile.
    identity = _identity("foreign-corpus", body)
    with pytest.raises(ValueError):
        replace(authored["profiles"], corpus=identity)


def test_leaf_edit_after_the_single_capture_refuses_before_preparation_returns(authored, monkeypatch):
    leaf = authored["executables"]["lean"]
    original = Path.read_bytes
    captures = []

    def edit_after_capture(path):
        raw = original(path)
        if path == leaf:
            captures.append(raw)
            if len(captures) == 1:
                path.write_bytes(raw + b"\n# authored concurrent leaf edit\n")
        return raw

    monkeypatch.setattr(Path, "read_bytes", edit_after_capture)
    # Without optional key profiles, the preparation factory's own final
    # leaf guard must still catch drift; no key-builder guard can rescue it.
    with pytest.raises(ValueError, match="executable bytes changed"):
        _prepare(authored, proof_key_profiles=None)
    assert len(captures) == 2 and captures[0] != captures[1]


def test_default_python_capture_refuses_a_changed_second_read(authored, monkeypatch):
    executable = Path(sys.executable).resolve()
    original = Path.read_bytes
    captures = []

    def changed_second_read(path):
        raw = original(path)
        if path == executable:
            captures.append(raw)
            if len(captures) == 2:
                return raw + b"\n# authored temporary Python read mismatch\n"
        return raw

    monkeypatch.setattr(Path, "read_bytes", changed_second_read)
    # The installed binary is never changed. Reusing the single captured
    # digest for both roles makes the second read a currentness check. Separate
    # per-role captures followed by a check of only role one would accept it.
    with pytest.raises(ValueError, match="executable bytes changed"):
        _prepare(authored, proof_key_profiles=None)
    assert len(captures) == 2
