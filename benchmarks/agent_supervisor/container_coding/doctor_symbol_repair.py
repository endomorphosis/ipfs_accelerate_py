"""Bounded Doctor mutation probe for diagnostic-symbol contract adaptation.

The candidate is an explicitly authored, exact replacement; this does not claim
that the default Doctor synthesized it. Native Hammer runs Z3 and Lean, then
DoctorWorktreeAdapter applies the candidate in a disposable checkout. Tests and
an exact source CAS are required before publishing that one file locally.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from index_preflight import qualify

TARGET = "ipfs_accelerate_py/agent_supervisor/analysis/doctor_contract_adapters.py"
ANCHOR = "        affected_symbol_refs=(finding.symbol,) if finding.symbol else (),"
REPLACEMENT = """        # Diagnostic symbols may be expressions rather than identifiers. Keep
        # their exact text in the diagnostic overlay, and bind a compact CID
        # at the deterministic contract boundary. Existing compact refs survive.
        affected_symbol_refs=(
            (
                content_identity({"schema": "doctor-symbol-ref@1", "symbol": finding.symbol})
                if any(char.isspace() for char in finding.symbol)
                else finding.symbol
            ),
        ) if finding.symbol else (),"""
LEAN = """def Compact (s : String) : Prop := s.toList.all (fun c => !c.isWhitespace) = true
def selectRef (valid : Bool) (original digest : String) : String :=
  if valid then original else digest
theorem compact_selection (valid : Bool) (original digest : String)
    (original_ok : valid = true → Compact original)
    (digest_ok : Compact digest) : Compact (selectRef valid original digest) := by
  cases valid with
  | false => exact digest_ok
  | true => exact original_ok rfl
theorem preserves_compact (original digest : String) :
    selectRef true original digest = original := by rfl
#print axioms compact_selection
#print axioms preserves_compact
"""
SMT = """(set-logic QF_UF)
(declare-const valid Bool)
(declare-const original_ok Bool)
(declare-const digest_ok Bool)
(assert (=> valid original_ok))
(assert digest_ok)
(assert (not (ite valid original_ok digest_ok)))
(check-sat)
"""
# Embedded in the pinned command, never loaded from an unbound script pathname.
ADAPTER = r"""
import hashlib,json,pathlib,subprocess,sys,tempfile
sys.path.insert(0,sys.argv[1])
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity,canonical_json
cfg=json.loads(sys.argv[2]);role=sys.argv[3]
d=json.load(sys.stdin);t=d['theorem'];l=d['lowering'];source=l['native_statement']
assert hashlib.sha256(source.encode()).hexdigest()==cfg['source_sha256']
assert t['theorem_body']==source
for name in ('lean','z3'):
 assert hashlib.sha256(pathlib.Path(cfg[name]).read_bytes()).hexdigest()==cfg[name+'_sha256']
if role=='solver':
 r=subprocess.run([cfg['z3'],'-in'],input=cfg['smt'],text=True,capture_output=True,timeout=15)
 assert r.returncode==0 and r.stdout.strip()=='unsat',r.stderr
else:
 assert d['native_receipt']['proof_object']==source
with tempfile.TemporaryDirectory(prefix='doctor-lean-') as tmp:
 p=pathlib.Path(tmp)/'Proof.lean';p.write_text(source)
 r=subprocess.run([cfg['lean'],str(p)],text=True,capture_output=True,timeout=30)
 assert r.returncode==0,(r.stdout,r.stderr)
 assert 'sorry' not in r.stdout+r.stderr
 expected=set(cfg.get("expected_axioms", ["'compact_selection' depends on axioms: [propext, Classical.choice, Quot.sound]","'preserves_compact' does not depend on any axioms"]))
 assert set(r.stdout.strip().splitlines())==expected,r.stdout
base={'theorem_cid':content_identity(t),'lowering_cid':content_identity(l),'property_id':t['property_id'],'consequence_ref':t['consequence_ref'],'premise_ids':t['premise_ids']}
if role=='solver':base.update(status='proved',proof_object=source)
else:base.update(status='kernel_verified',native_receipt_cid=content_identity(d['native_receipt']),proof_object_cid=content_identity({'proof_object':source}),kernel_id='kernel:lean4-compact-selection')
sys.stdout.write(canonical_json(base))
"""


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def save(path: Path, value) -> None:
    path.write_text(json.dumps(value, indent=2, default=str) + "\n")


def run(root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=False)
    before = (root / TARGET).read_bytes()
    text = before.decode()
    if text.count(ANCHOR) != 1:
        raise ValueError("exact candidate anchor is absent or ambiguous")
    after = text.replace(ANCHOR, REPLACEMENT).encode()
    ast.parse(after)
    # Verify the edit changes only the identified contract keyword expression.
    trees = [ast.parse(raw) for raw in (before, after)]
    for tree in trees:
        fn = next(
            n
            for n in tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "adapt_diagnostic_finding_to_deterministic"
        )
        fields = [
            n
            for n in ast.walk(fn)
            if isinstance(n, ast.keyword) and n.arg == "affected_symbol_refs"
        ]
        assert len(fields) == 1
        fields[0].value = ast.Constant(value=None)
    if ast.dump(trees[0]) != ast.dump(trees[1]):
        raise ValueError("candidate modifies unrelated syntax")
    (output / "candidate.py").write_bytes(after)
    save(
        output / "proposal.json",
        {
            "author": "assistant",
            "operator": "exact_contract_keyword_replacement",
            "target": TARGET,
            "before_sha256": digest(before),
            "after_sha256": digest(after),
            "unrelated_ast_unchanged": True,
            "autonomous_synthesis": False,
        },
    )
    index = qualify(root, output / "indexes", [TARGET])
    return apply_candidate(
        root,
        output,
        target=TARGET,
        before=before,
        after=after,
        index=index,
        lean_source=LEAN,
        smt=SMT,
        labels=(
            "theorem:compact-selection",
            "property:compact-ref",
            "claim:boundary-selection",
            "consequence:compact-symbol-reference",
        ),
        assumptions=("assumption:content-identity-returns-compact-cid",),
        expected_axioms=[
            "'compact_selection' depends on axioms: [propext, Classical.choice, Quot.sound]",
            "'preserves_compact' does not depend on any axioms",
        ],
        proof_scope="conditional compact-reference selection; not whole-program correctness",
        validation_code='import importlib.util,sys,pytest\nname="ipfs_accelerate_py.agent_supervisor.analysis.doctor_contract_adapters"\nspec=importlib.util.spec_from_file_location(name,sys.argv[1]);module=importlib.util.module_from_spec(spec);sys.modules[name]=module;spec.loader.exec_module(module)\nraise SystemExit(pytest.main(["-o","addopts=","--noconftest","-q","-o","log_cli=false","--tb=short","test/api/test_agent_supervisor_doctor_contract_adapters.py"]))\n',
    )


def apply_candidate(
    root,
    output,
    *,
    target,
    before,
    after,
    index,
    lean_source,
    smt,
    labels,
    assumptions,
    expected_axioms,
    proof_scope,
    validation_code,
):
    """Prove a scoped property, isolate the edit, validate, then publish with CAS."""
    from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
        ProgramLogicAuthorityRoots,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import (
        content_identity,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import (
        DeterministicDoctorHammer,
        DoctorHammerBounds,
        DoctorReviewedTheorem,
        DoctorExactLoweringReceipt,
        DoctorPinnedExecutable,
        DoctorExecutableRole,
    )
    from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import (
        DoctorSealedReceiptStore,
    )
    from ipfs_accelerate_py.agent_supervisor.runtime.doctor_worktree_adapter import (
        DoctorWorktreeAdapter,
        DoctorExactEdit,
    )

    lean = Path(subprocess.check_output(["elan", "which", "lean"], text=True).strip()).resolve()
    z3 = Path(shutil.which("z3")).resolve()
    bindings = {"before": digest(before), "after": digest(after), "snapshot": index["snapshot_id"]}
    cid = content_identity(bindings)
    values = {
        name: cid for name in ProgramLogicAuthorityRoots.__dataclass_fields__ if name != "SCHEMA"
    }
    values.update(
        repository_id=content_identity({"root": str(root)}),
        tree_id="sha256:" + digest(before),
        change_id="sha256:" + digest(after),
        index_id=index["snapshot_id"],
        toolchain_id=content_identity(
            {"lean": digest(lean.read_bytes()), "z3": digest(z3.read_bytes())}
        ),
    )
    roots = ProgramLogicAuthorityRoots(**values)
    anchors = " ".join(labels)
    source = "-- " + anchors + "\n-- candidate sha256:" + digest(after) + "\n" + lean_source
    (output / "Proof.lean").write_text(source)
    (output / "obligation.smt2").write_text(smt)
    theorem = DoctorReviewedTheorem(
        roots=roots,
        theorem_id=labels[0],
        property_id=labels[1],
        claim_id=labels[2],
        consequence_ref=labels[3],
        theorem_body=source,
        body_format="lean4",
        premise_ids=(content_identity({"python_ast_projection": bindings}),),
        assumption_ids=assumptions,
        review_receipt_id=content_identity({"review": proof_scope, "candidate": bindings}),
        translator_id=roots.translator_id,
        toolchain_id=roots.toolchain_id,
        policy_id=roots.policy_id,
    )
    lowering = DoctorExactLoweringReceipt.create(
        theorem, logic_ir_statement=source, native_statement=source
    )
    cfg = {
        "lean": str(lean),
        "z3": str(z3),
        "lean_sha256": digest(lean.read_bytes()),
        "z3_sha256": digest(z3.read_bytes()),
        "source_sha256": digest(source.encode()),
        "smt": smt,
        "expected_axioms": expected_axioms,
    }
    pins = []
    executable = Path(sys.executable).resolve()
    for role in ("solver", "kernel"):
        pins.append(
            DoctorPinnedExecutable(
                role=DoctorExecutableRole(role),
                executable_path=str(executable),
                executable_sha256="sha256:" + digest(executable.read_bytes()),
                argv=("-I", "-c", ADAPTER, str(root), json.dumps(cfg), role),
                verifier_id="verifier:" + role + ":lean-z3",
                toolchain_id=roots.toolchain_id,
                environment_id=roots.environment_id,
            )
        )
    store = DoctorSealedReceiptStore(
        output / "proof.sqlite", authority_id="authority:scoped-doctor-probe"
    )
    hammer = DeterministicDoctorHammer(
        bounds=DoctorHammerBounds(wall_time_ms=60000),
        authoritative_store=store,
        trusted_executable_pins=pins,
    )
    proof = hammer.verify_authoritative(
        theorem,
        lowering,
        solver_pin=pins[0],
        kernel_pin=pins[1],
        current_roots=roots,
        eligible_consequence_refs=(theorem.consequence_ref,),
    )
    save(output / "proof.json", proof.to_dict())
    if not proof.mutation_capable:
        raise RuntimeError("native proof did not verify; no mutation applied")
    hammer.reverify_authoritative(proof, current_roots=roots)
    checkout = output / "checkout"
    (checkout / target).parent.mkdir(parents=True)
    (checkout / target).write_bytes(before)
    for args in [
        ("init", "-q"),
        ("add", "."),
        (
            "-c",
            "user.name=Doctor probe",
            "-c",
            "user.email=doctor@localhost",
            "commit",
            "-qm",
            "Exact scoped snapshot",
        ),
    ]:
        subprocess.run(["git", "-C", str(checkout), *args], check=True)
    adapter = DoctorWorktreeAdapter(
        repository_root=checkout, state_root=output / "doctor-state", permitted_paths=(target,)
    )
    with adapter.prepare(session_id="bounded-contract-repair") as session:
        receipt = session.apply_group(
            (
                DoctorExactEdit(
                    path=target, before_hash="sha256:" + digest(before), after_bytes=after
                ),
            ),
            group_id="symbol-contract",
        )
        save(output / "apply.json", receipt.to_dict())
        candidate_path = session.worktree_root / target
        with (output / "validation.log").open("w") as log:
            tests = subprocess.run(
                [sys.executable, "-c", validation_code, str(candidate_path)],
                cwd=root,
                env={**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONPATH": str(root)},
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if tests.returncode:
            session.restore(reason="validation_failed")
            raise RuntimeError("candidate tests failed; shared source unchanged")
        if (root / target).read_bytes() != before:
            session.restore(reason="source_cas_failed")
            raise RuntimeError("shared source changed; refusing publication")
        # Explicitly authorized local integration, separately recorded from the
        # isolated Doctor receipt. No Git ref or completion state is promoted.
        staged = (root / target).with_suffix(".doctor-tmp")
        try:
            with staged.open("xb") as stream:
                stream.write(candidate_path.read_bytes())
            os.chmod(staged, (root / target).stat().st_mode)
            os.replace(staged, root / target)
        finally:
            if staged.exists():
                staged.unlink()
        save(
            output / "integration.json",
            {
                "target": target,
                "before_sha256": digest(before),
                "after_sha256": digest(after),
                "source_cas_matched": True,
                "tests_exit_code": tests.returncode,
                "git_ref_promoted": False,
            },
        )
        session.restore(reason="qualified_probe_cleanup")
    result = {
        "status": "repaired",
        "target": target,
        "doctor_applied": True,
        "native_hammer": True,
        "provers": ["z3", "lean4"],
        "proof_scope": proof_scope,
        "index_scope": index["scope"],
        "autonomous_synthesis": False,
        "full_system_qualified": False,
    }
    save(output / "result.json", result)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.root.resolve(), args.output.resolve())))
