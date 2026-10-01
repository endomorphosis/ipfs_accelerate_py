"""Real native Doctor composition smoke using an isolated authored fixture.

This is integration qualification, not a coding benchmark or proof of general
repair synthesis. No provider is called. Both actual Lean and Z3 are required.
"""

import ast
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from dataclasses import replace
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DeterministicDoctorFinding,
    DoctorRepairDisposition,
    DoctorOperatorKind,
    DoctorMode,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_dependency_graph import (
    PathSource,
    ProgramDependencyGraph,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_graph import ProgramGraphRoots
from ipfs_accelerate_py.agent_supervisor.runtime.deterministic_doctor_runtime import (
    DeterministicDoctorRuntime,
)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_repair_composition import (
    DoctorCompositionInputs,
    composition_snapshot,
    operator_consequence_ref,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transforms import (
    build_default_doctor_operator_registry,
    make_edit_site,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_synthesis import (
    DoctorSynthesisRequest,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_impact import (
    DoctorImpactRequest,
)
from ipfs_accelerate_py.agent_supervisor.analysis.program_logic_prediction_contracts import (
    ProgramLogicAuthorityRoots,
)
from ipfs_accelerate_py.agent_supervisor.proof.deterministic_doctor_hammer import (
    DeterministicDoctorHammer,
    DoctorHammerBounds,
    DoctorReviewedTheorem,
    DoctorExactLoweringReceipt,
    DoctorPinnedExecutable,
    DoctorExecutableRole,
)
from ipfs_accelerate_py.agent_supervisor.proof.doctor_proof_cache import DoctorSealedReceiptStore
from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
from ipfs_accelerate_py.agent_supervisor.validation.deterministic_doctor_policy import (
    DeterministicDoctorPolicy,
)
from ipfs_accelerate_py.agent_supervisor.runtime.doctor_worktree_adapter import (
    DoctorWorktreeAdapter,
)

REPOSITORY = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from doctor_symbol_repair import ADAPTER


def sha(raw):
    return hashlib.sha256(raw).hexdigest()


def prepare(out):
    """Build actual checkout/graph/prover bindings without invoking a prover."""
    out = Path(out).resolve()
    out.mkdir()
    root = out / "checkout"
    root.mkdir()
    text = "def process(amount: int) -> int:\n    return amount\n\ndef use(value: int) -> int:\n    return process(count=value)\n"
    (root / "pkg.py").write_text(text)
    for args in [
        ("init", "-q"),
        ("config", "user.email", "doctor@example.invalid"),
        ("config", "user.name", "Doctor"),
        ("add", "."),
        ("commit", "-qm", "fixture"),
    ]:
        subprocess.run(["git", "-C", str(root), *args], check=True)
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    ref = "refs/heads/doctor/composition"
    subprocess.run(["git", "-C", str(root), "update-ref", ref, head], check=True)
    runtime = DeterministicDoctorRuntime(
        checkout_root=root,
        index_root=out / "indexes",
        policy=DeterministicDoctorPolicy(enabled=True, default_mode=DoctorMode.SANDBOX_AUTO),
    )
    evidence = runtime.build_evidence()
    observed = evidence.snapshot.roots
    lean = Path(
        os.environ.get("DOCTOR_COMPOSITION_LEAN")
        or subprocess.check_output(
            ["elan", "which", "lean"],
            text=True,
        ).strip()
    ).resolve()
    z3 = Path(os.environ.get("DOCTOR_COMPOSITION_Z3") or shutil.which("z3")).resolve()
    toolchain = content_identity({"lean": sha(lean.read_bytes()), "z3": sha(z3.read_bytes())})
    graph = ProgramDependencyGraph(
        ProgramGraphRoots(
            forest_id=observed.forest_id,
            tree_id=observed.tree_id,
            overlay_id=observed.overlay_id,
            coverage_id=observed.file_root_id,
            included_roots=("pkg.py",),
            toolchain_id=toolchain,
        )
    ).build([PathSource(path="pkg.py", source=text, language="python")])
    (out / "graph.json").write_text(json.dumps(graph.to_dict(), indent=2))
    roots = replace(
        observed,
        graph_id=graph.graph_id,
        toolchain_id=toolchain,
        lease_id=content_identity({"writer_scope": "pkg.py", "base": head}),
        translator_id="translator:closed-keyword-rename@1",
    )
    snapshot = composition_snapshot(evidence.snapshot, roots)
    # The declaration independently constrains the caller's keyword. The
    # candidate is not used as an expectation, and donor code is not executed.
    parsed = ast.parse(text)
    declaration = next(
        node for node in parsed.body if isinstance(node, ast.FunctionDef) and node.name == "process"
    )
    assert [arg.arg for arg in declaration.args.args] == ["amount"]
    expected = content_identity(
        {"declaration": ast.dump(declaration), "source": sha(text.encode())}
    )
    finding = DeterministicDoctorFinding(
        roots=roots,
        finding_id=content_identity({"call_keyword": "count", "source": sha(text.encode())}),
        snapshot_id=snapshot.snapshot_id,
        disposition=DoctorRepairDisposition.SUPPORTED,
        observed_fact_refs=(content_identity({"keyword": "count", "source": sha(text.encode())}),),
        expected_behavior_refs=(expected,),
        affected_symbol_refs=("process",),
        consumer_refs=("consumer:use",),
        invalidation_refs=(roots.tree_id,),
    )
    registry = build_default_doctor_operator_registry(roots)
    proposal = registry.propose(
        DoctorOperatorKind.EXACT_RENAME,
        make_edit_site("pkg.py", "count", start=text.index("count=")),
        obligation_refs=(finding.content_id,),
        parameter_name="amount",
        previous_parameter_name="count",
    )
    synthesis = DoctorSynthesisRequest(
        roots=roots,
        proposal=proposal,
        span_text="count",
        file_text=text,
        value_ref=expected,
        placement_ref=proposal.edit_site.content_id,
    )
    expected_after_hash = "sha256:" + sha(b"amount")
    consequence = operator_consequence_ref(synthesis, expected_after_hash, (expected,))
    proots = ProgramLogicAuthorityRoots(
        **{
            key: getattr(roots, key)
            if hasattr(roots, key)
            else content_identity({"key": key, "finding": finding.content_id})
            for key in ProgramLogicAuthorityRoots.__dataclass_fields__
            if key != "SCHEMA"
        }
    )
    labels = (
        "theorem:keyword-renaming",
        "property:keyword-value-preservation",
        "claim:exact-keyword-binding",
        consequence,
    )
    lean_source = (
        "-- "
        + " ".join(labels)
        + '\ndef renameKeyword (value : Int) : String × Int := ("amount", value)\ntheorem selected_name (value : Int) : (renameKeyword value).1 = "amount" := by rfl\ntheorem preserved_value (value : Int) : (renameKeyword value).2 = value := by rfl\n#print axioms selected_name\n#print axioms preserved_value\n'
    )
    theorem = DoctorReviewedTheorem(
        roots=proots,
        theorem_id=labels[0],
        property_id=labels[1],
        claim_id=labels[2],
        consequence_ref=consequence,
        theorem_body=lean_source,
        body_format="lean4",
        premise_ids=(expected,),
        assumption_ids=("assumption:ast-identifier-projection",),
        review_receipt_id=content_identity(
            {
                "review": "closed keyword pair update; exact AST placement separately checked",
                "consequence": consequence,
            }
        ),
        translator_id=roots.translator_id,
        toolchain_id=roots.toolchain_id,
        policy_id=roots.policy_id,
    )
    lowering = DoctorExactLoweringReceipt.create(
        theorem, logic_ir_statement=lean_source, native_statement=lean_source
    )
    cfg = {
        "lean": str(lean),
        "z3": str(z3),
        "lean_sha256": sha(lean.read_bytes()),
        "z3_sha256": sha(z3.read_bytes()),
        "source_sha256": sha(lean_source.encode()),
        "smt": '(set-logic QF_SLIA)\n(declare-const value Int)\n(assert (or (not (= "amount" "amount")) (not (= value value))))\n(check-sat)\n',
        "expected_axioms": [
            "'selected_name' does not depend on any axioms",
            "'preserved_value' does not depend on any axioms",
        ],
    }
    executable = Path(sys.executable).resolve()
    pins = tuple(
        DoctorPinnedExecutable(
            role=DoctorExecutableRole(role),
            executable_path=str(executable),
            executable_sha256="sha256:" + sha(executable.read_bytes()),
            argv=("-I", "-c", ADAPTER, str(REPOSITORY), json.dumps(cfg), role),
            verifier_id="verifier:" + role + ":lean-z3",
            toolchain_id=roots.toolchain_id,
            environment_id=roots.environment_id,
        )
        for role in ("solver", "kernel")
    )
    store = DoctorSealedReceiptStore(
        out / "proof.sqlite", authority_id="authority:closed-keyword-composition"
    )
    hammer = DeterministicDoctorHammer(
        bounds=DoctorHammerBounds(wall_time_ms=60000),
        authoritative_store=store,
        trusted_executable_pins=pins,
    )
    impact = DoctorImpactRequest(
        roots=roots,
        subject_symbol_id="process",
        change_set_id=consequence,
        before_contract_ref=finding.observed_fact_refs[0],
        after_contract_ref=expected,
        evidence_refs=(finding.content_id,),
    )
    adapter = DoctorWorktreeAdapter(
        repository_root=root,
        state_root=out / "worktrees",
        permitted_paths=("pkg.py",),
        permitted_refs=(ref,),
    )
    inputs = DoctorCompositionInputs(
        evidence_id=evidence.evidence_id,
        snapshot=snapshot,
        finding=finding,
        synthesis=synthesis,
        expected_after_hash=expected_after_hash,
        theorem=theorem,
        lowering=lowering,
        solver_pin=pins[0],
        kernel_pin=pins[1],
        hammer=hammer,
        impact=impact,
        program_graph=graph,
        source_hashes={"pkg.py": sha(text.encode())},
        proof_scope="The selected keyword becomes amount and retains its integer value. No whole-program theorem.",
        worktree_adapter=adapter,
        target_ref=ref,
        base_ref=head,
    )
    runtime.bind_composition(inputs)
    return runtime, inputs


def run(out):
    runtime, inputs = prepare(out)
    return run_prepared(runtime, inputs, out)


def run_prepared(runtime, inputs, out):
    """Exercise an already prepared native binding, retaining every live gate."""
    out = Path(out).resolve()
    # Behavioral validation is separate from the scoped formal theorem. The
    # authored fixture runs in a subprocess, outside deterministic planning.
    before = subprocess.run(
        [sys.executable, "-I", "-c", inputs.synthesis.file_text + "\nassert use(7) == 7\n"],
        capture_output=True,
        text=True,
    )
    if before.returncode == 0 or "unexpected keyword argument 'count'" not in before.stderr:
        raise RuntimeError("fixture did not exhibit the expected keyword defect")
    report = runtime.plan()
    (out / "plan.json").write_text(json.dumps(report.to_dict(), indent=2))
    composed = runtime.composition_result
    for name in ("tactician", "proof", "synthesis", "impact", "compilation"):
        receipt = getattr(composed, name)
        if receipt is not None:
            (out / (name + ".json")).write_text(json.dumps(receipt.to_dict(), indent=2))
    if composed.plan is not None:
        (out / "compiled-plan.json").write_text(json.dumps(composed.plan.to_dict(), indent=2))
    if composed.admitted:
        clean = subprocess.check_output(
            ["git", "-C", str(runtime.checkout_root), "status", "--porcelain"],
            text=True,
        )
        if clean:
            raise RuntimeError("qualification checkout is not clean")
        report = runtime.execute(
            "repair", mode=DoctorMode.SANDBOX_AUTO, plan=composed.plan, exact_clean_target=True
        )
        (out / "repair.json").write_text(json.dumps(report.to_dict(), indent=2))
        if not report.result.changed:
            raise RuntimeError(report.result.explanation)
        transaction = runtime.composition_transaction
        (out / "transaction.json").write_text(json.dumps(transaction.to_dict(), indent=2))
        committed = subprocess.check_output(
            ["git", "-C", str(runtime.checkout_root), "rev-parse", inputs.target_ref],
            text=True,
        ).strip()
        if (
            transaction.merge_cas is None
            or committed != transaction.merge_cas.desired_ref
            or transaction.merge_cas.expected_ref != inputs.base_ref
            or committed == inputs.base_ref
        ):
            raise RuntimeError("durable target ref does not match the native CAS receipt")
        candidate = subprocess.check_output(
            ["git", "-C", str(runtime.checkout_root), "show", inputs.target_ref + ":pkg.py"],
            text=True,
        )
        after = subprocess.run(
            [sys.executable, "-I", "-c", candidate + "\nassert use(7) == 7\n"],
            capture_output=True,
            text=True,
        )
        if after.returncode:
            raise RuntimeError("committed candidate failed the independent behavior check")
        if candidate != inputs.synthesis.file_text.replace("count=value", "amount=value"):
            raise RuntimeError("transaction changed unrelated fixture bytes")
        (out / "qualification.json").write_text(
            json.dumps(
                {
                    "schema": "doctor-native-composition-qualification@1",
                    "tactician_planned": composed.tactician.is_planned,
                    "sealed_proof_verified": composed.proof.mutation_capable,
                    "native_synthesis_admitted": composed.synthesis.mutation_capable,
                    "graph_impact_closed": composed.impact.mutation_admissible,
                    "transaction_committed": report.result.changed,
                    "target_ref": inputs.target_ref,
                    "expected_base_commit": inputs.base_ref,
                    "committed_commit": committed,
                    "publication_verified": True,
                    "executed_validation_receipts": sum(
                        len(step.diagnostic_refs)
                        for group in transaction.group_receipts
                        for step in group.step_receipts
                    ),
                    "source_scope": ["pkg.py"],
                    "proof_scope": inputs.proof_scope,
                    "before_behavior_failed": True,
                    "after_behavior_passed": True,
                    "task_completion_authorized": False,
                    "provider_calls": 0,
                    "terminal_bench_result": False,
                },
                indent=2,
            )
        )
    else:
        raise RuntimeError("native composition did not admit its fixture repair")
    return runtime, inputs


if __name__ == "__main__":
    run(sys.argv[1])
