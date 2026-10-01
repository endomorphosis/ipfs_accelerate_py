"""Local header-contract proof, native Tactician, and observational proof index.

The kernel checks the guard on a sequence of characters, not arbitrary Python
execution. The reviewed AST translator is an explicit assumption. Neither the
index nor a local theorem closes Python's dynamic or interprocedural frontiers.
"""
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import sys

from ..analysis.deterministic_doctor_contracts import DeterministicDoctorFinding, DoctorRepairDisposition
from ..analysis import doctor_header_contracts as header_contracts
from ..analysis.doctor_header_contracts import OPERATOR_ID
from ..analysis.program_logic_prediction_contracts import ProgramLogicAuthorityRoots
from ..planning.deterministic_doctor_tactician import DeterministicDoctorTactician
from ..proof.deterministic_doctor_hammer import (
    DeterministicDoctorHammer, DoctorAuthoritativeProofReceipt, DoctorExactLoweringReceipt, DoctorExecutableRole,
    DoctorHammerBounds, DoctorPinnedExecutable, DoctorReviewedTheorem,
)
from ..proof.doctor_proof_cache import DoctorSealedReceiptStore
from ..proof.formal_verification_contracts import content_identity
from ..proof.proof_scope_index import (
    IndexedObligation, IndexedReceipt, IndexedScopeRecord, ProofInputKind,
    ProofScopeBlobRecord, ProofScopeKey, build_proof_scope_index,
)
from ..semantic_state.program_world_database import ProgramWorldDatabase
from .doctor_repair_composition import composition_snapshot
from .doctor_task_workflow import _PROVER_ADAPTER
from . import doctor_scoped_analysis as scoped_analysis
from . import doctor_security_ir as security_ir_bridge
from .supervisor_meta_index import SupervisorMetaIndex

OPERATOR = OPERATOR_ID
PROOF_SCOPE = (
    "For converted character sequences, the CR/LF/NUL guard rejects unsafe input "
    "and returns the original normalization on accepted input. Exact AST replay "
    "binds candidate bytes under the reviewed Python/WSGI protocol assumptions. "
    "This is not whole-program verification or full HTTP grammar conformance."
)

LEAN_SOURCE = r'''
def unsafeHeader (xs : List Char) : Bool :=
  xs.any (fun c => c == '\r' || c == '\n' || c == '\x00')
def guardHeader {α : Type} (xs : List Char) (normalized : α) : Option α :=
  match unsafeHeader xs with
  | true => none
  | false => some normalized
theorem rejected_unsafe {α : Type} (xs : List Char) (normalized : α)
    (h : unsafeHeader xs = true) : guardHeader xs normalized = none := by
  unfold guardHeader
  rw [h]
theorem preserved_safe {α : Type} (xs : List Char) (normalized : α)
    (h : unsafeHeader xs = false) : guardHeader xs normalized = some normalized := by
  unfold guardHeader
  rw [h]
theorem accepted_only_safe {α : Type} (xs : List Char) (normalized value : α)
    (h : guardHeader xs normalized = some value) : unsafeHeader xs = false := by
  cases hs : unsafeHeader xs with
  | false => rfl
  | true => unfold guardHeader at h
            rw [hs] at h
            contradiction
#print axioms rejected_unsafe
#print axioms preserved_safe
#print axioms accepted_only_safe
'''


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _implementation_bindings() -> dict[str, str]:
    """Pin the actual AST translator and declaration compiler, not just this wrapper."""
    from ipfs_datasets_py.logic.security_ir import formalization_adapter

    return {name: _sha(Path(module.__file__).read_bytes()) for name, module in (
        ("proof_wrapper_sha256", sys.modules[__name__]),
        ("header_ast_translator_sha256", header_contracts),
        ("security_ir_bridge_sha256", security_ir_bridge),
        ("security_ir_compiler_sha256", formalization_adapter),
    )}


def _replay_bound_candidate(*, scoped, contract: dict, candidate_cid: str) -> tuple[dict, str]:
    """Reconstruct the sole admitted candidate and any claimed SecurityIR bindings.

    A CID attached to the generic guard lemma is not evidence that arbitrary
    caller metadata describes that guard. Repeat the source/AST binding here
    even when the normal workflow has already checked it.
    """
    if (type(scoped) is not scoped_analysis.ScopedDoctorAnalysis or type(contract) is not dict
            or not isinstance(candidate_cid, str) or not candidate_cid):
        raise ValueError("exact native scoped analysis, header contract and candidate CID required")
    scoped.assert_current()
    security_fields = {"security_ir_declaration_cid", "security_ir_formalization_cid"}
    supplied_fields = set(contract) & security_fields
    if supplied_fields and supplied_fields != security_fields:
        raise ValueError("both independently verified SecurityIR bindings are required")
    try:
        protocol = header_contracts.WsgiHeaderProtocolContract(**contract["protocol"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("an exact reviewed header protocol contract is required") from exc
    _verified, spec, _paths = scoped_analysis._selection(
        scoped.repository, json.loads(scoped._admission_json), scoped.task_cid)
    analyses, candidates = [], []
    for output in spec["outputs"]:
        path = output["path"]
        if output["effect"] != "modify" or not path.endswith(".py"):
            continue
        if path not in scoped.sources:
            raise ValueError("declared Python output is absent from independently admitted source")
        analysis = header_contracts.analyze_http_header_contracts(
            scoped.sources[path].decode("utf-8"), protocol=protocol)
        analyses.append({"path": path, "status": analysis.status,
            "reason_codes": list(analysis.reason_codes), "source_sha256": analysis.source_sha256,
            "contracts": list(analysis.contracts), "open_frontiers": list(analysis.open_frontiers)})
        if analysis.candidate is not None:
            candidates.append((path, analysis.candidate))
    if len(candidates) != 1:
        raise ValueError("exactly one independently admitted header candidate is required")
    path, candidate = candidates[0]
    expected = dict(candidate.contract)
    if (content_identity(candidate.to_dict()) != candidate_cid
            or {name: value for name, value in contract.items() if name not in security_fields} != expected
            or not header_contracts.verify_header_candidate(scoped.sources[path].decode("utf-8"), candidate)):
        raise ValueError("header contract or candidate differs from exact scoped AST replay")
    if supplied_fields:
        compiled = security_ir_bridge.compile_header_security_ir(scoped=scoped, analyses=analyses)
        bindings = {"security_ir_declaration_cid": compiled.report["declaration_cid"],
                    "security_ir_formalization_cid": compiled.report["artifact_cid"]}
        if any(contract[name] != value for name, value in bindings.items()):
            raise ValueError("SecurityIR binding differs from independently compiled scoped declarations")
        expected.update(bindings)
    scoped.assert_current()
    return expected, path


def prove_header_contract(*, scoped, contract: dict, candidate_cid: str,
                          state: Path, lean: Path, z3: Path) -> tuple[dict, object | None]:
    """Execute real native planning, solver and kernel for one bound candidate."""
    implementation_bindings = _implementation_bindings()
    contract, candidate_path = _replay_bound_candidate(
        scoped=scoped, contract=contract, candidate_cid=candidate_cid)
    state = Path(state).absolute()
    if state.resolve() != state or state.is_relative_to(scoped.repository) or state.exists():
        raise ValueError("proof state must be a new external directory")
    for executable in (lean, z3):
        if not executable.is_file() or not os.access(executable, os.X_OK):
            return {"status": "unavailable", "reason_codes": ["required_local_prover_unavailable"],
                    "provider_calls": 0, "proof_scope": PROOF_SCOPE}, None
    lean, z3 = lean.resolve(), z3.resolve()
    contract_cid = content_identity(contract)
    consequence = content_identity({"operator": OPERATOR, "contract": contract_cid,
        "candidate": candidate_cid, "analysis": scoped.report["analysis_cid"]})
    source = ("-- theorem:" + OPERATOR + " property:reject-control-preserve-safe "
              "claim:local-header-guard " + consequence + "\n" + LEAN_SOURCE)
    translator = content_identity({"operator": OPERATOR, "implementations": implementation_bindings})
    toolchain = content_identity({"lean": _sha(lean.read_bytes()), "z3": _sha(z3.read_bytes()),
        "adapter": _sha(_PROVER_ADAPTER.encode()), "translator": translator})
    roots = replace(scoped.snapshot.roots, toolchain_id=toolchain, translator_id="translator:" + translator)
    snapshot = composition_snapshot(scoped.snapshot, roots)
    finding = DeterministicDoctorFinding(roots=roots,
        finding_id=content_identity({"contract": contract_cid, "analysis": scoped.report["analysis_cid"]}),
        snapshot_id=snapshot.snapshot_id, disposition=DoctorRepairDisposition.SUPPORTED,
        observed_fact_refs=(scoped.report["analysis_cid"],), expected_behavior_refs=(contract_cid,),
        affected_symbol_refs=(OPERATOR,), invalidation_refs=(roots.tree_id,))
    plan = DeterministicDoctorTactician().plan_finding(finding, snapshot=snapshot, current_roots=roots)
    report = {"status": "abstained", "reason_codes": list(plan.reason_codes), "provider_calls": 0,
        "proof_scope": PROOF_SCOPE, "contract_cid": contract_cid, "candidate_cid": candidate_cid,
        "candidate_path": candidate_path, "translator_cid": translator,
        "implementation_bindings": implementation_bindings,
        "security_ir_bindings_verified": "security_ir_declaration_cid" in contract,
        "security_ir_obligations_discharged": False,
        "tactician": plan.to_dict(), "whole_program_proved": False, "completion_authority": False}
    if contract.get("security_ir_declaration_cid"):
        report["security_ir_declaration_cid"] = contract["security_ir_declaration_cid"]
    if not plan.is_planned:
        return report, None
    proof_roots = ProgramLogicAuthorityRoots(**{name: getattr(roots, name) if hasattr(roots, name)
        else content_identity({"dimension": name, "analysis": scoped.report["analysis_cid"]})
        for name in ProgramLogicAuthorityRoots.__dataclass_fields__ if name != "SCHEMA"})
    theorem = DoctorReviewedTheorem(roots=proof_roots, theorem_id="theorem:" + OPERATOR,
        property_id="property:reject-control-preserve-safe", claim_id="claim:local-header-guard",
        consequence_ref=consequence, theorem_body=source, body_format="lean4", premise_ids=(contract_cid,),
        assumption_ids=("assumption:reviewed-exact-python-ast-translation", "assumption:declared-wsgi-protocol"),
        review_receipt_id=content_identity({"operator": OPERATOR, "implementations": implementation_bindings,
                                          "contract": contract_cid}),
        translator_id=roots.translator_id, toolchain_id=toolchain, policy_id=roots.policy_id)
    lowering = DoctorExactLoweringReceipt.create(theorem, logic_ir_statement=source, native_statement=source)
    # Arbitrary string x and arbitrary normalization n: accepted output must be
    # unchanged and a rejected input cannot return. The AST adapter separately
    # binds the Python guard/template; no solver success promotes that assumption.
    smt = r'''(set-logic QF_SLIA)
(declare-const x String)
(declare-const n String)
(define-fun bad () Bool (or (str.contains x "\u{d}") (str.contains x "\u{a}") (str.contains x "\u{0}")))
(define-fun accepted () Bool (not bad))
(define-fun result () String (ite accepted n ""))
(assert (or (and bad accepted) (and accepted (distinct result n))))
(check-sat)
'''
    cfg = {"lean": str(lean), "z3": str(z3), "lean_sha256": _sha(lean.read_bytes()),
        "z3_sha256": _sha(z3.read_bytes()), "source_sha256": _sha(source.encode()), "smt": smt,
        "kernel_id": "kernel:lean4-header-guard",
        "expected_axioms": [f"'{name}' does not depend on any axioms" for name in
                            ("rejected_unsafe", "preserved_safe", "accepted_only_safe")]}
    executable = Path(sys.executable).resolve()
    pins = tuple(DoctorPinnedExecutable(role=DoctorExecutableRole(role), executable_path=str(executable),
        executable_sha256="sha256:" + _sha(executable.read_bytes()),
        argv=("-I", "-c", _PROVER_ADAPTER, str(Path(__file__).resolve().parents[3]), json.dumps(cfg), role),
        verifier_id="verifier:" + role + ":header-guard", toolchain_id=toolchain, environment_id=roots.environment_id)
        for role in ("solver", "kernel"))
    state.mkdir(parents=True, mode=0o700)
    hammer = DeterministicDoctorHammer(bounds=DoctorHammerBounds(wall_time_ms=60_000),
        authoritative_store=DoctorSealedReceiptStore(state / "proof.sqlite", authority_id="authority:" + OPERATOR),
        trusted_executable_pins=pins)
    proof = hammer.verify_authoritative(theorem, lowering, solver_pin=pins[0], kernel_pin=pins[1],
        current_roots=proof_roots, eligible_consequence_refs=(consequence,))
    if proof.mutation_capable:
        proof = hammer.reverify_authoritative(proof, current_roots=proof_roots)
    scoped.assert_current()
    if _implementation_bindings() != implementation_bindings:
        raise ValueError("reviewed translator or SecurityIR compiler changed during proof execution")
    report.update(status="proved_local_contract" if proof.mutation_capable else "abstained",
        reason_codes=list(proof.reason_codes), proof_receipt_id=proof.content_id,
        proof=proof.to_dict(), toolchain_cid=toolchain, theorem_cid=theorem.content_id,
        proof_assumptions=list(theorem.assumption_ids))
    (state / "proof-report.json").write_text(json.dumps(report, sort_keys=True, indent=2) + "\n")
    return report, proof if proof.mutation_capable else None


def persist_contract_world(*, scoped, analyses: list[dict], proof_report: dict,
                           state: Path, task_id: str, proof=None, formalization=None) -> dict:
    """Build, hydrate and project the native dependency index and world catalog."""
    scoped.assert_current()
    if proof_report.get("status") not in {"proved_local_contract", "abstained", "unavailable", "unsupported"}:
        raise ValueError("unknown local contract proof state")
    if proof_report["status"] == "proved_local_contract":
        if (type(proof) is not DoctorAuthoritativeProofReceipt or not proof.mutation_capable
                or proof.content_id != proof_report.get("proof_receipt_id")
                or proof.roots.tree_id != scoped.snapshot.roots.tree_id
                or proof.theorem is None or proof.theorem.theorem_id != "theorem:" + OPERATOR
                or proof.theorem.premise_ids != (proof_report.get("contract_cid"),)
                or proof.selected_consequence_ref != content_identity({"operator": OPERATOR,
                    "contract": proof_report.get("contract_cid"), "candidate": proof_report.get("candidate_cid"),
                    "analysis": scoped.report["analysis_cid"]})):
            raise ValueError("proved index observation requires exact source-bound sealed native evidence")
    elif proof is not None:
        raise ValueError("unproved index observation cannot carry proof authority")
    state = Path(state).absolute()
    if state.resolve() != state or state.is_relative_to(scoped.repository) or state.exists():
        raise ValueError("contract catalog must be a new external directory")
    state.mkdir(parents=True, mode=0o700)
    scopes = []
    keys = []
    scope_ids = []
    for name, digest in scoped.report["source_hashes"].items():
        key = ProofScopeKey(ProofInputKind.FILE, name)
        scope_id = content_identity({"path": name, "sha256": digest})
        scope = IndexedScopeRecord(scope_id, name, digest, (key,))
        scopes.append(ProofScopeBlobRecord(name, digest, (scope,)))
        keys.append(key)
        scope_ids.append(scope_id)
    for kind, value in ((ProofInputKind.POLICY, scoped.report["manifest_cid"]),
                        (ProofInputKind.ASSUMPTION, "declared-wsgi-protocol"),
                        (ProofInputKind.TEMPLATE, OPERATOR)):
        keys.append(ProofScopeKey(kind, value))
    if proof_report.get("toolchain_cid"):
        keys.append(ProofScopeKey(ProofInputKind.TOOLCHAIN, proof_report["toolchain_cid"]))
    for kind, field in ((ProofInputKind.PREMISE, "contract_cid"), (ProofInputKind.PROGRAM_SNAPSHOT, "candidate_cid")):
        if proof_report.get(field):
            keys.append(ProofScopeKey(kind, proof_report[field]))
    if proof_report.get("security_ir_declaration_cid"):
        keys.append(ProofScopeKey(ProofInputKind.IR_FAMILY, "security_ir"))
        keys.append(ProofScopeKey(ProofInputKind.IR_ROOT, proof_report["security_ir_declaration_cid"]))
    obligation_id = proof_report.get("contract_cid") or content_identity({"analyses": analyses})
    obligation = IndexedObligation(obligation_id, tuple(scope_ids), tuple(keys), (),
        {"scope": PROOF_SCOPE, "analysis_cid": scoped.report["analysis_cid"], "status": proof_report["status"]})
    receipts = ()
    if proof_report["status"] == "proved_local_contract":
        receipts = (IndexedReceipt(proof_report["proof_receipt_id"], obligation_id,
                    tuple(scope_ids), tuple(keys), {"proof_scope": PROOF_SCOPE, "completion_authority": False}),)
    index = build_proof_scope_index(scope_blobs=scopes, obligations=(obligation,), receipts=receipts,
                                   root_id=scoped.report["source_tree_id"])
    index_body = index.to_dict()
    artifact = state / "proof-scope-index.json"
    artifact.write_text(json.dumps(index_body, sort_keys=True, indent=2) + "\n")
    record = {"task_id": task_id, "board": "doctor-contracts", "operation": "scoped_static_contracts",
        "analysis": scoped.report, "contracts": analyses, "proof_status": proof_report["status"],
        "proof_receipt_id": proof_report.get("proof_receipt_id"),
        "proof_scope": PROOF_SCOPE, "proof_index": index_body,
        "open_frontiers": scoped.report["open_frontiers"], "whole_program_proved": False,
        "formalization": formalization,
        "proposal_only": True, "completion_authority": False}
    world = ProgramWorldDatabase(state / "contracts.duckdb", state / "contracts-lake")
    persisted = world.persist(record)
    hydrated = world.records_for_decision(task_id=task_id, operation="scoped_static_contracts")
    if hydrated["n"] != 1 or hydrated["records"][0]["payload"]["proof_index"] != index_body:
        raise ValueError("native contract world hydration differs")
    meta = SupervisorMetaIndex(state / "metadata.duckdb", state / "metadata-lake")
    for kind, locator, ref in (("world_model", state / "contracts.duckdb", persisted["record_cid"]),
                               ("proof_certificate", artifact, content_identity({"artifact_sha256": _sha(artifact.read_bytes())}))):
        catalog = meta.register_catalog(kind=kind, locator_ref=str(locator),
            repository_id=scoped.snapshot.roots.repository_id, tree_id=scoped.report["source_tree_id"], project=False)
        meta.link_identity(subject_kind="task_id", subject_ref=task_id, catalog_id=catalog["catalog_id"],
                           record_kind=kind, record_ref=ref, project=False)
    projection = meta.project_ducklake()
    scoped.assert_current()
    return {"world_record": persisted, "metadata": projection, "artifact": str(artifact),
        "artifact_sha256": _sha(artifact.read_bytes()), "active_receipt_ids": list(index.active_receipt_ids),
        "hydrated": True, "completion_authority": False, "whole_program_proved": False}
