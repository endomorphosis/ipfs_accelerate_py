"""Bounded automatic Doctor repair for an existing, independently admitted task.

Reviewed operators repair one unambiguous local keyword/signature mismatch
or a direct call that ignores an explicit, closed local import alias. Expectations come from the existing declaration,
not a candidate or model. Native proof, synthesis, impact and worktree gates
produce an isolated candidate ref. Publication and task completion remain the
ordinary supervisor's responsibility. Unsupported cases propose native Doctor
plan successors; they never rewrite admitted tasks or declare completion.
"""
from __future__ import annotations

import ast
import base64
from dataclasses import dataclass, replace
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import sys
from typing import Any, Mapping

from ..analysis.deterministic_doctor_contracts import (
    DeterministicDoctorFinding, DoctorMode, DoctorOperatorKind, DoctorRepairDisposition,
)
from ..analysis.deterministic_doctor_impact import DoctorImpactRequest
from ..analysis.program_dependency_graph import PathSource, ProgramDependencyGraph
from ..analysis.program_graph import ProgramGraphRoots
from ..analysis.program_logic_prediction_contracts import ProgramLogicAuthorityRoots
from ..analysis.planning_analysis_factory import PlanningAnalysisSecretError
from ..objectives.doctor_plan_refill import (
    DoctorPlanContext, DoctorPlanNode, DoctorPlanRefill, DoctorPlanResidual, DoctorResidualKind,
)
from ..planning.deterministic_doctor_synthesis import DoctorSynthesisRequest
from ..planning.deterministic_doctor_transforms import build_default_doctor_operator_registry, make_edit_site
from ..proof.deterministic_doctor_hammer import (
    DeterministicDoctorHammer, DoctorExactLoweringReceipt, DoctorExecutableRole,
    DoctorHammerBounds, DoctorPinnedExecutable, DoctorReviewedTheorem,
    MAX_TEXT_BYTES, MAX_THEOREM_BYTES,
)
from ..proof.doctor_proof_cache import DoctorSealedReceiptStore
from ..proof.formal_verification_contracts import canonical_json, content_identity
from ..task_sources.intent_repository import IntentRepository
from . import local_planning_admission as local
from .deterministic_doctor_runtime import DeterministicDoctorRuntime
from .doctor_repair_composition import (
    DoctorCompositionError, DoctorCompositionInputs, _inert_function_module,
    _signature, composition_snapshot, operator_consequence_ref, operator_contract_reconstruction,
)
from .doctor_worktree_adapter import DoctorWorktreeAdapter
from . import doctor_alias_contract as alias_owner
from .doctor_source_partition import terminal_doctor_source_partition


OPERATOR = "closed-local-keyword-rename@1"
PROOF_SCOPE = (
    "Pair-name substitution preserves the argument value; independent AST and "
    "signature replay binds the exact Python edit. No whole-program correctness theorem."
)

# The complete adapter text and configuration are part of each trusted pin.
# Both roles execute real Lean; solver additionally runs real Z3. Target source
# is never imported or evaluated by either proof process.
_PROVER_ADAPTER = r'''
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
 assert r.returncode==0 and r.stdout.strip()=='unsat'
else:
 assert role=='kernel' and d['native_receipt']['proof_object']==source
with tempfile.TemporaryDirectory(prefix='doctor-keyword-') as tmp:
 p=pathlib.Path(tmp)/'Proof.lean';p.write_text(source)
 r=subprocess.run([cfg['lean'],str(p)],text=True,capture_output=True,timeout=30)
 assert r.returncode==0 and 'sorry' not in r.stdout+r.stderr
 assert set(r.stdout.strip().splitlines())==set(cfg['expected_axioms'])
base={'theorem_cid':content_identity(t),'lowering_cid':content_identity(l),'property_id':t['property_id'],'consequence_ref':t['consequence_ref'],'premise_ids':t['premise_ids']}
if role=='solver':base.update(status='proved',proof_object=source)
else:base.update(status='kernel_verified',native_receipt_cid=content_identity(d['native_receipt']),proof_object_cid=content_identity({'proof_object':source}),kernel_id=cfg.get('kernel_id','kernel:lean4-closed-keyword'))
sys.stdout.write(canonical_json(base))
'''


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _candidates(text: str) -> list[dict[str, Any]]:
    """Find every supported edit, without guessing between valid bindings."""
    tree, declarations = _inert_function_module(text)
    by_name = {node.name: node for node in declarations}
    if len(by_name) != len(declarations):
        raise DoctorCompositionError("duplicate local declarations")
    lines = text.splitlines(keepends=True)
    candidates = []
    for call in (node for node in ast.walk(tree) if isinstance(node, ast.Call)):
        if not isinstance(call.func, ast.Name) or call.func.id not in by_name:
            continue
        if any(kw.arg is None for kw in call.keywords) or any(isinstance(arg, ast.Starred) for arg in call.args):
            continue
        signature = _signature(by_name[call.func.id])
        keywords = {kw.arg: None for kw in call.keywords}
        if len(keywords) != len(call.keywords):
            continue
        try:
            signature.bind(*([None] * len(call.args)), **keywords)
        except TypeError:
            pass
        else:
            continue
        for keyword in call.keywords:
            if keyword.arg in signature.parameters:
                continue
            for replacement in signature.parameters:
                if replacement in keywords or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", replacement):
                    continue
                repaired = {name: value for name, value in keywords.items() if name != keyword.arg}
                repaired[replacement] = None
                try:
                    signature.bind(*([None] * len(call.args)), **repaired)
                except TypeError:
                    continue
                offset = len("".join(lines[:keyword.lineno - 1])) + len(
                    lines[keyword.lineno - 1].encode()[:keyword.col_offset].decode()
                )
                candidates.append({
                    "subject": call.func.id, "previous": keyword.arg,
                    "replacement": replacement, "offset": offset,
                    "declaration": ast.dump(by_name[call.func.id]),
                })
    return candidates


@dataclass(frozen=True)
class PreparedDoctorTaskRepair:
    runtime: DeterministicDoctorRuntime
    intent: IntentRepository
    admission: Mapping[str, Any]
    task: Mapping[str, Any]
    plan_context: DoctorPlanContext
    refill: DoctorPlanRefill
    state_root: Path
    inputs: DoctorCompositionInputs | None
    report: Mapping[str, Any]
    analysis_failure: str = ""


def _analysis_unavailable_observation(prepared: PreparedDoctorTaskRepair) -> dict:
    """A sanitized capability observation, never a successful source analysis."""
    if prepared.analysis_failure != "secret_material":
        raise DoctorCompositionError("unsupported Doctor analysis failure")
    contract = prepared.task["body"][local.CONTRACT_KEY]["payload"]
    payload = {
        "schema": "supervisor-doctor-analysis-unavailable@1",
        "task_cid": prepared.task["task_cid"], "task_revision": prepared.task["revision"],
        "manifest_cid": contract["manifest_cid"],
        "source_tree_id": local._tree(contract["manifest"]["payload"]["sources"]),
        "status": "unavailable", "reason_code": "doctor_analysis_secret_screen_refused",
        "native_analysis_reason_code": prepared.analysis_failure,
        "diagnostic_snapshot_created": False, "finding_created": False, "proof_created": False,
        "provider_calls": 0, "execution_authority": False, "completion_authority": False,
    }
    return {**payload, "observation_cid": content_identity(payload)}


def _residual_evidence(prepared: PreparedDoctorTaskRepair) -> tuple[str, tuple[str, ...]]:
    if prepared.analysis_failure:
        observation = _analysis_unavailable_observation(prepared)
        if prepared.report.get("analysis_observation") != observation:
            raise DoctorCompositionError("unavailable Doctor observation differs from admitted source")
        return observation["source_tree_id"], (observation["observation_cid"],)
    evidence = prepared.runtime.evidence
    if evidence is None:
        raise DoctorCompositionError("native Doctor evidence is unavailable")
    return evidence.snapshot.roots.tree_id, (evidence.evidence_id,)


def _refill(prepared: PreparedDoctorTaskRepair, reasons: tuple[str, ...], *, transaction_id: str = "") -> dict:
    task = prepared.task
    contract = task["body"][local.CONTRACT_KEY]["payload"]
    outputs = tuple(row["path"] for row in contract["task_spec"]["outputs"])
    operator = prepared.report.get("operator", OPERATOR)
    issue = content_identity({"task_cid": task["task_cid"], "reasons": list(reasons), "operator": operator})
    root_id, evidence_refs = _residual_evidence(prepared)
    capability = ("doctor:screened-source-analysis" if prepared.analysis_failure else
        "doctor:local-lean-z3-proof" if "required_local_prover_unavailable" in reasons else "")
    residual = DoctorPlanResidual(
        issue_id=issue, root_id=root_id,
        kind=DoctorResidualKind.CAPABILITY_GAP if capability else DoctorResidualKind.UNSUPPORTED,
        parent_task_cid=task["task_cid"],
        parent_goal_cid=task["goal_cid"], parent_goal_id=prepared.plan_context.nodes[0].goal_id,
        plan_id=task["plan_cid"],
        predicted_files=outputs[:8], context_paths=outputs[:8],
        attempted_strategies=("native-source-analysis",) if prepared.analysis_failure else (operator,),
        required_capability=capability or "doctor:operator-contract-coverage",
        reason_codes=reasons, evidence_refs=evidence_refs,
        validation_commands=tuple(shlex.join(check["argv"]) for check in contract["task_spec"]["validations"]),
        title="Resolve remaining admitted repair obligations", transaction_id=transaction_id,
        metadata={"omitted_output_paths": max(0, len(outputs) - 8)},
        rationale="The bounded Doctor operator did not establish complete repair; retain native admission and validation gates.",
    )
    result = prepared.refill.refill((residual,), plan=prepared.plan_context)
    return result.to_dict()


def _assert_task_current(prepared: PreparedDoctorTaskRepair) -> None:
    local.verify_local_benchmark_admission(prepared.admission, initial=True)
    current = prepared.intent.get_task(prepared.task["task_cid"])
    if current is None or content_identity(current) != content_identity(prepared.task):
        raise DoctorCompositionError("native task changed during Doctor preparation")


def prepare_doctor_task_repair(
    *, runtime: DeterministicDoctorRuntime, intent: IntentRepository,
    admission: Mapping[str, Any], task_cid: str, state_root: Path,
    solver_executable: Path, kernel_executable: Path, candidate_ref: str,
    refill: DoctorPlanRefill | None = None,
) -> PreparedDoctorTaskRepair:
    """Select and prepare one reviewed operator from current admitted source.

    The caller supplies a dedicated native candidate ref already at the signed
    baseline. Creating/allocating that ref is not repair or publication. The
    real Doctor transaction alone advances it after its independent gates.
    """
    if type(runtime) is not DeterministicDoctorRuntime or not isinstance(intent, IntentRepository):
        raise DoctorCompositionError("native Doctor runtime and intent owner required")
    verified = local.verify_local_benchmark_admission(admission, initial=True)
    manifest, graph = verified["manifest"], verified["graph"]
    root = runtime.checkout_root
    if root != Path(manifest["repository"]):
        raise DoctorCompositionError("Doctor repository differs from signed admission")
    task = intent.get_task(task_cid)
    graph_tasks = [item for item in graph.tasks if item.task_cid == task_cid]
    if task is None or len(graph_tasks) != 1 or task["status"] not in {"ready", "in_progress"}:
        raise DoctorCompositionError("Doctor requires an existing active admitted task")
    contract, _, _, _ = local._contract(task["body"], task_cid)
    expected_task = graph_tasks[0]
    if (contract["manifest_cid"] != verified["receipt"]["manifest_cid"]
            or task["task_alias"] != expected_task.task_key or task["goal_cid"] != expected_task.goal_cid):
        raise DoctorCompositionError("Doctor task differs from the signed graph")
    state = Path(state_root).absolute()
    if state.resolve() != state or state.is_relative_to(root) or state.exists():
        raise DoctorCompositionError("Doctor state must be a new external non-symlink directory")
    if not re.fullmatch(r"refs/heads/doctor/[A-Za-z0-9][A-Za-z0-9._/-]*", candidate_ref):
        raise DoctorCompositionError("Doctor requires an isolated candidate ref")
    goal = next(item for item in graph.goals if item.goal_cid == task["goal_cid"])
    plan_row = intent.get_plan(task["plan_cid"])
    plan_context = DoctorPlanContext(
        plan_root=task["plan_cid"], plan_revision=plan_row["revision"],
        nodes=(DoctorPlanNode(node_cid=task_cid, kind="task", goal_id=goal.goal_key,
                             lifecycle="running" if task["status"] == "in_progress" else "unstarted",
                             predicted_files=tuple(row["path"] for row in contract["task_spec"]["outputs"])),),
    )
    prepared = PreparedDoctorTaskRepair(runtime, intent, json.loads(canonical_json(admission)), json.loads(canonical_json(task)),
        plan_context, refill or DoctorPlanRefill(), state, None, {})
    report = {
        "schema": "supervisor-doctor-task-preparation@1", "operator": OPERATOR,
        "task_cid": task_cid, "task_id": task["task_alias"], "task_revision": task["revision"],
        "plan_cid": task["plan_cid"], "goal_cid": task["goal_cid"],
        "manifest_cid": verified["receipt"]["manifest_cid"],
        "provider_calls": 0, "completion_authority": False,
        "canonical_source_edits": 0, "canonical_task_mutated": False,
    }
    try:
        evidence = runtime.build_evidence(refresh=True)
    except PlanningAnalysisSecretError:
        # The optional whole-checkout analysis remains refused. Never retry
        # without screening, narrow away consumers, or retain flagged content.
        # The separate signed task still needs current admission before its
        # existing model route may receive this candidate-only capability gap.
        _assert_task_current(prepared)
        prepared = replace(prepared, analysis_failure="secret_material")
        reasons = ("doctor_analysis_secret_screen_refused",)
        unavailable = {**report, "status": "residual", "reason_codes": list(reasons),
            "analysis_status": "unavailable", "analysis_observation": _analysis_unavailable_observation(prepared),
            "evidence_id": None, "diagnostic_snapshot_id": None, "finding_count": None,
            "source_hashes": None}
        prepared = replace(prepared, report=unavailable)
        return replace(prepared, report={**unavailable, "plan_refill": _refill(prepared, reasons)})
    reasons = []
    ledger = {row.path: row.content_digest.removeprefix("sha256:") for row in evidence.source_inventory}
    if (len(ledger) != len(evidence.source_inventory) or set(ledger) != set(manifest["sources"])
            or "diagnostic_source_bound_reached" in evidence.notes):
        reasons.append("unsupported_or_incomplete_source_inventory")
    elif any(ledger[name] != manifest["sources"][name]["sha256"] for name in ledger):
        raise DoctorCompositionError("native Doctor inventory differs from signed source")
    partition = terminal_doctor_source_partition(repository=root, admission=admission, task_cid=task_cid)
    program_paths = tuple(sorted(ledger)) if partition is None else partition.program_paths
    if partition is not None and any(role == "task_data" for _, role, _ in partition.support_hashes):
        # A validated data parser establishes input classification, not a
        # behavioral contract for its consumers. Retain every hash and defer
        # repair until the operator can reconstruct that dependency itself.
        reasons.append("doctor_task_data_contract_unavailable")
    support_kinds = {} if partition is None else {
        name: {"instruction": "text_reference", "task_profile": "structured_data",
               "structural_smoke": "semantic_ast"}.get(role, "unsupported_task_data")
        for name, role, _ in partition.support_hashes
    }
    if any(row.coverage_kind != support_kinds.get(row.path, "semantic_ast")
           or (row.path not in support_kinds and not row.path.endswith(".py"))
           for row in evidence.source_inventory):
        reasons.append("unsupported_or_incomplete_source_inventory")
    selected = []
    aliases = {}
    for output in contract["task_spec"]["outputs"]:
        name = output["path"]
        if output["effect"] != "modify" or name not in program_paths or not name.endswith(".py"):
            reasons.append("unsupported_output_effect_or_language")
            continue
        try:
            text = (root / name).read_text(encoding="utf-8")
            selected.extend({**candidate, "path": name, "text": text} for candidate in _candidates(text))
        except (DoctorCompositionError, SyntaxError, UnicodeError):
            # Imports cannot enter the legacy inert-module contract. A separate
            # strict source-derived contract may authorize exactly one alias
            # edit; unsupported donor or scope shapes retain the residual.
            try:
                if "unsupported_or_incomplete_source_inventory" in reasons:
                    raise alias_owner.ImportedAliasContractError("incomplete alias source inventory")
                sources = alias_owner.read_imported_alias_sources(repository=root,
                    source_hashes={path: ledger[path] for path in program_paths})
                alias = alias_owner.discover_imported_alias_repair(sources=sources, path=name)
                if alias is None:
                    raise alias_owner.ImportedAliasContractError("no closed alias repair")
                alias.assert_current(root)
                aliases[name] = alias
                selected.append({**alias.to_dict(), "text": sources[name]})
            except (alias_owner.ImportedAliasContractError, OSError, SyntaxError, UnicodeError):
                reasons.append("unsupported_module_or_signature_shape")
    if len(selected) != 1:
        reasons.append("ambiguous_supported_repairs" if len(selected) > 1 else "no_supported_keyword_mismatch")
    report = {**report, "analysis_status": "available", "evidence_id": evidence.evidence_id,
        "diagnostic_snapshot_id": evidence.snapshot.snapshot_id, "finding_count": len(evidence.findings),
        "source_hashes": ledger,
    }
    if partition is not None:
        report["source_partition"] = partition.observation()
    if reasons:
        _assert_task_current(prepared)
        return replace(prepared, report={**report, "status": "residual", "reason_codes": sorted(set(reasons)),
            "plan_refill": _refill(prepared, tuple(sorted(set(reasons))))})
    choice = selected[0]
    alias_contract = aliases.get(choice["path"])
    operator = alias_contract.operator if alias_contract is not None else OPERATOR
    proof_scope = alias_owner.PROOF_SCOPE if alias_contract is not None else PROOF_SCOPE
    report = {**report, "operator": operator}
    prepared = replace(prepared, report=report)
    try:
        lean, z3 = Path(kernel_executable).resolve(strict=True), Path(solver_executable).resolve(strict=True)
        if not all(path.is_file() and os.access(path, os.X_OK) for path in (lean, z3)):
            raise OSError("prover is not executable")
    except OSError:
        _assert_task_current(prepared)
        reasons = ("required_local_prover_unavailable",)
        return replace(prepared, report={**report, "status": "residual", "reason_codes": list(reasons),
            "plan_refill": _refill(prepared, reasons)})
    toolchain = content_identity({"lean": _sha(lean.read_bytes()), "z3": _sha(z3.read_bytes()), "adapter": _sha(_PROVER_ADAPTER.encode())})
    observed = evidence.snapshot.roots
    program_graph = ProgramDependencyGraph(ProgramGraphRoots(
        forest_id=observed.forest_id, tree_id=observed.tree_id, overlay_id=observed.overlay_id,
        coverage_id=observed.file_root_id, included_roots=program_paths, toolchain_id=toolchain,
    )).build([PathSource(path=name, source=(root / name).read_text(), language="python") for name in program_paths])
    base = manifest["baseline_commit"]
    roots = replace(observed, graph_id=program_graph.graph_id, toolchain_id=toolchain,
        lease_id=content_identity({"task_cid": task_cid, "source": ledger, "candidate_ref": candidate_ref, "base": base,
            **({"source_partition": partition.observation()["partition_cid"]} if partition is not None else {})}),
        translator_id="translator:" + operator)
    snapshot = composition_snapshot(evidence.snapshot, roots)
    expected = (alias_contract.contract_id if alias_contract is not None else
        content_identity({"declaration": choice["declaration"], "source": ledger[choice["path"]]}))
    finding = DeterministicDoctorFinding(
        roots=roots, finding_id=content_identity({"operator": operator, "choice": {k:v for k,v in choice.items() if k != "text"}, "sources": ledger}),
        snapshot_id=snapshot.snapshot_id, disposition=DoctorRepairDisposition.SUPPORTED,
        observed_fact_refs=(content_identity({("callee" if alias_contract is not None else "keyword"): choice["previous"],
            "source": ledger[choice["path"]]}),),
        expected_behavior_refs=(expected,), affected_symbol_refs=(choice["subject"],),
        invalidation_refs=(roots.tree_id,),
    )
    proposal = build_default_doctor_operator_registry(roots).propose(
        DoctorOperatorKind.EXACT_RENAME, make_edit_site(choice["path"], choice["previous"], start=choice["offset"]),
        obligation_refs=(finding.content_id,), parameter_name=choice["replacement"], previous_parameter_name=choice["previous"],
    )
    synthesis = DoctorSynthesisRequest(roots=roots, proposal=proposal, span_text=choice["previous"],
        file_text=choice["text"], value_ref=expected, placement_ref=proposal.edit_site.content_id)
    try:
        operator_contract_reconstruction(synthesis, choice["subject"], alias_contract=alias_contract)
    except DoctorCompositionError:
        _assert_task_current(prepared)
        reasons = ("unsupported_binding_scope",)
        return replace(prepared, report={**report, "status": "residual", "reason_codes": list(reasons),
            "plan_refill": _refill(prepared, reasons)})
    expected_hash = "sha256:" + _sha(choice["replacement"].encode())
    consequence = operator_consequence_ref(synthesis, expected_hash, (expected,))
    proof_roots = ProgramLogicAuthorityRoots(**{name: getattr(roots, name) if hasattr(roots, name)
        else content_identity({"name": name, "finding": finding.content_id})
        for name in ProgramLogicAuthorityRoots.__dataclass_fields__ if name != "SCHEMA"})
    label = json.dumps(choice["replacement"])
    lean_source = (
        f"-- theorem:{OPERATOR} property:keyword-value-preservation claim:exact-keyword-binding {consequence}\n"
        f"def renameKeyword {{α : Type}} (value : α) : String × α := ({label}, value)\n"
        f"theorem selected_name {{α : Type}} (value : α) : (renameKeyword value).1 = {label} := by rfl\n"
        "theorem preserved_value {α : Type} (value : α) : (renameKeyword value).2 = value := by rfl\n"
        "#print axioms selected_name\n#print axioms preserved_value\n"
    )
    projection = alias_contract.formal_projection(consequence) if alias_contract is not None else None
    if projection is not None:
        lean_source = projection["lean"]
        if len(lean_source.encode("utf-8")) > MAX_THEOREM_BYTES:
            _assert_task_current(prepared)
            reasons = ("operator_proof_bounds_exceeded",)
            return replace(prepared, report={**report, "status": "residual", "reason_codes": list(reasons),
                "plan_refill": _refill(prepared, reasons)})
    theorem = DoctorReviewedTheorem(roots=proof_roots, theorem_id="theorem:" + operator,
        property_id="property:source-alias-argument-preservation" if projection else "property:keyword-value-preservation",
        claim_id="claim:closed-local-alias-binding" if projection else "claim:exact-keyword-binding",
        consequence_ref=consequence, theorem_body=lean_source, body_format="lean4", premise_ids=(expected,),
        assumption_ids=(("assumption:exact-ast-identifier-projection", "assumption:closed-local-module-resolution")
                        if projection else ("assumption:exact-ast-identifier-projection",)),
        review_receipt_id=content_identity({"reviewed_operator": operator, "implementation": _sha(Path(__file__).read_bytes()),
            **({"alias_contract_implementation": _sha(Path(alias_owner.__file__).read_bytes())} if projection else {}),
            "consequence": consequence}),
        translator_id=roots.translator_id, toolchain_id=toolchain, policy_id=roots.policy_id)
    lowering = DoctorExactLoweringReceipt.create(theorem, logic_ir_statement=lean_source, native_statement=lean_source)
    config = {"lean": str(lean), "z3": str(z3), "lean_sha256": _sha(lean.read_bytes()), "z3_sha256": _sha(z3.read_bytes()),
        "source_sha256": _sha(lean_source.encode()),
        "smt": f'(set-logic QF_SLIA)\n(declare-const value Int)\n(assert (or (not (= {label} {label})) (not (= value value))))\n(check-sat)\n',
        "expected_axioms": ["'selected_name' does not depend on any axioms", "'preserved_value' does not depend on any axioms"]}
    if projection is not None:
        config.update(smt=projection["smt"], expected_axioms=projection["expected_axioms"],
                      kernel_id="kernel:lean4-closed-imported-alias")
    executable = Path(sys.executable).resolve()
    command_prefix = ("-I", "-c", _PROVER_ADAPTER, str(Path(__file__).resolve().parents[3]), json.dumps(config))
    if projection is not None and any(len(arg.encode("utf-8")) > MAX_TEXT_BYTES
                                     for arg in (*command_prefix, "solver", "kernel")):
        _assert_task_current(prepared)
        reasons = ("operator_proof_bounds_exceeded",)
        return replace(prepared, report={**report, "status": "residual", "reason_codes": list(reasons),
            "plan_refill": _refill(prepared, reasons)})
    pins = tuple(DoctorPinnedExecutable(role=DoctorExecutableRole(role), executable_path=str(executable),
        executable_sha256="sha256:" + _sha(executable.read_bytes()),
        argv=(*command_prefix, role),
        verifier_id="verifier:" + role + (":lean-z3-imported-alias" if projection else ":lean-z3-keyword"), toolchain_id=toolchain, environment_id=roots.environment_id)
        for role in ("solver", "kernel"))
    state.mkdir(parents=True)
    hammer = DeterministicDoctorHammer(bounds=DoctorHammerBounds(wall_time_ms=60_000),
        authoritative_store=DoctorSealedReceiptStore(state / "proof.sqlite", authority_id="authority:" + operator),
        trusted_executable_pins=pins)
    adapter = DoctorWorktreeAdapter(repository_root=root, state_root=state / "worktrees",
        permitted_paths=(choice["path"],), permitted_refs=(candidate_ref,))
    current_ref = adapter._git(root, "symbolic-ref", "--quiet", "HEAD", check=False)
    if current_ref.returncode not in (0, 1):
        raise DoctorCompositionError("canonical checkout ref is unavailable")
    if current_ref.stdout.decode().strip() == candidate_ref:
        raise DoctorCompositionError("candidate ref must not be the canonical checkout ref")
    if adapter._git_text(root, "rev-parse", "--verify", candidate_ref + "^{commit}") != base:
        raise DoctorCompositionError("candidate ref differs from signed baseline")
    inputs = DoctorCompositionInputs(evidence_id=evidence.evidence_id, snapshot=snapshot, finding=finding,
        synthesis=synthesis, expected_after_hash=expected_hash, theorem=theorem, lowering=lowering,
        solver_pin=pins[0], kernel_pin=pins[1], hammer=hammer,
        impact=DoctorImpactRequest(roots=roots, subject_symbol_id=choice["subject"], change_set_id=consequence,
            before_contract_ref=finding.observed_fact_refs[0], after_contract_ref=expected, evidence_refs=(finding.content_id,)),
        program_graph=program_graph, source_hashes=ledger, proof_scope=proof_scope,
        worktree_adapter=adapter, target_ref=candidate_ref, base_ref=base, source_partition=partition, alias_contract=alias_contract)
    _assert_task_current(prepared)
    runtime.bind_composition(inputs)
    return replace(prepared, inputs=inputs, report={**report, "status": "prepared", "reason_codes": [],
        "selected_path": choice["path"], "candidate_ref": candidate_ref, "proof_scope": proof_scope,
        "operator_finding_id": finding.finding_id, "expected_contract_cid": expected})


def execute_doctor_task_repair(prepared: PreparedDoctorTaskRepair) -> dict:
    """Run the real gates and export only their independently checked candidate."""
    if type(prepared) is not PreparedDoctorTaskRepair:
        raise DoctorCompositionError("typed Doctor task preparation required")
    _assert_task_current(prepared)
    if prepared.inputs is None:
        return dict(prepared.report)
    runtime, inputs = prepared.runtime, prepared.inputs
    runtime.plan()
    composed = runtime.composition_result
    if composed is None:
        return {**dict(prepared.report), "status": "residual", "source_edits": 0,
            "plan_refill": _refill(prepared, ("native_doctor_planning_not_admitted",))}
    result = {**dict(prepared.report), "stages": dict(composed.stages), "source_edits": 0}
    if not composed.admitted:
        reasons = tuple(sorted({"native_doctor_gate_abstained", *(code for stage in composed.stages.values()
            for code in stage.get("reason_codes", ()))}))
        return {**result, "status": "residual", "plan_refill": _refill(prepared, reasons)}
    _assert_task_current(prepared)
    runtime.execute("repair", mode=DoctorMode.SANDBOX_AUTO, plan=composed.plan, exact_clean_target=True)
    transaction = runtime.composition_transaction
    if transaction is None:
        return {**result, "status": "residual", "plan_refill": _refill(prepared, ("native_doctor_transaction_not_admitted",))}
    result["transaction"] = transaction.to_dict()
    if not transaction.committed:
        return {**result, "status": "residual", "plan_refill": _refill(prepared, ("native_doctor_transaction_rejected",), transaction_id=transaction.transaction_id)}
    _assert_task_current(prepared)
    cas = transaction.merge_cas
    adapter = inputs.worktree_adapter
    if (cas is None or cas.ref_name != inputs.target_ref or cas.expected_ref != inputs.base_ref
            or adapter._git_text(runtime.checkout_root, "rev-parse", cas.ref_name) != cas.desired_ref):
        raise DoctorCompositionError("native Doctor candidate ref differs from its durable transaction")
    changed = adapter._git_text(runtime.checkout_root, "diff", "--name-only", inputs.base_ref, cas.desired_ref).splitlines()
    if changed != [inputs.synthesis.proposal.edit_site.path]:
        raise DoctorCompositionError("native Doctor candidate changes differ from the exact operator")
    path = changed[0]
    after = adapter._git(runtime.checkout_root, "show", cas.desired_ref + ":" + path).stdout
    if isinstance(after, str):
        after = after.encode()
    expected_after = operator_contract_reconstruction(inputs.synthesis, inputs.impact.subject_symbol_id,
                                                       alias_contract=inputs.alias_contract).encode()
    if after != expected_after:
        raise DoctorCompositionError("native Doctor candidate bytes differ from validated synthesis")
    handoff = {
        "schema": "supervisor-doctor-candidate-handoff@1", "operator": prepared.report["operator"],
        "task_cid": prepared.task["task_cid"], "task_id": prepared.task["task_alias"],
        "task_revision": prepared.task["revision"], "manifest_cid": prepared.report["manifest_cid"],
        "repository": str(runtime.checkout_root), "base_commit": inputs.base_ref,
        "candidate_commit": cas.desired_ref, "candidate_ref": cas.ref_name,
        "permitted_outputs": [row["path"] for row in prepared.task["body"][local.CONTRACT_KEY]["payload"]["task_spec"]["outputs"]],
        "transaction_id": transaction.transaction_id, "transaction_receipt_id": transaction.content_id,
        "proof_receipt_id": composed.proof.content_id, "proof_scope": inputs.proof_scope,
        "edits": [{"path": path, "before_sha256": inputs.source_hashes[path], "after_sha256": _sha(after),
                   "after_bytes_base64": base64.b64encode(after).decode()}],
        "provider_calls": 0, "publication_authority": False, "completion_authority": False,
    }
    handoff["handoff_cid"] = content_identity(handoff)
    artifact = prepared.state_root / "candidate-handoff.json"
    raw = json.dumps(handoff, sort_keys=True, indent=2).encode() + b"\n"
    with artifact.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    artifact.chmod(0o444)
    return {**result, "status": "candidate_ready", "source_edits": 1,
        "handoff_path": str(artifact), "handoff_sha256": _sha(raw), "handoff": handoff,
        "refresh_required_after_native_publication": True, "completion_authority": False}
