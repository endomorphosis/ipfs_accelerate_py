"""Fail-closed coverage for bounded program-repair synthesis (PDR-051)."""

from __future__ import annotations

import ast
import importlib
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.contract_repair_contracts import (
    AuthorityRoots,
    DecisionDisposition,
    EvidenceReference,
    RepairCandidate,
    RepairStrategy,
    RepairTargetDecision,
    SourceSpan,
    candidate_set_identity,
)
from ipfs_accelerate_py.agent_supervisor.analysis.deterministic_doctor_contracts import (
    DoctorAuthorityRoots,
    DoctorOperatorKind,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_synthesis import (
    DETERMINISTIC_DOCTOR_SYNTHESIZER_CAPABILITY_VERSION,
    DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE,
    DeterministicDoctorSynthesizer,
    DoctorSynthesisRequest,
    create_deterministic_doctor_synthesizer,
    materialize_proof_admitted_overlay,
)
from ipfs_accelerate_py.agent_supervisor.planning.deterministic_doctor_transforms import (
    DoctorOperatorProposal,
    DoctorRepairOperatorRegistry,
    build_default_doctor_operator_registry,
    make_edit_site,
)
from ipfs_accelerate_py.agent_supervisor.planning.program_repair_synthesis import (
    PROGRAM_REPAIR_SYNTHESIZER_INTERFACE,
    DeclaredEqualityTheory,
    EqualityRule,
    ProgramRepairAuthorityError,
    ProgramRepairBounds,
    ProgramRepairDisposition,
    ProgramRepairMode,
    ProgramRepairReason,
    ProgramRepairRequest,
    ProgramRepairSynthesizer,
    ProgramRepairSynthesisError,
    ResidualHybridDisposition,
    ResidualHybridPacket,
    ResidualHybridRepairService,
    create_program_repair_synthesizer,
    prove_equality_under_theory,
    synthesize_program_repair,
)
from ipfs_accelerate_py.agent_supervisor.planning.repair_operator_registry import (
    RepairOperatorKind,
    build_default_repair_operator_registry,
)
from ipfs_accelerate_py.agent_supervisor.proof.counterexample_guided_tactician import (
    CandidateKind,
    CandidateValidationStatus,
    CegisBudget,
    CegisStopReason,
    RefinementCandidate,
)
from ipfs_accelerate_py.agent_supervisor.proof.formal_counterexamples import (
    CounterexampleKind,
    RepairClass,
    normalize_counterexample,
)
from ipfs_accelerate_py.agent_supervisor.proof.missing_input_synthesis import (
    SynthesisDisposition,
    ValueMappingProof,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def roots(**overrides: str) -> DoctorAuthorityRoots:
    base = {
        "repository_id": "repository:fixture",
        "forest_id": "forest:fixture",
        "tree_id": "tree:fixture",
        "overlay_id": "overlay:fixture",
        "file_root_id": "file-root:fixture",
        "ast_root_id": "ast:fixture",
        "graph_id": "graph:fixture",
        "corpus_id": "corpus:fixture",
        "index_id": "index:fixture",
        "model_id": "model:fixture",
        "cache_id": "cache:fixture",
        "operator_registry_id": "operators:fixture",
        "translator_id": "translator:fixture",
        "solver_id": "solver:fixture",
        "kernel_id": "kernel:fixture",
        "toolchain_id": "toolchain:fixture",
        "policy_id": "policy:fixture",
        "sandbox_id": "sandbox:fixture",
        "environment_id": "environment:fixture",
        "lease_id": "lease:fixture",
    }
    base.update(overrides)
    return DoctorAuthorityRoots(**base)


def doctor_registry(auth: DoctorAuthorityRoots | None = None) -> DoctorRepairOperatorRegistry:
    return build_default_doctor_operator_registry(auth or roots())


def mapping(
    *,
    disposition: SynthesisDisposition = SynthesisDisposition.UNIQUE_PROVED,
    expression_ref: str = "expr:ctx",
    proved: tuple[str, ...] | None = None,
) -> ValueMappingProof:
    if proved is None:
        proved = (
            ("candidate:ctx",)
            if disposition is SynthesisDisposition.UNIQUE_PROVED
            else ()
        )
    return ValueMappingProof(
        requirement_id="missing:context",
        consumer_id="consumer:one",
        disposition=disposition,
        facet_results=(),
        proved_candidate_ids=proved,
        refuted_candidate_ids=(),
        expression_ref=expression_ref,
        type_ref="type:Context",
        repository_id="repository:fixture",
        tree_id="tree:fixture",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
        reason_codes=("unique_source",),
    )


def proof_receipt(
    *,
    admitted: bool = True,
    unique: bool = True,
    consequence: str = "consequence:unique-repair",
    llm_invocation_count: int = 0,
    model_provider_call_count: int = 0,
    write_authority: bool = False,
) -> dict[str, object]:
    eligible = (consequence,) if unique else (consequence, "consequence:other")
    return {
        "disposition": "admitted" if admitted else "abstained",
        "uniqueness_satisfied": unique and admitted,
        "selected_consequence_ref": consequence if admitted else "",
        "eligible_consequence_refs": list(eligible),
        "finding_id": "finding:one",
        "plan_receipt_id": "plan:one",
        "receipt_id": "proof:one",
        "llm_invocation_count": llm_invocation_count,
        "model_provider_call_count": model_provider_call_count,
        "write_authority": write_authority,
        "roots": {
            "repository_id": "repository:fixture",
            "tree_id": "tree:fixture",
        },
    }


def repair_decision(path: str = "pkg/caller.py") -> RepairTargetDecision:
    repair_roots = AuthorityRoots(
        repository_id="repository:fixture",
        forest_id="forest:fixture",
        tree_id="tree:fixture",
        graph_id="graph:fixture",
        index_id="index:fixture",
        model_id="model:fixture",
        config_id="config:fixture",
        translator_id="translator:fixture",
        toolchain_id="toolchain:fixture",
        policy_id="policy:fixture",
    )
    candidate = RepairCandidate(
        repair_roots,
        "trace:one",
        RepairStrategy.NEW_IMPLEMENTATION,
        SourceSpan(path, 0, 12, "blob:one"),
        (EvidenceReference("candidate", "candidate:one", producer_id="test"),),
    )
    candidates = (candidate,)
    return RepairTargetDecision(
        roots=repair_roots,
        candidates=candidates,
        candidate_set_id=candidate_set_identity(candidates),
        disposition=DecisionDisposition.ADMITTED,
        strategy=RepairStrategy.NEW_IMPLEMENTATION,
        selected_candidate_id=candidate.content_id,
        permitted_read_paths=(path,),
        permitted_write_paths=(path,),
        evidence_refs=(EvidenceReference("authority", "authority:one", producer_id="test"),),
        proof_refs=(EvidenceReference("proof", "proof:one", producer_id="test"),),
        invalidation_refs=("tree:fixture",),
    )


def propose_add_argument(
    reg: DoctorRepairOperatorRegistry,
    source: str = "process(event)",
    *,
    path: str = "pkg/caller.py",
    proof_admitted: bool = True,
) -> DoctorOperatorProposal:
    site = make_edit_site(path, source)
    return reg.propose(
        DoctorOperatorKind.ADD_ARGUMENT,
        site,
        obligation_refs=("obligation:one",),
        proof_refs=("proof:one",),
        value_source_refs=("value:ctx",),
        expression_ref="expr:ctx",
        parameter_name="context",
        proof_admitted=proof_admitted,
    )


def doctor_request(
    *,
    source: str = "process(event)",
    path: str = "pkg/caller.py",
    file_text: str | None = None,
) -> DoctorSynthesisRequest:
    auth = roots()
    reg = doctor_registry(auth)
    values: dict[str, object] = {
        "roots": auth,
        "proposal": propose_add_argument(reg, source, path=path),
        "span_text": source,
        "expression_text": "ctx",
        "value_mapping": mapping(),
        "decision": repair_decision(path=path),
        "proof_receipt": proof_receipt(),
        "selected_consequence_ref": "consequence:unique-repair",
        "value_ref": "value:ctx",
        "placement_ref": f"placement:{path}:0:{len(source)}",
        "finding_id": "finding:one",
        "plan_receipt_id": "plan:one",
        "proof_receipt_id": "proof:one",
        "require_proof_receipt": True,
    }
    if file_text is not None:
        values["file_text"] = file_text
    return DoctorSynthesisRequest(**values)  # type: ignore[arg-type]


def equality_theory() -> DeclaredEqualityTheory:
    return DeclaredEqualityTheory(
        theory_id="theory:arith@1",
        review_refs=("review:equality_theory@1", "review:equality_rewrite@1"),
        rules=(
            EqualityRule(
                rule_id="rule:add-zero",
                lhs="(+ x 0)",
                rhs="x",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:arith@1",
            ),
            EqualityRule(
                rule_id="rule:commute-add",
                lhs="(+ a b)",
                rhs="(+ b a)",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:arith@1",
            ),
        ),
        repository_id="repository:fixture",
        tree_id="tree:fixture",
    )


def counterexample():
    return normalize_counterexample(
        {
            "kind": CounterexampleKind.GENERIC_FAILURE.value,
            "failure": {"code": "repair-required"},
        },
        kind=CounterexampleKind.GENERIC_FAILURE,
        violated_property="obligation:one",
        bindings={
            "plan_id": "plan:base",
            "task_id": "PDR-051",
            "ast_scope_id": "symbol:target",
            "tree_id": "tree:fixture",
            "assumption_id": "assumption:dep",
            "provider_id": "tool:z3",
            "policy_id": "policy:fixture",
            "obligation_id": "obligation:one",
        },
        finite_bounds={"portfolio_width": 1, "deadline": 20},
        repair_classes=(RepairClass.ADD_DEPENDENCY,),
    )


# ---------------------------------------------------------------------------
# Interface / authority surface
# ---------------------------------------------------------------------------


def test_interface_identity_and_factory() -> None:
    auth = roots()
    synth = create_program_repair_synthesizer(auth)
    assert synth.INTERFACE == PROGRAM_REPAIR_SYNTHESIZER_INTERFACE
    assert PROGRAM_REPAIR_SYNTHESIZER_INTERFACE == "ProgramRepairSynthesizer@1"
    assert synth.registry.registry_id == build_default_repair_operator_registry().registry_id
    # Doctor surface stays frozen at @1 with PDR-051 capability revision 2.
    doctor = create_deterministic_doctor_synthesizer(auth)
    assert doctor.INTERFACE == DETERMINISTIC_DOCTOR_SYNTHESIZER_INTERFACE
    assert doctor.INTERFACE == "DeterministicDoctorSynthesizer@1"
    assert doctor.capability_version == DETERMINISTIC_DOCTOR_SYNTHESIZER_CAPABILITY_VERSION
    assert DETERMINISTIC_DOCTOR_SYNTHESIZER_CAPABILITY_VERSION == 2


def test_module_does_not_import_provider_or_llm_surfaces() -> None:
    source_path = Path(
        "ipfs_accelerate_py/agent_supervisor/planning/program_repair_synthesis.py"
    )
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    imported: list[str] = []
    for node in tree.body:
        if isinstance(node, ast.Import):
            imported.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported.append(module)
            imported.extend(
                f"{module}.{alias.name}" if module else alias.name
                for alias in node.names
            )
    forbidden = (
        "llm_router",
        "model_provider",
        "openai",
        "anthropic",
        "change_propagation_provider_router",
        "todo_daemon",
        "integrations",
    )
    joined = " ".join(imported)
    for marker in forbidden:
        assert marker not in joined
    before = set(sys.modules)
    importlib.import_module(
        "ipfs_accelerate_py.agent_supervisor.planning.program_repair_synthesis"
    )
    added = set(sys.modules) - before
    for name in added:
        lowered = name.lower()
        assert "llm_router" not in lowered
        assert "openai" not in lowered
        assert "anthropic" not in lowered


def test_bounds_hard_zero_model_calls() -> None:
    bounds = ProgramRepairBounds()
    assert bounds.max_model_calls == 0
    with pytest.raises(ProgramRepairAuthorityError):
        ProgramRepairBounds(max_model_calls=1)


# ---------------------------------------------------------------------------
# Deterministic doctor-backed synthesis (zero model calls, proposal-only)
# ---------------------------------------------------------------------------


def test_deterministic_doctor_request_produces_proposal_only_candidate() -> None:
    source = "process(event)"
    file_text = "def caller():\n    return process(event)\n"
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:one",),
        target_paths=("pkg/caller.py",),
        operator_kinds=(RepairOperatorKind.ADD_ARGUMENT.value,),
        placement_refs=("placement:exact",),
        value_refs=("value:ctx",),
        proof_refs=("proof:one",),
        mode=ProgramRepairMode.DETERMINISTIC,
        doctor_request=doctor_request(source=source, file_text=file_text),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.disposition is ProgramRepairDisposition.SUPPORTED
    assert receipt.admitted
    assert receipt.selected_candidate is not None
    assert receipt.selected_candidate.proposal_only is True
    assert receipt.selected_candidate.write_authority is False
    assert receipt.selected_candidate.semantic_authority is False
    assert receipt.llm_invocation_count == 0
    assert receipt.model_provider_call_count == 0
    assert receipt.provider_invoked is False
    assert receipt.deterministic_zero_model_calls is True
    assert receipt.write_performed is False
    assert ProgramRepairReason.ZERO_MODEL_CALLS.value in receipt.reason_codes
    assert ProgramRepairReason.PROPOSAL_ONLY.value in receipt.reason_codes
    assert receipt.doctor_receipt is not None
    assert DeterministicDoctorSynthesizer.prove_zero_model_calls(receipt.doctor_receipt)


def test_monkeypatched_llm_routes_remain_untouched_in_deterministic_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _boom(*_a: object, **_k: object) -> object:
        raise RuntimeError("llm route must never be called")

    fake_llm = types.ModuleType("ipfs_accelerate_py.agent_supervisor.llm_router")
    fake_llm.complete = _boom  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, fake_llm.__name__, fake_llm)

    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:one",),
        target_paths=("pkg/caller.py",),
        mode=ProgramRepairMode.DETERMINISTIC,
        doctor_request=doctor_request(),
    )
    receipt = create_program_repair_synthesizer(roots()).synthesize(request)
    assert receipt.admitted
    assert receipt.llm_invocation_count == 0
    with pytest.raises(RuntimeError, match="never be called"):
        fake_llm.complete("prompt")


# ---------------------------------------------------------------------------
# Reviewed operator grammar search
# ---------------------------------------------------------------------------


def test_search_only_reviewed_operators_under_exact_roots() -> None:
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:one",),
        target_paths=("pkg/module.py",),
        operator_kinds=(
            RepairOperatorKind.ADD_ARGUMENT.value,
            RepairOperatorKind.EQUALITY_REWRITE.value,
            "not_a_real_operator",
        ),
        placement_refs=("placement:exact",),
        value_refs=("value:unique",),
        proof_refs=("proof:nomination",),
        review_refs=(
            "review:equality_theory@1",
            "review:equality_rewrite@1",
        ),
        mode=ProgramRepairMode.ENUMERATIVE,
        bounds=ProgramRepairBounds(max_enumerative_candidates=4),
        equality_theory=equality_theory(),
        source_term="(+ x 0)",
        target_term="x",
    )
    receipt = synthesize_program_repair(request)
    # Equality rewrite is reviewed and proves under declared theory.
    assert any(
        c.operator_kind == RepairOperatorKind.EQUALITY_REWRITE.value
        for c in receipt.candidates
    )
    assert all(c.proposal_only for c in receipt.candidates)
    assert receipt.llm_invocation_count == 0
    # Unreviewed kinds never produce candidates.
    assert all(c.operator_kind != "not_a_real_operator" for c in receipt.candidates)


def test_scope_widening_multiple_target_paths_rejected() -> None:
    with pytest.raises(ProgramRepairAuthorityError):
        ProgramRepairRequest(
            roots=roots(),
            obligation_refs=("obligation:one",),
            target_paths=("a.py", "b.py"),
            mode=ProgramRepairMode.DETERMINISTIC,
        )


def test_path_escape_rejected() -> None:
    with pytest.raises(ProgramRepairAuthorityError):
        ProgramRepairRequest(
            roots=roots(),
            obligation_refs=("obligation:one",),
            target_paths=("../escape.py",),
            mode=ProgramRepairMode.DETERMINISTIC,
        )


# ---------------------------------------------------------------------------
# E-graph / equality rewrites under declared theory
# ---------------------------------------------------------------------------


def test_equality_rewrite_proves_under_declared_theory() -> None:
    theory = equality_theory()
    proved = prove_equality_under_theory(theory, "(+ x 0)", "x")
    assert proved.proved
    assert proved.proposal_only is True
    assert proved.grants_write_authority is False
    assert proved.grants_semantic_authority is False
    assert "rule:add-zero" in proved.applied_rule_ids

    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:eq",),
        target_paths=("pkg/expr.py",),
        mode=ProgramRepairMode.EQUALITY_REWRITE,
        equality_theory=theory,
        source_term="(+ x 0)",
        target_term="x",
        postcondition_refs=("post:equivalent_under_declared_theory",),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.disposition is ProgramRepairDisposition.SUPPORTED
    assert receipt.equality_receipt is not None
    assert receipt.equality_receipt.proved
    assert receipt.selected_candidate is not None
    assert receipt.selected_candidate.proposal_only
    assert receipt.deterministic_zero_model_calls


def test_equality_unproved_without_matching_rules() -> None:
    theory = DeclaredEqualityTheory(
        theory_id="theory:narrow@1",
        review_refs=("review:equality_theory@1",),
        rules=(
            EqualityRule(
                rule_id="rule:only-a",
                lhs="aaa",
                rhs="bbb",
                review_ref="review:equality_rewrite@1",
                theory_id="theory:narrow@1",
            ),
        ),
    )
    receipt = prove_equality_under_theory(theory, "unrelated", "other")
    assert not receipt.proved
    assert receipt.status.value in {"unproved", "budget_exhausted"}


def test_equality_requires_declared_theory() -> None:
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:eq",),
        target_paths=("pkg/expr.py",),
        mode=ProgramRepairMode.EQUALITY_REWRITE,
        source_term="a",
        target_term="b",
    )
    receipt = synthesize_program_repair(request)
    assert receipt.disposition is ProgramRepairDisposition.ABSTAIN
    assert ProgramRepairReason.UNDECLARED_THEORY.value in receipt.reason_codes


def test_equality_theory_cannot_grant_semantic_authority() -> None:
    with pytest.raises(ProgramRepairAuthorityError):
        DeclaredEqualityTheory(
            theory_id="theory:bad@1",
            review_refs=("review:equality_theory@1",),
            rules=(
                EqualityRule(
                    rule_id="rule:x",
                    lhs="a",
                    rhs="b",
                    review_ref="review:x",
                    theory_id="theory:bad@1",
                ),
            ),
            grants_semantic_authority=True,
        )


# ---------------------------------------------------------------------------
# CEGIS independent validation + fixed budgets
# ---------------------------------------------------------------------------


def test_cegis_closes_only_on_fresh_matching_receipt() -> None:
    cx = counterexample()

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        return {
            "receipt_id": "receipt:ok",
            "counterexample_id": binding["counterexample_id"],
            "repository_tree_id": binding["repository_tree_id"],
            "property_id": binding["property_id"],
            "assumption_ids": list(binding.get("assumption_ids") or ()),
            "bound_digest": binding["bound_digest"],
            "tool_id": binding["tool_id"],
            "policy_id": binding["policy_id"],
            "repaired_plan_id": binding["repaired_plan_id"],
            "freshness": "current",
            "outcome": "verified",
            "available": True,
        }

    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:one",),
        target_paths=("pkg/mod.py",),
        operator_kinds=(RepairOperatorKind.ADD_ARGUMENT.value,),
        mode=ProgramRepairMode.CEGIS,
        counterexample=cx,
        cegis_verify=verify,
        bounds=ProgramRepairBounds(max_cegis_iterations=2),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.disposition is ProgramRepairDisposition.SUPPORTED
    assert receipt.cegis_result is not None
    assert receipt.cegis_result.closed
    assert receipt.cegis_result.stop_reason is CegisStopReason.CLOSED
    assert receipt.deterministic_zero_model_calls
    assert receipt.selected_candidate is not None
    assert receipt.selected_candidate.proposal_only


def test_cegis_independent_validation_rejects_before_verifier() -> None:
    cx = counterexample()
    verify_calls = {"n": 0}

    def refine(witness, context):
        del witness, context
        return (
            RefinementCandidate(
                candidate_id="candidate:bad",
                kind=CandidateKind.REPAIR,
                goal_id="obligation:one",
                repaired_tree_id="tree:fixture",
                repaired_plan_id="plan:x",
                statement="does not address",
                addresses_witness=False,
            ),
        )

    def validate(candidate, context):
        del context
        if not candidate.addresses_witness:
            return (
                CandidateValidationStatus.INVALID,
                ProgramRepairReason.CEGIS_INDEPENDENT_REJECT.value,
            )
        return CandidateValidationStatus.VALID, "ok"

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        verify_calls["n"] += 1
        return {
            "receipt_id": "receipt:should-not-run",
            "counterexample_id": binding["counterexample_id"],
            "repository_tree_id": binding["repository_tree_id"],
            "property_id": binding["property_id"],
            "assumption_ids": list(binding.get("assumption_ids") or ()),
            "bound_digest": binding["bound_digest"],
            "tool_id": binding["tool_id"],
            "policy_id": binding["policy_id"],
            "repaired_plan_id": binding["repaired_plan_id"],
            "freshness": "current",
            "outcome": "verified",
            "available": True,
        }

    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:one",),
        target_paths=("pkg/mod.py",),
        mode=ProgramRepairMode.CEGIS,
        counterexample=cx,
        cegis_refine=refine,
        cegis_validate=validate,
        cegis_verify=verify,
        bounds=ProgramRepairBounds(max_cegis_iterations=1),
    )
    receipt = synthesize_program_repair(request)
    assert not receipt.admitted
    assert receipt.cegis_result is not None
    assert not receipt.cegis_result.closed
    assert verify_calls["n"] == 0


def test_cegis_terminates_on_fixed_budget() -> None:
    cx = counterexample()
    iterations = {"n": 0}

    def refine(witness, context):
        iterations["n"] += 1
        del witness, context
        return (
            RefinementCandidate(
                candidate_id=f"candidate:open:{iterations['n']}",
                kind=CandidateKind.REPAIR,
                goal_id="obligation:one",
                repaired_tree_id="tree:fixture",
                repaired_plan_id="plan:open",
                statement="still open",
                addresses_witness=True,
            ),
        )

    def verify(binding: dict[str, Any]) -> dict[str, Any]:
        return {
            "receipt_id": f"receipt:open:{binding.get('candidate_id')}",
            "counterexample_id": binding["counterexample_id"],
            "repository_tree_id": binding["repository_tree_id"],
            "property_id": binding["property_id"],
            "assumption_ids": list(binding.get("assumption_ids") or ()),
            "bound_digest": binding["bound_digest"],
            "tool_id": binding["tool_id"],
            "policy_id": binding["policy_id"],
            "repaired_plan_id": binding["repaired_plan_id"],
            "freshness": "current",
            "outcome": "still_violated",
            "available": True,
        }

    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:one",),
        target_paths=("pkg/mod.py",),
        mode=ProgramRepairMode.CEGIS,
        counterexample=cx,
        cegis_refine=refine,
        cegis_verify=verify,
        bounds=ProgramRepairBounds(
            max_cegis_iterations=2,
            max_identical_failures=8,
        ),
    )
    receipt = synthesize_program_repair(request)
    assert receipt.cegis_result is not None
    assert not receipt.cegis_result.closed
    assert receipt.cegis_result.iteration_count <= 2
    assert receipt.llm_invocation_count == 0


# ---------------------------------------------------------------------------
# Residual-only hybrid service
# ---------------------------------------------------------------------------


def test_residual_packet_emitted_for_behavior_fixed_syntax_debt() -> None:
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:syntax",),
        target_paths=("pkg/syntax.py",),
        postcondition_refs=("post:behavior-fixed",),
        test_refs=("test:unit:syntax",),
        mode=ProgramRepairMode.DETERMINISTIC,
        behavior_fixed_syntax_debt=True,
        allow_hybrid_residual=True,
        syntax_slot_id="syntax:fn-body",
        span_text="pass",
    )
    receipt = synthesize_program_repair(request)
    assert receipt.disposition is ProgramRepairDisposition.RESIDUAL_DEBT
    assert receipt.residual_packet is not None
    packet = receipt.residual_packet
    assert packet.target_path == "pkg/syntax.py"
    assert packet.behavior_fixed is True
    assert packet.may_change_authority is False
    assert packet.may_change_dependencies is False
    assert packet.may_change_meaning is False
    assert packet.may_add_imports is False
    assert packet.may_add_files is False
    assert packet.postcondition_refs == ("post:behavior-fixed",)
    assert packet.test_refs == ("test:unit:syntax",)
    assert packet.semantics_digest
    # Deterministic orchestrator still proves zero model calls.
    assert receipt.llm_invocation_count == 0
    assert receipt.deterministic_zero_model_calls


def test_hybrid_service_admits_exact_syntax_only() -> None:
    packet = ResidualHybridPacket(
        packet_id="residual:test",
        target_path="pkg/syntax.py",
        span_start=0,
        span_end=4,
        semantics_digest="sha256:semantics",
        postcondition_refs=("post:behavior-fixed",),
        test_refs=("test:unit:syntax",),
        obligation_refs=("obligation:syntax",),
        repository_id="repository:fixture",
        tree_id="tree:fixture",
    )
    service = ResidualHybridRepairService(
        bounds=ProgramRepairBounds(max_hybrid_calls=2, max_hybrid_tokens=1000)
    )
    admission = service.admit(
        packet,
        {"syntax": "return 0", "path": "pkg/syntax.py", "semantics_digest": "sha256:semantics"},
        response_tokens=10,
        model_calls=1,
    )
    assert admission.admitted
    assert admission.syntax == "return 0"
    assert admission.proposal_only is True
    assert admission.usage is not None
    assert admission.usage.write_authority is False
    assert admission.usage.dependency_change is False
    assert admission.usage.meaning_change is False


def test_hybrid_rejects_extra_file_import_dependency_and_scope_widening() -> None:
    packet = ResidualHybridPacket(
        packet_id="residual:test",
        target_path="pkg/syntax.py",
        span_start=0,
        span_end=4,
        semantics_digest="sha256:semantics",
        postcondition_refs=("post:behavior-fixed",),
        test_refs=("test:unit:syntax",),
        obligation_refs=("obligation:syntax",),
        repository_id="repository:fixture",
        tree_id="tree:fixture",
    )
    service = ResidualHybridRepairService()

    extra_file = service.admit(
        packet,
        {
            "syntax": "x = 1",
            "path": "pkg/syntax.py",
            "extra_paths": ("pkg/other.py",),
            "semantics_digest": "sha256:semantics",
        },
    )
    assert not extra_file.admitted
    assert any("extra_file" in r or "scope" in r for r in extra_file.reason_codes)

    extra_import = service.admit(
        packet,
        {
            "syntax": "import os",
            "path": "pkg/syntax.py",
            "extra_imports": ("os",),
            "semantics_digest": "sha256:semantics",
        },
    )
    assert not extra_import.admitted

    dependency = service.admit(
        packet,
        {
            "syntax": "x",
            "path": "pkg/syntax.py",
            "new_dependencies": ("requests",),
            "semantics_digest": "sha256:semantics",
        },
    )
    assert not dependency.admitted

    scope = service.admit(
        packet,
        {
            "syntax": "x",
            "path": "pkg/other.py",  # different path
            "semantics_digest": "sha256:semantics",
        },
    )
    assert not scope.admitted

    meaning = service.admit(
        packet,
        {
            "syntax": "x",
            "path": "pkg/syntax.py",
            "semantics_digest": "sha256:different-meaning",
        },
    )
    assert not meaning.admitted

    authority = service.admit(
        packet,
        {
            "syntax": "x",
            "path": "pkg/syntax.py",
            "write_authority": True,
            "semantics_digest": "sha256:semantics",
        },
    )
    assert not authority.admitted


def test_hybrid_packet_forbids_authority_dependency_meaning_flags() -> None:
    with pytest.raises(ProgramRepairAuthorityError):
        ResidualHybridPacket(
            packet_id="residual:bad",
            target_path="pkg/syntax.py",
            span_start=0,
            span_end=1,
            semantics_digest="sha256:s",
            postcondition_refs=(),
            test_refs=(),
            obligation_refs=("o",),
            repository_id="repository:fixture",
            tree_id="tree:fixture",
            may_change_dependencies=True,
        )


def test_malformed_hybrid_proposal_fails() -> None:
    packet = ResidualHybridPacket(
        packet_id="residual:test",
        target_path="pkg/syntax.py",
        span_start=0,
        span_end=1,
        semantics_digest="sha256:s",
        postcondition_refs=(),
        test_refs=(),
        obligation_refs=("o",),
        repository_id="repository:fixture",
        tree_id="tree:fixture",
    )
    service = ResidualHybridRepairService()
    with pytest.raises(ProgramRepairSynthesisError):
        service.admit(packet, "not-json{")


# ---------------------------------------------------------------------------
# Non-idempotent / proposal authority
# ---------------------------------------------------------------------------


def test_candidate_cannot_claim_write_or_model_calls() -> None:
    from ipfs_accelerate_py.agent_supervisor.planning.program_repair_synthesis import (
        ProgramRepairCandidate,
    )

    with pytest.raises(ProgramRepairAuthorityError):
        ProgramRepairCandidate(
            candidate_id="c1",
            operator_kind="add_argument",
            operator_id="op",
            path="pkg/a.py",
            mode=ProgramRepairMode.DETERMINISTIC,
            write_authority=True,
        )
    with pytest.raises(ProgramRepairAuthorityError):
        ProgramRepairCandidate(
            candidate_id="c2",
            operator_kind="add_argument",
            operator_id="op",
            path="pkg/a.py",
            mode=ProgramRepairMode.DETERMINISTIC,
            llm_invocation_count=1,
        )


def test_receipt_round_trip_and_exports() -> None:
    theory = equality_theory()
    request = ProgramRepairRequest(
        roots=roots(),
        obligation_refs=("obligation:eq",),
        target_paths=("pkg/expr.py",),
        mode=ProgramRepairMode.EQUALITY_REWRITE,
        equality_theory=theory,
        source_term="(+ x 0)",
        target_term="x",
    )
    receipt = synthesize_program_repair(request)
    payload = receipt.to_dict()
    assert payload["interface"] == PROGRAM_REPAIR_SYNTHESIZER_INTERFACE
    assert payload["proposal_only"] is True
    assert payload["llm_invocation_count"] == 0
    assert payload["deterministic_zero_model_calls"] is True
    assert "schema" in payload


def test_doctor_capability_version_and_zero_model_proof() -> None:
    receipt = materialize_proof_admitted_overlay(
        doctor_request(), registry=doctor_registry()
    )
    assert DeterministicDoctorSynthesizer.prove_zero_model_calls(receipt)
    assert receipt.llm_invocation_count == 0
