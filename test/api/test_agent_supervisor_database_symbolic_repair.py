"""Tests for DatabaseSymbolicPlanner@1 and RepairLineage@1 (DQP-024).

Evidence subset: candidate reuse, stale AST, counterexample, partial plan,
unsupported operator, fixed point, proof invalidation, abstention.

Acceptance:

* LLM cannot invent scope/semantics outside admitted plan
* Stale or incomplete impact prevents write
* Proof cache hits rederive applicability
* All accepted repairs reach code-and-logic fixed point or roll back
* Unsupported classes require approval/abstain
"""

from __future__ import annotations

from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.database_impact_graph import (
    DEFAULT_POLICY_ID,
    EdgeKind,
    ImpactCompleteness,
    ImpactEdgeSpec,
    ImpactSymbolSpec,
    duckdb_available as impact_duckdb_available,
    open_database_impact_graph,
)
from ipfs_accelerate_py.agent_supervisor.analysis.mutation_ledger import (
    MutationContext,
    MutationFileSpec,
    MutationStatus,
    duckdb_available as mutation_duckdb_available,
    open_mutation_ledger,
)
from ipfs_accelerate_py.agent_supervisor.planning.database_symbolic_planning import (
    AUTHORITY_CLASS as PLAN_AUTHORITY,
    DATABASE_SYMBOLIC_PLANNER_INTERFACE,
    LLM_OUTPUT_POLICY,
    PLAN_AUTHORITY_POLICY,
    CandidateDisposition,
    DatabaseSymbolicPlanner,
    DiscoveryStage,
    LLMProposal,
    PlanDisposition,
    PlanRejectReason,
    SymbolicCandidateSpec,
    SymbolicPlanBindings,
    SymbolicPlanRequest,
    duckdb_available as planner_duckdb_available,
    open_database_symbolic_planner,
    operator_is_supported,
    operator_requires_approval,
)
from ipfs_accelerate_py.agent_supervisor.proof.database_repair_evidence import (
    AUTHORITY_CLASS as REPAIR_AUTHORITY,
    CACHE_ASSURANCE_POLICY,
    REPAIR_LINEAGE_INTERFACE,
    WRITE_AUTHORITY_POLICY,
    AssuranceLevel,
    DatabaseRepairEvidenceStore,
    FixedPointEvidence,
    FixedPointStatus,
    ProofCacheLookupStatus,
    ProofCacheRejectReason,
    RepairAttemptRequest,
    RepairDisposition,
    RepairProofCacheEntry,
    RepairProofCacheKey,
    RepairRejectReason,
    duckdb_available as repair_duckdb_available,
    open_database_repair_evidence_store,
)


pytestmark = pytest.mark.skipif(
    not (
        planner_duckdb_available()
        and repair_duckdb_available()
        and impact_duckdb_available()
        and mutation_duckdb_available()
    ),
    reason="DuckDB is required for DQP-024 hermetic symbolic-repair tests",
)


PYTHON_V1 = """\
class Service:
    def dispatch(self, request):
        return request
"""

PYTHON_V2 = """\
class Service:
    def dispatch(self, request):
        self.status = "running"
        return request
"""


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _open_planner(tmp_path: Path) -> DatabaseSymbolicPlanner:
    return open_database_symbolic_planner(tmp_path / "symbolic_planner.duckdb")


def _open_repair(tmp_path: Path) -> DatabaseRepairEvidenceStore:
    return open_database_repair_evidence_store(tmp_path / "repair_evidence.duckdb")


def _materialize_impact(tmp_path: Path, *, with_dynamic: bool = False):
    graph = open_database_impact_graph(tmp_path / "impact.duckdb")
    graph.open()
    edges = [
        ImpactEdgeSpec(
            source_symbol="consume",
            target_symbol="Service.dispatch",
            edge_kind=EdgeKind.CALLS,
            path="src/consumer.py",
        ),
        ImpactEdgeSpec(
            source_symbol="test_consume",
            target_symbol="consume",
            edge_kind=EdgeKind.TESTS,
            path="test/test_consumer.py",
        ),
        ImpactEdgeSpec(
            source_symbol="proof_dispatch_total",
            target_symbol="Service.dispatch",
            edge_kind=EdgeKind.PROOFS,
            path="proofs/dispatch.lean",
        ),
    ]
    if with_dynamic:
        edges.append(
            ImpactEdgeSpec(
                source_symbol="dispatch_dynamic",
                target_symbol="<dynamic>",
                edge_kind=EdgeKind.DYNAMIC,
                path="src/dyn.py",
                is_dynamic=True,
                reason="getattr_dispatch",
            )
        )
    graph.materialize(
        snapshot_id="snapshot:demo-1",
        edges=edges,
        symbols=[
            ImpactSymbolSpec(
                "Service.dispatch", path="src/service.py", language="python"
            ),
            ImpactSymbolSpec("consume", path="src/consumer.py", language="python"),
            ImpactSymbolSpec(
                "test_consume", path="test/test_consumer.py", language="python"
            ),
            ImpactSymbolSpec(
                "proof_dispatch_total",
                path="proofs/dispatch.lean",
                language="lean",
            ),
        ],
        parser_id="python-ast@test",
        policy_id=DEFAULT_POLICY_ID,
        repository_id="repo:demo",
        tree_id="tree:abc",
    )
    return graph


def _record_mutation(tmp_path: Path):
    ledger = open_mutation_ledger(tmp_path / "mutation.duckdb")
    fence = ledger.register_fence(
        worktree_id="worktree:wt-1",
        token="fence-token-alpha",
        before_snapshot_id="snapshot:demo-1",
        before_tree_id="tree:abc",
        lease_id="lease:1",
        session_id="session:1",
    )
    result = ledger.record_mutation(
        MutationContext(
            task_id="task:DQP-024",
            attempt_id="attempt:1",
            plan_id="plan:seed",
            operator_id="operator:daemon",
            provider_id="provider:test",
            daemon_id="daemon:impl-1",
            session_id=fence.session_id or "session:1",
            worktree_id=fence.worktree_id,
            lease_id=fence.lease_id or "lease:1",
            fence_id=fence.fence_id,
            before_snapshot_id=fence.before_snapshot_id or "snapshot:demo-1",
            after_snapshot_id="snapshot:after-1",
            before_tree_id=fence.before_tree_id or "tree:abc",
            after_tree_id="tree:after-1",
            repository_id="repo:demo",
            declared_effects={"symbols": ["Service.dispatch"]},
        ),
        [
            MutationFileSpec(
                path="src/service.py",
                before_content=PYTHON_V1,
                after_content=PYTHON_V2,
            )
        ],
    )
    return ledger, result


def _fresh_bindings(**overrides) -> SymbolicPlanBindings:
    base = {
        "snapshot_id": "snapshot:demo-1",
        "tree_id": "tree:abc",
        "repository_id": "repo:demo",
        "seed_symbols": ("Service.dispatch",),
        "symbol_ids": ("Service.dispatch", "consume"),
        "ast_mutation_ids": ("ast:mutation:service-dispatch",),
        "admitted_write_paths": ("src/service.py",),
        "admitted_read_paths": ("src/service.py", "src/consumer.py"),
        "current_ast_digest": "sha256:" + ("ab" * 32),
        "expected_ast_digest": "sha256:" + ("ab" * 32),
        "parser_id": "python-ast@test",
        "policy_id": DEFAULT_POLICY_ID,
    }
    base.update(overrides)
    return SymbolicPlanBindings(**base)


def _admitted_candidate(**overrides) -> SymbolicCandidateSpec:
    base = {
        "operator_class": "constraint_rewrite",
        "write_paths": ("src/service.py",),
        "symbol_ids": ("Service.dispatch",),
        "ast_mutation_ids": ("ast:mutation:service-dispatch",),
        "obligation_ids": ("obl:dispatch-total",),
        "source": "deterministic",
        "body": {"rewrite": "align_status_field"},
    }
    base.update(overrides)
    return SymbolicCandidateSpec(**base)


def _plan_with_complete_impact(tmp_path: Path, planner, **request_overrides):
    graph = _materialize_impact(tmp_path)
    try:
        closure = graph.impact_closure(["Service.dispatch"])
        assert closure.blocks_automatic_repair is False
        assert closure.completeness is ImpactCompleteness.COMPLETE

        ledger, mutation_result = _record_mutation(tmp_path)
        try:
            assert mutation_result.admitted is True
            ast_ids = tuple(
                item.ast_mutation_id
                for item in ledger.list_ast_mutations(
                    mutation_result.mutation.mutation_id
                )
            )
            bindings = _fresh_bindings(
                mutation_id=mutation_result.mutation.mutation_id,
                impact_query_id=closure.query_id,
                impact_revision_id=closure.revision_id,
                schema_id=closure.schema_id,
                ast_mutation_ids=ast_ids
                or ("ast:mutation:service-dispatch",),
            )
            request_kwargs = {
                "task_id": "task:DQP-024",
                "attempt_id": "attempt:1",
                "bindings": bindings,
                "seed_symbols": ("Service.dispatch",),
                "candidates": (_admitted_candidate(ast_mutation_ids=ast_ids or ("ast:mutation:service-dispatch",)),),
            }
            request_kwargs.update(request_overrides)
            # If caller replaced bindings, keep them.
            if "bindings" in request_overrides:
                request_kwargs["bindings"] = request_overrides["bindings"]
            receipt = planner.plan(
                SymbolicPlanRequest(**request_kwargs),
                impact_closure=closure,
                mutation=mutation_result.mutation,
            )
            return receipt, closure, mutation_result
        finally:
            ledger.close()
    finally:
        graph.close()


# ---------------------------------------------------------------------------
# Interface identities / cold import
# ---------------------------------------------------------------------------


def test_interface_identities() -> None:
    assert DATABASE_SYMBOLIC_PLANNER_INTERFACE == "DatabaseSymbolicPlanner@1"
    assert REPAIR_LINEAGE_INTERFACE == "RepairLineage@1"
    assert DatabaseSymbolicPlanner.INTERFACE == DATABASE_SYMBOLIC_PLANNER_INTERFACE
    assert DatabaseRepairEvidenceStore.INTERFACE == REPAIR_LINEAGE_INTERFACE
    assert PLAN_AUTHORITY == "derived_evidence"
    assert REPAIR_AUTHORITY == "derived_evidence"
    assert PLAN_AUTHORITY_POLICY == "no_write_authority"
    assert LLM_OUTPUT_POLICY == "nomination_only"
    assert CACHE_ASSURANCE_POLICY == "never_promote_assurance"
    assert WRITE_AUTHORITY_POLICY == "plan_write_admitted_only"
    assert operator_is_supported("constraint_rewrite")
    assert operator_requires_approval("llm_freeform")
    assert not operator_is_supported("llm_freeform")


def test_cold_import_and_construction_have_no_side_effects() -> None:
    planner = DatabaseSymbolicPlanner("/tmp/should-not-exist-until-open.duckdb")
    repair = DatabaseRepairEvidenceStore(
        "/tmp/should-not-exist-until-open-repair.duckdb"
    )
    assert planner.is_open is False
    assert repair.is_open is False


# ---------------------------------------------------------------------------
# Happy path: discovery precedes synthesis; plan binds exact IDs
# ---------------------------------------------------------------------------


def test_deterministic_discovery_precedes_synthesis_and_binds_ids(
    tmp_path: Path,
) -> None:
    with _open_planner(tmp_path) as planner:
        receipt, closure, mutation_result = _plan_with_complete_impact(
            tmp_path, planner
        )
        plan = receipt.plan
        assert plan.interface == DATABASE_SYMBOLIC_PLANNER_INTERFACE
        assert plan.disposition is PlanDisposition.ADMITTED
        assert plan.write_admitted is True
        assert plan.impact_complete is True
        assert plan.blocks_automatic_repair is False

        # Discovery stages are ordered and complete.
        stages = list(plan.discovery_stages)
        assert stages[0] == DiscoveryStage.BINDINGS.value
        assert DiscoveryStage.MUTATION_DISCOVERY.value in stages
        assert DiscoveryStage.IMPACT_QUERY.value in stages
        assert DiscoveryStage.AST_FRESHNESS.value in stages
        assert DiscoveryStage.CANDIDATE_SYNTHESIS.value in stages
        assert stages[-1] == DiscoveryStage.ADMISSION.value
        assert stages.index(DiscoveryStage.IMPACT_QUERY.value) < stages.index(
            DiscoveryStage.CANDIDATE_SYNTHESIS.value
        )

        # Exact AST / symbol / mutation identities.
        assert plan.bindings.mutation_id == mutation_result.mutation.mutation_id
        assert plan.bindings.impact_query_id == closure.query_id
        assert plan.bindings.snapshot_id == "snapshot:demo-1"
        assert plan.bindings.ast_mutation_ids
        assert "Service.dispatch" in plan.bindings.symbol_ids
        assert plan.bindings.ast_is_fresh is True

        # Candidate admitted with exact path/symbol bindings.
        assert len(plan.candidates) == 1
        candidate = plan.candidates[0]
        assert candidate.disposition in {
            CandidateDisposition.ADMITTED,
            CandidateDisposition.REUSED,
        }
        assert candidate.write_paths == ("src/service.py",)
        assert "Service.dispatch" in candidate.symbol_ids

        # Durable load.
        loaded = planner.get_plan(plan.plan_id)
        assert loaded is not None
        assert loaded.plan_digest == plan.plan_digest
        assert loaded.write_admitted is True
        meta = planner.metadata()
        assert meta["interface"] == DATABASE_SYMBOLIC_PLANNER_INTERFACE
        assert meta["plan_authority_policy"] == PLAN_AUTHORITY_POLICY


# ---------------------------------------------------------------------------
# LLM cannot invent scope / semantics
# ---------------------------------------------------------------------------


def test_llm_cannot_invent_scope_outside_admitted_plan(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner:
        graph = _materialize_impact(tmp_path)
        try:
            closure = graph.impact_closure(["Service.dispatch"])
            bindings = _fresh_bindings(
                impact_query_id=closure.query_id,
                impact_revision_id=closure.revision_id,
                schema_id=closure.schema_id,
            )
            # LLM invents a write path outside admitted scope.
            proposal = LLMProposal(
                proposed_write_paths=("src/unrelated.py",),
                proposed_symbols=("Service.dispatch",),
                proposed_operator_class="enumerative",
                claims={"admitted": True, "valid": True},
            )
            receipt = planner.plan(
                SymbolicPlanRequest(
                    task_id="task:llm-scope",
                    bindings=bindings,
                    seed_symbols=("Service.dispatch",),
                    candidates=(_admitted_candidate(),),
                    llm_proposal=proposal,
                ),
                impact_closure=closure,
            )
            plan = receipt.plan
            assert plan.disposition is PlanDisposition.REJECTED
            assert plan.write_admitted is False
            assert PlanRejectReason.SCOPE_INVENTION.value in plan.reasons
            assert PlanRejectReason.PROVIDER_CLAIM.value in plan.reasons
            assert plan.llm_audit["policy"] == LLM_OUTPUT_POLICY
            assert plan.llm_audit["accepted_as_nomination"] is False
            assert plan.llm_audit["scope_invention"] is True
        finally:
            graph.close()


def test_llm_semantic_invention_is_rejected(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner:
        graph = _materialize_impact(tmp_path)
        try:
            closure = graph.impact_closure(["Service.dispatch"])
            bindings = _fresh_bindings(
                impact_query_id=closure.query_id,
                impact_revision_id=closure.revision_id,
                schema_id=closure.schema_id,
            )
            proposal = LLMProposal(
                proposed_write_paths=("src/service.py",),
                proposed_symbols=("Service.dispatch",),
                proposed_operator_class="constraint_rewrite",
                claims={
                    "semantic_change": True,
                    "meaning_change": True,
                    "completion_claim": True,
                },
            )
            receipt = planner.plan(
                SymbolicPlanRequest(
                    task_id="task:llm-semantic",
                    bindings=bindings,
                    seed_symbols=("Service.dispatch",),
                    candidates=(_admitted_candidate(),),
                    llm_proposal=proposal,
                ),
                impact_closure=closure,
            )
            plan = receipt.plan
            assert plan.disposition is PlanDisposition.REJECTED
            assert plan.write_admitted is False
            assert PlanRejectReason.SEMANTIC_INVENTION.value in plan.reasons
            assert PlanRejectReason.PROVIDER_CLAIM.value in plan.reasons
        finally:
            graph.close()


def test_llm_in_scope_nomination_is_accepted_as_nomination_only(
    tmp_path: Path,
) -> None:
    with _open_planner(tmp_path) as planner:
        graph = _materialize_impact(tmp_path)
        try:
            closure = graph.impact_closure(["Service.dispatch"])
            bindings = _fresh_bindings(
                impact_query_id=closure.query_id,
                impact_revision_id=closure.revision_id,
                schema_id=closure.schema_id,
            )
            proposal = LLMProposal(
                proposed_write_paths=("src/service.py",),
                proposed_symbols=("Service.dispatch",),
                proposed_ast_ids=("ast:mutation:service-dispatch",),
                proposed_operator_class="enumerative",
            )
            receipt = planner.plan(
                SymbolicPlanRequest(
                    task_id="task:llm-ok",
                    bindings=bindings,
                    seed_symbols=("Service.dispatch",),
                    candidates=(),
                    llm_proposal=proposal,
                ),
                impact_closure=closure,
            )
            plan = receipt.plan
            assert plan.llm_audit["accepted_as_nomination"] is True
            assert plan.disposition is PlanDisposition.ADMITTED
            assert plan.write_admitted is True
            assert any(c.source == "llm_residual" for c in plan.candidates)
            # Residual body remains nomination-only.
            residual = next(c for c in plan.candidates if c.source == "llm_residual")
            assert residual.body.get("nomination_only") is True
        finally:
            graph.close()


# ---------------------------------------------------------------------------
# Stale AST / incomplete impact prevent write
# ---------------------------------------------------------------------------


def test_stale_ast_prevents_write(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner:
        graph = _materialize_impact(tmp_path)
        try:
            closure = graph.impact_closure(["Service.dispatch"])
            bindings = _fresh_bindings(
                impact_query_id=closure.query_id,
                impact_revision_id=closure.revision_id,
                schema_id=closure.schema_id,
                current_ast_digest="sha256:" + ("11" * 32),
                expected_ast_digest="sha256:" + ("22" * 32),
            )
            assert bindings.ast_is_fresh is False
            receipt = planner.plan(
                SymbolicPlanRequest(
                    task_id="task:stale-ast",
                    bindings=bindings,
                    seed_symbols=("Service.dispatch",),
                    candidates=(_admitted_candidate(),),
                ),
                impact_closure=closure,
            )
            plan = receipt.plan
            assert plan.disposition is PlanDisposition.ABSTAINED
            assert plan.write_admitted is False
            assert PlanRejectReason.STALE_AST.value in plan.reasons
        finally:
            graph.close()


def test_incomplete_impact_and_blocking_frontier_prevent_write(
    tmp_path: Path,
) -> None:
    with _open_planner(tmp_path) as planner:
        graph = _materialize_impact(tmp_path, with_dynamic=True)
        try:
            closure = graph.impact_closure(["Service.dispatch"])
            assert closure.blocks_automatic_repair is True
            assert closure.completeness is not ImpactCompleteness.COMPLETE

            bindings = _fresh_bindings(
                impact_query_id=closure.query_id,
                impact_revision_id=closure.revision_id,
                schema_id=closure.schema_id,
            )
            receipt = planner.plan(
                SymbolicPlanRequest(
                    task_id="task:blocking-impact",
                    bindings=bindings,
                    seed_symbols=("Service.dispatch",),
                    candidates=(_admitted_candidate(),),
                ),
                impact_closure=closure,
            )
            plan = receipt.plan
            assert plan.write_admitted is False
            assert plan.disposition is PlanDisposition.ABSTAINED
            assert plan.blocks_automatic_repair is True
            assert PlanRejectReason.BLOCKING_FRONTIER.value in plan.reasons
            assert PlanRejectReason.INCOMPLETE_IMPACT.value in plan.reasons

            # Repair store also refuses write on non-write-admitted plan.
            with _open_repair(tmp_path) as repair:
                lineage = repair.apply_repair(
                    RepairAttemptRequest(
                        plan=plan,
                        operator_class="constraint_rewrite",
                        write_paths=("src/service.py",),
                        fixed_point=FixedPointEvidence(
                            code_fixed=True,
                            logic_fixed=True,
                        ),
                    )
                )
                assert lineage.disposition is RepairDisposition.ABSTAINED
                assert lineage.write_committed is False
                assert (
                    RepairRejectReason.PLAN_WRITE_NOT_ADMITTED.value
                    in lineage.reasons
                )
        finally:
            graph.close()


def test_missing_impact_prevents_write(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner:
        bindings = _fresh_bindings()
        receipt = planner.plan(
            SymbolicPlanRequest(
                task_id="task:no-impact",
                bindings=bindings,
                seed_symbols=("Service.dispatch",),
                candidates=(_admitted_candidate(),),
            ),
            impact_closure=None,
        )
        plan = receipt.plan
        assert plan.write_admitted is False
        assert plan.disposition is PlanDisposition.ABSTAINED
        assert PlanRejectReason.MISSING_IMPACT.value in plan.reasons


# ---------------------------------------------------------------------------
# Candidate reuse (exact identity)
# ---------------------------------------------------------------------------


def test_candidate_reuse_is_exact_identity(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner:
        candidate = _admitted_candidate()
        planner.register_reusable_candidate(candidate)

        graph = _materialize_impact(tmp_path)
        try:
            closure = graph.impact_closure(["Service.dispatch"])
            bindings = _fresh_bindings(
                impact_query_id=closure.query_id,
                impact_revision_id=closure.revision_id,
                schema_id=closure.schema_id,
            )
            # Exact same candidate body → reused.
            receipt = planner.plan(
                SymbolicPlanRequest(
                    task_id="task:reuse",
                    bindings=bindings,
                    seed_symbols=("Service.dispatch",),
                    candidates=(candidate,),
                ),
                impact_closure=closure,
            )
            plan = receipt.plan
            assert plan.candidates[0].disposition is CandidateDisposition.REUSED
            assert plan.candidates[0].reused is True
            assert plan.write_admitted is True

            # Same id, different body → reuse mismatch reject.
            mutated = _admitted_candidate(
                candidate_id=candidate.candidate_id,
                body={"rewrite": "different"},
            )
            receipt2 = planner.plan(
                SymbolicPlanRequest(
                    task_id="task:reuse-mismatch",
                    bindings=bindings,
                    seed_symbols=("Service.dispatch",),
                    candidates=(mutated,),
                ),
                impact_closure=closure,
            )
            assert (
                receipt2.plan.candidates[0].disposition
                is CandidateDisposition.REJECTED
            )
            assert (
                PlanRejectReason.CANDIDATE_REUSE_MISMATCH.value
                in receipt2.plan.candidates[0].reason
            )
        finally:
            graph.close()


# ---------------------------------------------------------------------------
# Unsupported operator / approval / counterexample / partial
# ---------------------------------------------------------------------------


def test_unsupported_operator_requires_approval_or_abstain(
    tmp_path: Path,
) -> None:
    with _open_planner(tmp_path) as planner:
        graph = _materialize_impact(tmp_path)
        try:
            closure = graph.impact_closure(["Service.dispatch"])
            bindings = _fresh_bindings(
                impact_query_id=closure.query_id,
                impact_revision_id=closure.revision_id,
                schema_id=closure.schema_id,
            )
            unsupported = _admitted_candidate(operator_class="llm_freeform")
            receipt = planner.plan(
                SymbolicPlanRequest(
                    task_id="task:unsupported",
                    bindings=bindings,
                    seed_symbols=("Service.dispatch",),
                    candidates=(unsupported,),
                    approval_granted=False,
                ),
                impact_closure=closure,
            )
            plan = receipt.plan
            assert plan.disposition is PlanDisposition.REQUIRES_APPROVAL
            assert plan.write_admitted is False
            assert plan.candidates[0].disposition in {
                CandidateDisposition.REQUIRES_APPROVAL,
                CandidateDisposition.UNSUPPORTED,
            }

            # With approval, operator class can proceed if otherwise valid.
            receipt2 = planner.plan(
                SymbolicPlanRequest(
                    task_id="task:unsupported-approved",
                    bindings=bindings,
                    seed_symbols=("Service.dispatch",),
                    candidates=(unsupported,),
                    approval_granted=True,
                ),
                impact_closure=closure,
            )
            # Approval-required class with grant becomes admitted candidate.
            assert receipt2.plan.candidates[0].disposition in {
                CandidateDisposition.ADMITTED,
                CandidateDisposition.REUSED,
            }
        finally:
            graph.close()


def test_counterexample_rejects_plan_and_repair(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner:
        receipt, _, _ = _plan_with_complete_impact(
            tmp_path,
            planner,
            counterexample_ids=("cex:dispatch-null",),
        )
        plan = receipt.plan
        assert plan.disposition is PlanDisposition.REJECTED
        assert plan.write_admitted is False
        assert PlanRejectReason.COUNTEREXAMPLE.value in plan.reasons

        with _open_repair(tmp_path) as repair:
            # Even if we forge write_admitted via a fresh admitted plan later,
            # counterexamples on the attempt reject.
            good, _, _ = _plan_with_complete_impact(
                tmp_path, planner, task_id="task:good-for-cex"
            )
            lineage = repair.apply_repair(
                RepairAttemptRequest(
                    plan=good.plan,
                    operator_class="constraint_rewrite",
                    write_paths=("src/service.py",),
                    counterexample_ids=("cex:dispatch-null",),
                    fixed_point=FixedPointEvidence(
                        code_fixed=True, logic_fixed=True
                    ),
                )
            )
            assert lineage.disposition is RepairDisposition.REJECTED
            assert (
                RepairRejectReason.COUNTEREXAMPLE.value in lineage.reasons
            )


def test_partial_plan_disposition_when_allowed(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner:
        graph = _materialize_impact(tmp_path)
        try:
            closure = graph.impact_closure(["Service.dispatch"])
            bindings = _fresh_bindings(
                impact_query_id=closure.query_id,
                impact_revision_id=closure.revision_id,
                schema_id=closure.schema_id,
            )
            # No candidates and allow_partial → partial (not write-admitted).
            receipt = planner.plan(
                SymbolicPlanRequest(
                    task_id="task:partial",
                    bindings=bindings,
                    seed_symbols=("Service.dispatch",),
                    candidates=(),
                    allow_partial=True,
                ),
                impact_closure=closure,
            )
            plan = receipt.plan
            assert plan.disposition is PlanDisposition.PARTIAL
            assert plan.write_admitted is False
        finally:
            graph.close()


# ---------------------------------------------------------------------------
# Proof cache hits rederive applicability
# ---------------------------------------------------------------------------


def test_proof_cache_hit_rederives_applicability(tmp_path: Path) -> None:
    with _open_repair(tmp_path) as repair:
        roots = {
            "snapshot": "snapshot:demo-1",
            "ast": "sha256:" + ("ab" * 32),
            "policy": DEFAULT_POLICY_ID,
            "mutation": "mutation:1",
        }
        key = RepairProofCacheKey(
            subject_id="obl:dispatch-total",
            semantic_roots=roots,
            obligation_ids=("obl:dispatch-total",),
            plan_id="plan:1",
            snapshot_id="snapshot:demo-1",
            mutation_id="mutation:1",
            policy_id=DEFAULT_POLICY_ID,
        )
        entry = RepairProofCacheEntry(
            key=key,
            verdict="pass",
            assurance_level=AssuranceLevel.SOLVER_CHECKED,
            content_digest="",
            body={"goal": "dispatch_total"},
        )
        stored = repair.put_proof_cache_entry(entry)
        assert stored.content_digest.startswith("sha256:")

        # Exact roots → applicable hit with rederived flag.
        hit = repair.lookup_proof_cache(
            key,
            current_roots=roots,
            required_assurance=AssuranceLevel.VALIDATED,
        )
        assert hit.status is ProofCacheLookupStatus.HIT
        assert hit.hit is True
        assert hit.applicable is True
        assert hit.applicability_rederived is True
        assert hit.entry is not None
        assert hit.entry.assurance_level == AssuranceLevel.SOLVER_CHECKED.value

        # Root drift → rejected even though key_id matches storage slot.
        drifted = repair.lookup_proof_cache(
            key,
            current_roots={**roots, "ast": "sha256:" + ("cd" * 32)},
            required_assurance=AssuranceLevel.VALIDATED,
        )
        assert drifted.status is ProofCacheLookupStatus.REJECTED
        assert drifted.applicable is False
        assert drifted.applicability_rederived is True
        assert drifted.reason is ProofCacheRejectReason.ROOT_MISMATCH

        # Required assurance above stored → no promotion.
        promoted = repair.lookup_proof_cache(
            key,
            current_roots=roots,
            required_assurance=AssuranceLevel.ATTESTED,
        )
        assert promoted.status is ProofCacheLookupStatus.REJECTED
        assert (
            promoted.reason is ProofCacheRejectReason.ASSURANCE_PROMOTION
        )

        # Invalidation tombstone.
        inv_id = repair.invalidate_proof_cache(
            key.key_id, reason="semantic_root_changed"
        )
        assert inv_id
        after = repair.lookup_proof_cache(key, current_roots=roots)
        assert after.status is ProofCacheLookupStatus.REJECTED
        assert after.reason is ProofCacheRejectReason.INVALIDATED
        assert after.applicability_rederived is True


def test_expired_and_poisoned_proof_cache_fail_closed(tmp_path: Path) -> None:
    with _open_repair(tmp_path) as repair:
        key = RepairProofCacheKey(
            subject_id="obl:x",
            semantic_roots={"snapshot": "s1"},
        )
        now = 1_700_000_000_000
        expired = RepairProofCacheEntry(
            key=key,
            verdict="pass",
            assurance_level=AssuranceLevel.VALIDATED,
            content_digest="",
            body={"ok": True},
            created_at_ms=now - 10_000,
            expires_at_ms=now - 1,
        )
        repair.put_proof_cache_entry(expired)
        result = repair.lookup_proof_cache(
            key, current_roots={"snapshot": "s1"}, now_ms=now
        )
        assert result.status is ProofCacheLookupStatus.REJECTED
        assert result.reason is ProofCacheRejectReason.EXPIRED

        poisoned_key = RepairProofCacheKey(
            subject_id="obl:y",
            semantic_roots={"snapshot": "s1"},
        )
        poisoned = RepairProofCacheEntry(
            key=poisoned_key,
            verdict="pass",
            assurance_level=AssuranceLevel.VALIDATED,
            content_digest="",
            body={"ok": True},
            created_at_ms=now,
            expires_at_ms=now + 60_000,
            poisoned=True,
        )
        repair.put_proof_cache_entry(poisoned)
        poisoned_result = repair.lookup_proof_cache(
            poisoned_key, current_roots={"snapshot": "s1"}, now_ms=now
        )
        assert poisoned_result.status is ProofCacheLookupStatus.REJECTED
        assert poisoned_result.reason is ProofCacheRejectReason.POISONED


# ---------------------------------------------------------------------------
# Fixed point or roll back
# ---------------------------------------------------------------------------


def test_accepted_repair_requires_code_and_logic_fixed_point(
    tmp_path: Path,
) -> None:
    with _open_planner(tmp_path) as planner, _open_repair(tmp_path) as repair:
        receipt, _, mutation_result = _plan_with_complete_impact(
            tmp_path, planner
        )
        plan = receipt.plan
        assert plan.write_admitted is True

        # Incomplete fixed point → roll back.
        incomplete = repair.apply_repair(
            RepairAttemptRequest(
                plan=plan,
                operator_class="constraint_rewrite",
                write_paths=("src/service.py",),
                obligation_ids=("obl:dispatch-total",),
                mutation_id=mutation_result.mutation.mutation_id,
                worktree_id="worktree:wt-1",
                worktree_digest="sha256:" + ("11" * 32),
                expected_worktree_digest="sha256:" + ("11" * 32),
                fixed_point=FixedPointEvidence(
                    code_fixed=True,
                    logic_fixed=False,
                    residual_obligations=("obl:dispatch-total",),
                    validation_receipt_id="val:1",
                    proof_receipt_id="",
                ),
            )
        )
        assert incomplete.disposition is RepairDisposition.ROLLED_BACK
        assert incomplete.rolled_back is True
        assert incomplete.write_committed is False
        assert incomplete.fixed_point_status is FixedPointStatus.ROLLED_BACK
        assert (
            RepairRejectReason.FIXED_POINT_FAILED.value in incomplete.reasons
        )
        assert incomplete.interface == REPAIR_LINEAGE_INTERFACE

        rollback = repair.get_rollback(incomplete.lineage_id)
        assert rollback is not None
        assert rollback.lineage_id == incomplete.lineage_id
        assert "src/service.py" in rollback.restored_paths

        # Full fixed point → accepted and write committed.
        complete = repair.apply_repair(
            RepairAttemptRequest(
                plan=plan,
                operator_class="constraint_rewrite",
                write_paths=("src/service.py",),
                obligation_ids=("obl:dispatch-total",),
                mutation_id=mutation_result.mutation.mutation_id,
                worktree_id="worktree:wt-1",
                worktree_digest="sha256:" + ("22" * 32),
                expected_worktree_digest="sha256:" + ("22" * 32),
                fixed_point=FixedPointEvidence(
                    code_fixed=True,
                    logic_fixed=True,
                    residual_obligations=(),
                    residual_paths=(),
                    validation_receipt_id="val:2",
                    proof_receipt_id="proof:2",
                    worktree_digest="sha256:" + ("22" * 32),
                    expected_worktree_digest="sha256:" + ("22" * 32),
                ),
            )
        )
        assert complete.disposition is RepairDisposition.ACCEPTED
        assert complete.accepted is True
        assert complete.write_committed is True
        assert complete.rolled_back is False
        assert complete.fixed_point_status is FixedPointStatus.REACHED
        assert complete.fixed_point["reached"] is True
        assert complete.mutation_id == mutation_result.mutation.mutation_id
        assert complete.plan_id == plan.plan_id

        loaded = repair.get_lineage(complete.lineage_id)
        assert loaded is not None
        assert loaded.accepted is True
        assert loaded.to_dict()["authority"] == REPAIR_AUTHORITY
        assert loaded.to_dict()["cache_assurance_policy"] == CACHE_ASSURANCE_POLICY

        meta = repair.metadata()
        assert meta["interface"] == REPAIR_LINEAGE_INTERFACE
        assert meta["write_authority_policy"] == WRITE_AUTHORITY_POLICY


def test_worktree_mismatch_prevents_write(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner, _open_repair(tmp_path) as repair:
        receipt, _, _ = _plan_with_complete_impact(tmp_path, planner)
        plan = receipt.plan
        lineage = repair.apply_repair(
            RepairAttemptRequest(
                plan=plan,
                operator_class="constraint_rewrite",
                write_paths=("src/service.py",),
                worktree_digest="sha256:" + ("aa" * 32),
                expected_worktree_digest="sha256:" + ("bb" * 32),
                fixed_point=FixedPointEvidence(
                    code_fixed=True,
                    logic_fixed=True,
                ),
            )
        )
        assert lineage.write_committed is False
        assert lineage.disposition is RepairDisposition.ABSTAINED
        assert RepairRejectReason.WORKTREE_MISMATCH.value in lineage.reasons


def test_inapplicable_proof_cache_blocks_write(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner, _open_repair(tmp_path) as repair:
        receipt, _, _ = _plan_with_complete_impact(tmp_path, planner)
        plan = receipt.plan
        key = RepairProofCacheKey(
            subject_id="obl:dispatch-total",
            semantic_roots={"snapshot": "snapshot:demo-1", "ast": "old"},
            plan_id=plan.plan_id,
            snapshot_id=plan.bindings.snapshot_id,
        )
        # Store under old roots, then consult with different current roots
        # by constructing a key that shares key_id... Actually key_id includes
        # roots, so use put then invalidate path: store with roots A, lookup
        # with same key but pass different current_roots.
        repair.put_proof_cache_entry(
            RepairProofCacheEntry(
                key=key,
                verdict="pass",
                assurance_level=AssuranceLevel.VALIDATED,
                content_digest="",
                body={"goal": "x"},
            )
        )
        lineage = repair.apply_repair(
            RepairAttemptRequest(
                plan=plan,
                operator_class="constraint_rewrite",
                write_paths=("src/service.py",),
                proof_cache_key=key,
                # apply_repair re-looks up with key.semantic_roots; force
                # inapplicable by poisoning after store.
                fixed_point=FixedPointEvidence(
                    code_fixed=True, logic_fixed=True
                ),
            )
        )
        # With matching roots this would be a hit; poison to force reject.
        # Re-run after invalidation.
        repair.invalidate_proof_cache(key.key_id, reason="test_poison")
        lineage2 = repair.apply_repair(
            RepairAttemptRequest(
                plan=plan,
                operator_class="constraint_rewrite",
                write_paths=("src/service.py",),
                proof_cache_key=key,
                fixed_point=FixedPointEvidence(
                    code_fixed=True, logic_fixed=True
                ),
            )
        )
        assert lineage2.write_committed is False
        assert (
            RepairRejectReason.PROOF_CACHE_INAPPLICABLE.value
            in lineage2.reasons
        )
        assert lineage2.proof_cache["applicability_rederived"] is True
        # First lineage without invalidation should have been accepted when
        # roots match.
        assert lineage.disposition is RepairDisposition.ACCEPTED or (
            lineage.proof_cache.get("applicable") is True
        )


def test_scope_escape_on_repair_is_rejected(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner, _open_repair(tmp_path) as repair:
        receipt, _, _ = _plan_with_complete_impact(tmp_path, planner)
        plan = receipt.plan
        lineage = repair.apply_repair(
            RepairAttemptRequest(
                plan=plan,
                operator_class="constraint_rewrite",
                write_paths=("src/other.py",),  # not in admitted paths
                fixed_point=FixedPointEvidence(
                    code_fixed=True, logic_fixed=True
                ),
            )
        )
        assert lineage.disposition is RepairDisposition.REJECTED
        assert lineage.write_committed is False
        assert RepairRejectReason.SCOPE_ESCAPE.value in lineage.reasons


def test_unsupported_repair_operator_requires_approval(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner, _open_repair(tmp_path) as repair:
        receipt, _, _ = _plan_with_complete_impact(tmp_path, planner)
        plan = receipt.plan
        lineage = repair.apply_repair(
            RepairAttemptRequest(
                plan=plan,
                operator_class="architecture_rewrite",
                write_paths=("src/service.py",),
                approval_granted=False,
                fixed_point=FixedPointEvidence(
                    code_fixed=True, logic_fixed=True
                ),
            )
        )
        assert lineage.disposition is RepairDisposition.REQUIRES_APPROVAL
        assert lineage.write_committed is False
        assert RepairRejectReason.REQUIRES_APPROVAL.value in lineage.reasons


def test_list_plans_and_lineages(tmp_path: Path) -> None:
    with _open_planner(tmp_path) as planner, _open_repair(tmp_path) as repair:
        receipt, _, _ = _plan_with_complete_impact(tmp_path, planner)
        plans = planner.list_plans(task_id="task:DQP-024")
        assert any(p.plan_id == receipt.plan.plan_id for p in plans)

        lineage = repair.apply_repair(
            RepairAttemptRequest(
                plan=receipt.plan,
                operator_class="constraint_rewrite",
                write_paths=("src/service.py",),
                fixed_point=FixedPointEvidence(
                    code_fixed=True, logic_fixed=True
                ),
            )
        )
        listed = repair.list_lineages(plan_id=receipt.plan.plan_id)
        assert any(item.lineage_id == lineage.lineage_id for item in listed)
        accepted = repair.list_lineages(
            disposition=RepairDisposition.ACCEPTED
        )
        assert any(item.lineage_id == lineage.lineage_id for item in accepted)


def test_mutation_status_not_accepted_blocks_auto_admission(
    tmp_path: Path,
) -> None:
    """Rejected mutation cannot drive an auto-admitted write plan."""

    with _open_planner(tmp_path) as planner:
        ledger = open_mutation_ledger(tmp_path / "mutation_bad.duckdb")
        try:
            fence = ledger.register_fence(
                worktree_id="worktree:wt-stale",
                token="token-stale",
                before_snapshot_id="snapshot:demo-1",
            )
            # Supersede fence so mutation is rejected as stale.
            ledger.register_fence(
                worktree_id="worktree:wt-stale",
                token="token-new",
                before_snapshot_id="snapshot:demo-1",
            )
            bad = ledger.record_mutation(
                MutationContext(
                    task_id="task:stale-mut",
                    worktree_id=fence.worktree_id,
                    fence_id=fence.fence_id,
                    before_snapshot_id="snapshot:demo-1",
                ),
                [
                    MutationFileSpec(
                        path="src/service.py",
                        before_content=PYTHON_V1,
                        after_content=PYTHON_V2,
                    )
                ],
            )
            assert bad.admitted is False
            assert bad.mutation.status is not MutationStatus.ACCEPTED

            graph = _materialize_impact(tmp_path)
            try:
                closure = graph.impact_closure(["Service.dispatch"])
                bindings = _fresh_bindings(
                    mutation_id=bad.mutation.mutation_id,
                    impact_query_id=closure.query_id,
                    impact_revision_id=closure.revision_id,
                    schema_id=closure.schema_id,
                )
                receipt = planner.plan(
                    SymbolicPlanRequest(
                        task_id="task:stale-mut",
                        bindings=bindings,
                        seed_symbols=("Service.dispatch",),
                        candidates=(_admitted_candidate(),),
                    ),
                    impact_closure=closure,
                    mutation=bad.mutation,
                )
                assert receipt.plan.write_admitted is False
                assert (
                    PlanRejectReason.MUTATION_NOT_ACCEPTED.value
                    in receipt.plan.reasons
                )
            finally:
                graph.close()
        finally:
            ledger.close()
