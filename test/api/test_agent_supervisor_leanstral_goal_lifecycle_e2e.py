from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import (
    ASTGraphRAGReferenceRecord,
    CapabilityRecord,
    CodeReferenceKind,
    ConfiguredLeanstralGoalLifecycleSupervisor,
    EvidenceGapRecord,
    GoalDevelopmentMode,
    GoalDevelopmentPolicy,
    GoalDevelopmentTemplate,
    GoalRefinementVerifier,
    GoalState,
    ImplementationEvidenceKind,
    ImplementationResultEvidence,
    LeanstralGoalDevelopmentInvocation,
    LeanstralGoalDevelopmentProvider,
    PriorCounterexampleRecord,
    ReusableReceiptRecord,
    build_configured_leanstral_goal_lifecycle_supervisor,
    build_leanstral_goal_development_context,
    compile_candidate_proof_scopes,
    derive_fresh_implementation_obligations,
    evaluate_code_proof_goal_completion,
)
from ipfs_accelerate_py.agent_supervisor.formal_logic_vocabulary import (
    LOGIC_VOCABULARY_VERSION,
    ReviewedPredicate,
    TDFOLVocabulary,
    TermSort,
    atom,
    constant,
)
from ipfs_accelerate_py.agent_supervisor.formal_planning_contracts import (
    Actor,
    ActorKind,
    Effect,
    EffectOperation,
    EventKind,
    EvidenceRequirement,
    EvidenceRequirementKind,
    Fluent,
    FluentValueType,
    FormalWorkPlan,
    Goal,
    Norm,
    NormKind,
    PlanEvent,
    PlanTask,
    RefinementMode,
    Subgoal,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    EvidenceAuthority,
    EvidenceFreshness,
    EvidenceKind,
    EvidenceVerdict,
    ProofEvidence,
    ProofReceipt,
    ProofVerdict,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.goal_development_contracts import (
    GoalDevelopmentRequest,
)
from ipfs_accelerate_py.agent_supervisor.goal_refinement_verification import (
    JsonlRefinementAuditStore,
    RefinementRepairCandidate,
    RefinementVerificationPolicy,
)
from ipfs_accelerate_py.agent_supervisor.multi_prover_router import (
    AttemptOutcome,
    PropertyKind,
    ProverOutput,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    ObjectiveWorkKind,
    ObjectiveWorkProposal,
    objective_goal_content_id,
    parse_goal_heap,
)


NOW = datetime(2026, 7, 24, 0, 0, tzinfo=timezone.utc)


def _objective_fixture(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    objective = repo / "objective.md"
    objective.write_text(
        """# Objective

## ROOT Preserve truthful goal completion

- Status: active
- Acceptance: implementation evidence remains fresh
- Evidence: evidence:implementation
- Outputs: src/runtime.py
""",
        encoding="utf-8",
    )
    completion = repo / "completion.json"
    completion.write_text(
        '{"ROOT":{"state":"provisionally_complete","verified":false}}\n',
        encoding="utf-8",
    )
    state = repo / "state"
    return repo, objective, completion


def _invocation(objective: Path) -> LeanstralGoalDevelopmentInvocation:
    root = parse_goal_heap(objective.read_text(encoding="utf-8"))[0]
    policy = GoalDevelopmentPolicy(
        mode=GoalDevelopmentMode.SHADOW,
        max_depth=3,
        max_breadth=4,
        max_proposals=6,
        max_bytes=64_000,
        max_tokens=4_096,
    )
    request = GoalDevelopmentRequest(
        root_goal_id=root.goal_id,
        root_goal_content_id=objective_goal_content_id(root),
        satisfaction_formula_id="formula:root-satisfied",
        assumption_ids=("assumption:reviewed",),
        evidence_requirement_ids=(
            "evidence:implementation",
            "evidence:validation",
        ),
        vocabulary_profile_id="reviewed-tdfol",
        vocabulary_version=LOGIC_VOCABULARY_VERSION,
        repository_tree_id="git-tree:fixture",
        scope_ids=("scope:runtime",),
        policy_digest=policy.policy_digest,
        mode=policy.mode,
    )
    templates = (
        GoalDevelopmentTemplate(
            template_id="template:implementation",
            satisfaction_formula_id="formula:implementation",
            evidence_requirement_ids=("evidence:implementation",),
            assurance_ids=("assurance:kernel",),
            resource_ids=("resource:model",),
            scope_ids=("scope:runtime",),
            validation_check_ids=("check:type",),
        ),
        GoalDevelopmentTemplate(
            template_id="template:validation",
            satisfaction_formula_id="formula:validation",
            evidence_requirement_ids=("evidence:validation",),
            assurance_ids=("assurance:kernel",),
            resource_ids=("resource:kernel",),
            scope_ids=("scope:runtime",),
            validation_check_ids=("check:test",),
        ),
    )
    context = build_leanstral_goal_development_context(
        request,
        templates=templates,
        evidence_gaps=(
            EvidenceGapRecord(
                "gap:implementation",
                ("evidence:implementation", "evidence:validation"),
            ),
        ),
        code_references=(
            ASTGraphRAGReferenceRecord(
                "ast:runtime",
                CodeReferenceKind.AST,
                request.repository_tree_id,
                request.scope_ids,
                ("symbol:runtime.dispatch",),
            ),
        ),
        capabilities=(
            CapabilityRecord(
                "capability:kernel",
                ("resource:kernel",),
                ("check:test",),
                True,
            ),
        ),
        prior_counterexamples=(
            PriorCounterexampleRecord(
                "counterexample:stale-receipt",
                "stale_implementation_binding",
            ),
        ),
        reusable_receipts=(
            ReusableReceiptRecord(
                "receipt:reviewed",
                ("evidence:implementation",),
                "assurance:kernel",
                ("scope:runtime",),
            ),
        ),
        max_records_per_kind=2,
    )
    return LeanstralGoalDevelopmentInvocation(
        request=request,
        policy=policy,
        context=context,
        resource_budget=ResourceBudget(
            wall_time_ms=5_000,
            model_token_limit=2_048,
            max_output_bytes=64_000,
        ),
    )


def _proposal(
    proposal_id: str,
    *,
    template: str,
    evidence: str,
    resource: str,
    check: str,
    depends_on: tuple[str, ...] = (),
) -> dict[str, object]:
    return {
        "proposal_id": proposal_id,
        "parent_id": "ROOT",
        "template_id": template,
        "title": proposal_id.replace("subgoal:", "").replace("-", " ").title(),
        "evidence_requirement_ids": [evidence],
        "assurance_ids": ["assurance:kernel"],
        "resource_ids": [resource],
        "scope_ids": ["scope:runtime"],
        "validation_check_ids": [check],
        "depends_on": list(depends_on),
    }


def _output(
    invocation: LeanstralGoalDevelopmentInvocation,
    proposals: tuple[dict[str, object], ...],
) -> str:
    return json.dumps(
        {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "leanstral-goal-development-output@1"
            ),
            "operation": "goal_development.v1",
            "request_id": invocation.request_id,
            "proposals": list(proposals),
        }
    )


class _UnavailableProvider:
    def develop(self, _invocation):
        raise ModuleNotFoundError("fixture route is unavailable")


class _WrongTypeProvider:
    def develop(self, _invocation):
        return ["not", "a", "versioned", "result"]


def test_configured_shadow_path_audits_candidates_metrics_preview_and_restart(
    tmp_path: Path,
) -> None:
    repo, objective, completion = _objective_fixture(tmp_path)
    invocation = _invocation(objective)
    implementation = _proposal(
        "subgoal:implementation",
        template="template:implementation",
        evidence="evidence:implementation",
        resource="resource:model",
        check="check:type",
    )
    validation = _proposal(
        "subgoal:validation",
        template="template:validation",
        evidence="evidence:validation",
        resource="resource:kernel",
        check="check:test",
        depends_on=("subgoal:implementation",),
    )
    providers = (
        LeanstralGoalDevelopmentProvider(
            llm_generate=lambda *_a, **_k: _output(invocation, (implementation,))
        ),
        LeanstralGoalDevelopmentProvider(
            llm_generate=lambda *_a, **_k: _output(
                invocation, (implementation, validation)
            )
        ),
        LeanstralGoalDevelopmentProvider(
            llm_generate=lambda *_a, **_k: "{malformed"
        ),
        _UnavailableProvider(),
        _WrongTypeProvider(),
    )
    supervisor = build_configured_leanstral_goal_lifecycle_supervisor(
        state_dir=repo / "state",
        providers=providers,
        max_candidates=5,
    )
    assert supervisor.config.mode is GoalDevelopmentMode.SHADOW
    assert isinstance(supervisor, ConfiguredLeanstralGoalLifecycleSupervisor)
    assert invocation.context.max_records_per_kind == 2
    prompt = providers[0].build_prompt(invocation)
    assert '"canonical_source":' not in prompt
    assert len(invocation.context.code_references) == 1

    supervisor.config.state_dir.mkdir(parents=True)
    supervisor.config.generation_path.write_text(
        '{"existing_generation":"operator-reviewed"}\n',
        encoding="utf-8",
    )
    objective_work = (
        ObjectiveWorkProposal(
            kind=ObjectiveWorkKind.SUBGOAL,
            title="Implement runtime conformance",
            parent_goal_id="ROOT",
            parent_objective_terms=("truthful", "completion"),
            expected_evidence_delta=("evidence:implementation",),
            dependencies=(),
            predicted_files=("src/runtime.py",),
            predicted_symbols=("runtime.dispatch",),
            validation_commands=("python -m pytest test_runtime.py -q",),
            confidence=0.8,
            estimated_cost=2.0,
            novelty=0.9,
            depth=1,
            source="leanstral-reviewed-projection",
            source_id="subgoal:implementation",
        ),
        ObjectiveWorkProposal(
            kind=ObjectiveWorkKind.SUBGOAL,
            title="Validate runtime conformance",
            parent_goal_id="ROOT",
            parent_objective_terms=("truthful", "completion"),
            expected_evidence_delta=("evidence:validation",),
            dependencies=(),
            predicted_files=("test/test_runtime.py",),
            predicted_symbols=("test_runtime_conformance",),
            validation_commands=("python -m pytest test_runtime.py -q",),
            confidence=0.8,
            estimated_cost=1.0,
            novelty=0.9,
            depth=1,
            source="leanstral-reviewed-projection",
            source_id="subgoal:validation",
        ),
    )
    original_heap = objective.read_bytes()
    original_completion = completion.read_bytes()
    original_generation = supervisor.config.generation_path.read_bytes()

    run = supervisor.run(
        invocation,
        repo_root=repo,
        objective_path=objective,
        completion_state_path=completion,
        objective_work=objective_work,
    )

    assert run.mode is GoalDevelopmentMode.SHADOW
    assert run["candidate_count"] == 5
    assert run.selected_draft_id == run["proposal_receipt"]["draft_id"]
    assert len(run["candidates"][1]["result"]["draft"]["proposals"]) == 2
    assert run["candidates"][2]["result"]["fallback_reason"] == "malformed_output"
    assert run["candidates"][3]["result"]["fallback_reason"] == "unavailable"
    assert run["candidates"][4]["reason_codes"] == ["type_or_schema_rejection"]
    assert run["objective_admission"]["status"] == "shadow"
    assert run["objective_admission"]["preview"]["materialized"]
    assert run["admission_receipt"]["decision"] == "not_admitted"
    assert run["admission_receipt"]["reason_codes"] == ["shadow_mode"]
    assert run.objective_heap_unchanged
    assert run.completion_state_unchanged
    assert run.generation_state_unchanged
    assert objective.read_bytes() == original_heap
    assert completion.read_bytes() == original_completion
    assert supervisor.config.generation_path.read_bytes() == original_generation

    audit_lines = supervisor.config.audit_path.read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(audit_lines) == 1
    assert json.loads(audit_lines[0])["run"]["run_id"] == run.run_id
    metrics = json.loads(supervisor.config.metrics_path.read_text(encoding="utf-8"))
    assert metrics["totals"]["attempt_count"] == 5
    assert metrics["totals"]["schema_validation_count"] == 5
    assert metrics["totals"]["schema_acceptance_count"] == 2
    assert metrics["totals"]["fallback_count"] == 3

    # State loss/corruption recovers from the append-only audit journal.
    supervisor.config.state_path.write_text("{corrupt", encoding="utf-8")
    recovered = supervisor.recover()
    assert recovered is not None
    assert recovered.run_id == run.run_id
    assert recovered.to_dict() == run.to_dict()


def _formal_plan() -> FormalWorkPlan:
    task_ready = atom(
        ReviewedPredicate.TASK_READY,
        constant(TermSort.TASK, "task:implement"),
    )
    root_satisfied = atom(
        ReviewedPredicate.GOAL_SATISFIED,
        constant(TermSort.GOAL, "goal:root"),
    )
    child_satisfied = TDFOLVocabulary.subgoal_satisfaction("subgoal:child", 4)
    evidence = EvidenceRequirement(
        "evidence:test",
        EvidenceRequirementKind.TEST,
        ("task:implement", "subgoal:child"),
        freshness_seconds=60,
        fallback_check_ids=("pytest:fixture",),
    )
    return FormalWorkPlan(
        vocabulary_profile_id="reviewed-tdfol",
        vocabulary_version=LOGIC_VOCABULARY_VERSION,
        source_ids=("source:objective",),
        repository_tree_id="git-tree:fixture",
        trace_bound=4,
        actors=(
            Actor(
                "actor:supervisor",
                ActorKind.SUPERVISOR,
                authority_ids=("authority:assign",),
            ),
            Actor(
                "actor:agent",
                ActorKind.AGENT,
                capabilities=("python",),
                authority_ids=("authority:implement",),
            ),
        ),
        goals=(
            Goal(
                "goal:root",
                "actor:supervisor",
                root_satisfied.formula_id,
                evidence_requirement_ids=("evidence:test",),
            ),
        ),
        subgoals=(
            Subgoal(
                "subgoal:child",
                "goal:root",
                child_satisfied.formula_id,
                refinement_mode=RefinementMode.SUFFICIENT,
                evidence_requirement_ids=("evidence:test",),
            ),
        ),
        tasks=(
            PlanTask(
                "task:implement",
                "goal:root",
                subgoal_id="subgoal:child",
                actor_ids=("actor:agent",),
                effect_ids=("effect:done",),
                event_ids=("event:execute",),
                evidence_requirement_ids=("evidence:test",),
                metadata={
                    "required_authority_ids": ["authority:implement"],
                    "resource_ids": ["cpu"],
                },
            ),
        ),
        events=(
            PlanEvent(
                "event:execute",
                EventKind.EXECUTED,
                "actor:agent",
                "task:implement",
                2,
            ),
        ),
        fluents=(Fluent("fluent:done", FluentValueType.BOOLEAN, False),),
        preconditions=(),
        effects=(
            Effect(
                "effect:done",
                EffectOperation.ASSIGN,
                "task:implement",
                fluent_id="fluent:done",
                event_id="event:execute",
                value=True,
            ),
        ),
        norms=(
            Norm(
                "norm:implement",
                NormKind.OBLIGATION,
                "actor:agent",
                "task:implement",
                issuer_actor_id="actor:supervisor",
                activation_formula_id=task_ready.formula_id,
                valid_from=0,
                valid_until=4,
            ),
        ),
        temporal_constraints=(),
        evidence_requirements=(evidence,),
        formulas=(task_ready, root_satisfied, child_satisfied),
        metadata={"resource_ids": ["cpu", "memory"]},
    )


def test_counterexample_repair_requires_frozen_root_and_independent_acceptance(
    tmp_path: Path,
) -> None:
    plan = _formal_plan()
    repaired = replace(plan, metadata={**plan.metadata, "repair_round": 1})
    store = JsonlRefinementAuditStore(tmp_path / "refinement.jsonl")

    def runner(request, _cancellation):
        if (
            request.obligation.metadata["plan_id"] == plan.content_id
            and request.obligation.property_kind is PropertyKind.FINITE_CONSTRAINT
        ):
            return ProverOutput(
                AttemptOutcome.COUNTEREXAMPLE,
                evidence={"model": {"authority": False}},
                conclusive=True,
            )
        if request.obligation.metadata["plan_id"] == plan.content_id:
            return ProverOutput(AttemptOutcome.UNKNOWN)
        return ProverOutput(AttemptOutcome.VERIFIED)

    repair_requests = []

    def repairer(request):
        repair_requests.append(request)
        return RefinementRepairCandidate(repaired, request.frozen_context)

    result = GoalRefinementVerifier(
        policy=RefinementVerificationPolicy(
            allow_leanstral_repairs=True,
            max_repair_rounds=1,
        ),
        audit_store=store,
    ).verify(
        plan,
        runner,
        assumption_ids=("assumption:reviewed",),
        repairer=repairer,
    )

    assert result.verified
    assert len(result.rounds) == 2
    assert result.rounds[0].counterexamples
    assert repair_requests[0].counterexamples
    assert result.repair_receipts[0].accepted
    assert result.repair_receipts[0].root_unchanged
    assert result.repair_receipts[0].assumptions_unchanged
    assert all(item.proved for item in result.rounds[1].portfolio_results)
    assert all(
        any(attempt.authoritative for attempt in portfolio.attempts)
        for portfolio in result.rounds[1].portfolio_results
    )
    assert store.path.read_text(encoding="utf-8").count("\n") > len(
        result.rounds
    )


def _implementation_manifest(repository_tree: str):
    scopes = compile_candidate_proof_scopes(
        (
            {
                "new_path": "src/runtime.py",
                "status": "add",
                "after_source": (
                    "def dispatch(value: int) -> int:\n"
                    "    return value + 1\n"
                ),
            },
        )
    )
    bounds = {"python": "3.12", "tests": ["test_runtime.py"]}
    evidence = ImplementationResultEvidence(
        kind=ImplementationEvidenceKind.TEST,
        accepted_plan_id="plan:accepted",
        repository_id="repo:fixture",
        repository_tree_id=repository_tree,
        scope_ids=scopes.scope_ids,
        passed=True,
        observed_at=NOW,
        validation_bounds=bounds,
        producer_id="pytest:fixture",
        artifact_id="artifact:test",
    )
    return derive_fresh_implementation_obligations(
        scopes,
        accepted_plan_id="plan:accepted",
        repository_id="repo:fixture",
        repository_tree_id=repository_tree,
        validation_bounds=bounds,
        test_evidence=(evidence,),
        planned_effect_ids=("effect:dispatch",),
        task_id="LEAN-GOAL-009",
    )


def _implementation_receipt(obligation, binding) -> ProofReceipt:
    evidence = ProofEvidence(
        EvidenceKind.KERNEL_VERIFICATION,
        EvidenceAuthority.KERNEL,
        EvidenceVerdict.ACCEPTED,
        "artifact:kernel-reconstruction",
        obligation.obligation_id,
        "kernel:lean-fixture",
        independent=True,
    )
    return ProofReceipt(
        obligation_id=obligation.obligation_id,
        plan_id=binding.accepted_plan_id,
        attempt_id=f"attempt:{obligation.obligation_id}",
        repository_id=binding.repository_id,
        repository_tree_id=binding.repository_tree_id,
        ast_scope_ids=obligation.ast_scope_ids,
        premise_ids=obligation.premise_ids,
        translator_id="translator:fixture",
        solver_id="solver:fixture",
        kernel_id="kernel:lean-fixture",
        toolchain_id="toolchain:locked",
        policy_id="policy:implementation-conformance",
        resource_budget=ResourceBudget(wall_time_ms=5_000),
        verdict=ProofVerdict.PROVED,
        evidence=(evidence,),
        freshness=EvidenceFreshness.CURRENT,
        finished_at=NOW.isoformat(),
        metadata=binding.receipt_metadata(obligation=obligation),
    )


def test_implementation_conformance_accepts_fresh_receipts_and_reopens_stale_tree(
) -> None:
    current = _implementation_manifest("git-tree:current")
    fresh_receipts = tuple(
        _implementation_receipt(obligation, current.binding)
        for obligation in current.obligations
    )

    accepted = evaluate_code_proof_goal_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        binding=current.binding,
        required_obligations=current,
        receipts=fresh_receipts,
    )

    assert accepted.verified
    assert accepted.state is GoalState.VERIFIED_COMPLETE
    assert accepted.all_obligations_satisfied
    assert all(item["valid"] for item in accepted.receipt_results)

    old = _implementation_manifest("git-tree:old")
    stale_receipts = tuple(
        _implementation_receipt(obligation, old.binding)
        for obligation in old.obligations
    )
    reopened = evaluate_code_proof_goal_completion(
        current_state=GoalState.VERIFIED_COMPLETE,
        binding=current.binding,
        required_obligations=current,
        receipts=stale_receipts,
    )

    assert reopened.state is GoalState.REOPENED
    assert reopened.reopened
    assert reopened.stale
    assert not reopened.verified
    assert "receipt_not_required_by_fresh_obligation_set" in reopened.reason_codes
