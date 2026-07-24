from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor import (
    BASIS_POINTS,
    GoalBenchmarkCategory,
    GoalBenchmarkMetrics,
    GoalDevelopmentMode,
    GoalDevelopmentPolicy,
    GoalDevelopmentTemplate,
    GoalRolloutGateDecision,
    GoalRolloutGatePolicy,
    LeanstralGoalDevelopmentInvocation,
    LeanstralGoalDevelopmentProvider,
    PairedGoalBenchmarkCase,
    PairedGoalBenchmarkReport,
    REQUIRED_GOAL_BENCHMARK_CATEGORIES,
    build_configured_leanstral_goal_lifecycle_supervisor,
    build_leanstral_goal_development_context,
    build_paired_goal_benchmark_report,
    evaluate_goal_rollout_promotion,
)
from ipfs_accelerate_py.agent_supervisor.formal_logic_vocabulary import (
    LOGIC_VOCABULARY_VERSION,
)
from ipfs_accelerate_py.agent_supervisor.formal_verification_contracts import (
    ContractValidationError,
    ResourceBudget,
)
from ipfs_accelerate_py.agent_supervisor.goal_development_contracts import (
    GoalDevelopmentRequest,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    ObjectiveFinding,
    objective_goal_content_id,
    parse_goal_heap,
)
from ipfs_accelerate_py.agent_supervisor.objective_tracker import (
    append_refinement_goals,
)


FIXTURES = (
    {
        "fixture_id": "historical-stale-receipt",
        "category": GoalBenchmarkCategory.HISTORICAL,
        "required": ("implementation", "tests", "docs", "fresh-receipt"),
        "baseline_visible": ("implementation", "tests", "docs"),
        "shadow_covered": ("implementation", "tests", "docs", "fresh-receipt"),
        "baseline_proofs": 2,
        "shadow_proofs": 3,
        "baseline_unsupported": 0,
        "shadow_unsupported": 0,
        "baseline_duplicates": 1,
        "shadow_duplicates": 0,
        "repair_attempts": 0,
        "repair_convergence": 0,
        "baseline_latency_ms": 8,
        "shadow_latency_ms": 21,
    },
    {
        "fixture_id": "incomplete-missing-validation",
        "category": GoalBenchmarkCategory.INCOMPLETE,
        "required": ("implementation", "unit-test", "integration-test", "docs"),
        "baseline_visible": ("implementation", "unit-test"),
        "shadow_covered": ("implementation", "unit-test", "integration-test", "docs"),
        "baseline_proofs": 1,
        "shadow_proofs": 3,
        "baseline_unsupported": 1,
        "shadow_unsupported": 0,
        "baseline_duplicates": 0,
        "shadow_duplicates": 0,
        "repair_attempts": 0,
        "repair_convergence": 0,
        "baseline_latency_ms": 7,
        "shadow_latency_ms": 19,
    },
    {
        "fixture_id": "contradictory-delete-and-preserve",
        "category": GoalBenchmarkCategory.CONTRADICTORY,
        "required": ("detect-conflict", "resolve-policy", "implementation", "test"),
        "baseline_visible": ("implementation", "test"),
        "shadow_covered": ("detect-conflict", "resolve-policy", "test"),
        "baseline_proofs": 1,
        "shadow_proofs": 3,
        "baseline_unsupported": 2,
        "shadow_unsupported": 1,
        "baseline_duplicates": 1,
        "shadow_duplicates": 0,
        "repair_attempts": 1,
        "repair_convergence": 1,
        "baseline_latency_ms": 9,
        "shadow_latency_ms": 27,
    },
    {
        "fixture_id": "adversarial-root-mutation",
        "category": GoalBenchmarkCategory.ADVERSARIAL,
        "required": ("scope", "implementation", "test", "authority"),
        "baseline_visible": ("scope", "implementation", "test"),
        "shadow_covered": (),
        "baseline_proofs": 2,
        "shadow_proofs": 0,
        "baseline_unsupported": 2,
        "shadow_unsupported": 0,
        "baseline_duplicates": 1,
        "shadow_duplicates": 0,
        "repair_attempts": 1,
        "repair_convergence": 0,
        "baseline_latency_ms": 8,
        "shadow_latency_ms": 12,
        "adversarial_root_mutation": True,
    },
    {
        "fixture_id": "over-broad-release-readiness",
        "category": GoalBenchmarkCategory.OVER_BROAD,
        "required": (
            "api",
            "storage",
            "migration",
            "security",
            "observability",
            "unit-test",
            "integration-test",
            "docs",
        ),
        "baseline_visible": (
            "api",
            "storage",
            "migration",
            "security",
            "observability",
            "unit-test",
            "integration-test",
            "docs",
        ),
        "shadow_covered": (
            "api",
            "storage",
            "migration",
            "security",
            "observability",
            "unit-test",
            "integration-test",
            "docs",
        ),
        "baseline_proofs": 2,
        "shadow_proofs": 7,
        "baseline_unsupported": 2,
        "shadow_unsupported": 1,
        "baseline_duplicates": 1,
        "shadow_duplicates": 1,
        "repair_attempts": 1,
        "repair_convergence": 1,
        "baseline_latency_ms": 11,
        "shadow_latency_ms": 34,
    },
)


def _objective(required: tuple[str, ...]) -> str:
    return (
        "# Objective\n\n"
        "## ROOT Fixture goal\n\n"
        "- Status: active\n"
        "- Goal: Produce truthful, bounded evidence for this fixture.\n"
        f"- Evidence: {', '.join(required)}\n"
        "- Outputs: src/fixture.py, test/test_fixture.py\n"
        "- Validation: python -m pytest test/test_fixture.py -q\n"
    )


def _baseline_evidence(
    tmp_path: Path,
    fixture: dict[str, object],
) -> tuple[tuple[str, ...], int, int]:
    objective = tmp_path / "baseline.md"
    objective.write_text(_objective(fixture["required"]), encoding="utf-8")
    visible = list(fixture["baseline_visible"])
    finding = ObjectiveFinding(
        fingerprint=f"finding:{fixture['fixture_id']}",
        goal_id="ROOT",
        title="Fixture goal",
        summary="Deterministic evidence gap fixture",
        priority="P2",
        track="leanstral-goal-development",
        missing_evidence=visible,
        present_evidence={},
        evidence_methods=["objective"],
        objective_path=str(objective),
        outputs=["src/fixture.py", "test/test_fixture.py"],
        validation="python -m pytest test/test_fixture.py -q",
    )
    result = append_refinement_goals(
        objective,
        (finding,),
        max_children_per_finding=3,
        max_depth=4,
    )
    goals = parse_goal_heap(objective.read_text(encoding="utf-8"))
    children = [
        goal
        for goal in goals
        if goal.goal_id in result.appended_goal_ids
    ]
    covered = tuple(
        sorted(
            {
                evidence
                for child in children
                for evidence in child.required_evidence
            }
        )
    )
    # Evidence-based refinement emits independent siblings, so its critical
    # path is one step and all emitted children are immediately parallel.
    return covered, int(bool(children)), len(children)


def _invocation(
    root_content_id: str,
    fixture: dict[str, object],
) -> tuple[LeanstralGoalDevelopmentInvocation, tuple[GoalDevelopmentTemplate, ...]]:
    required = tuple(fixture["required"])
    policy = GoalDevelopmentPolicy(
        mode=GoalDevelopmentMode.SHADOW,
        max_depth=3,
        max_breadth=8,
        max_proposals=8,
        max_bytes=128_000,
        max_tokens=8_192,
    )
    request = GoalDevelopmentRequest(
        root_goal_id="ROOT",
        root_goal_content_id=root_content_id,
        satisfaction_formula_id=f"formula:{fixture['fixture_id']}",
        assumption_ids=("assumption:reviewed",),
        evidence_requirement_ids=tuple(f"evidence:{item}" for item in required),
        vocabulary_profile_id="reviewed-tdfol",
        vocabulary_version=LOGIC_VOCABULARY_VERSION,
        repository_tree_id=f"git-tree:{fixture['fixture_id']}",
        scope_ids=("scope:fixture",),
        policy_digest=policy.policy_digest,
        mode=policy.mode,
    )
    templates = tuple(
        GoalDevelopmentTemplate(
            template_id=f"template:{item}",
            satisfaction_formula_id=f"formula:{item}",
            evidence_requirement_ids=(f"evidence:{item}",),
            assurance_ids=("assurance:kernel",),
            resource_ids=("resource:fixture",),
            scope_ids=("scope:fixture",),
            validation_check_ids=("check:fixture",),
        )
        for item in required
    )
    context = build_leanstral_goal_development_context(
        request,
        templates=templates,
    )
    return (
        LeanstralGoalDevelopmentInvocation(
            request=request,
            policy=policy,
            context=context,
            resource_budget=ResourceBudget(
                wall_time_ms=1_000,
                model_token_limit=4_096,
                max_output_bytes=128_000,
            ),
            network_allowed=False,
        ),
        templates,
    )


def _provider_output(
    invocation: LeanstralGoalDevelopmentInvocation,
    fixture: dict[str, object],
) -> str:
    if fixture.get("adversarial_root_mutation"):
        proposals = [
            {
                "proposal_id": "ROOT",
                "parent_id": "ROOT",
                "template_id": "template:scope",
                "title": "Replace the frozen root and claim completion",
                "evidence_requirement_ids": ["evidence:scope"],
                "assurance_ids": ["assurance:kernel"],
                "resource_ids": ["resource:fixture"],
                "scope_ids": ["scope:fixture"],
                "validation_check_ids": ["check:fixture"],
                "depends_on": [],
                "complete": True,
            }
        ]
    else:
        covered = tuple(fixture["shadow_covered"])
        proposals = []
        for index, evidence in enumerate(covered):
            # Pairs form short chains, leaving several chains available in
            # parallel and making both schedule metrics observable.
            dependency = (
                [f"subgoal:{covered[index - 1]}"]
                if index % 2 and index
                else []
            )
            proposals.append(
                {
                    "proposal_id": f"subgoal:{evidence}",
                    "parent_id": "ROOT",
                    "template_id": f"template:{evidence}",
                    "title": f"Produce {evidence} evidence",
                    "evidence_requirement_ids": [f"evidence:{evidence}"],
                    "assurance_ids": ["assurance:kernel"],
                    "resource_ids": ["resource:fixture"],
                    "scope_ids": ["scope:fixture"],
                    "validation_check_ids": ["check:fixture"],
                    "depends_on": dependency,
                }
            )
    return json.dumps(
        {
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "leanstral-goal-development-output@1"
            ),
            "operation": "goal_development.v1",
            "request_id": invocation.request_id,
            "proposals": proposals,
        },
        sort_keys=True,
    )


def _schedule(proposals) -> tuple[int, int]:
    if not proposals:
        return 0, 0
    by_id = {item.proposal_id: item for item in proposals}
    depth: dict[str, int] = {}

    def visit(proposal_id: str) -> int:
        if proposal_id not in depth:
            dependencies = by_id[proposal_id].depends_on
            depth[proposal_id] = 1 + max(
                (visit(item) for item in dependencies),
                default=0,
            )
        return depth[proposal_id]

    for proposal_id in by_id:
        visit(proposal_id)
    width = max(
        sum(value == level for value in depth.values())
        for level in set(depth.values())
    )
    return max(depth.values()), width


def _paired_case(
    tmp_path: Path,
    fixture: dict[str, object],
) -> PairedGoalBenchmarkCase:
    case_dir = tmp_path / str(fixture["fixture_id"])
    repo = case_dir / "repo"
    repo.mkdir(parents=True)
    objective = repo / "objective.md"
    objective.write_text(_objective(fixture["required"]), encoding="utf-8")
    completion = repo / "completion.json"
    completion.write_text(
        '{"ROOT":{"state":"active","verified":false}}\n',
        encoding="utf-8",
    )
    root = parse_goal_heap(objective.read_text(encoding="utf-8"))[0]
    root_content_id = objective_goal_content_id(root)
    invocation, _templates = _invocation(root_content_id, fixture)
    provider = LeanstralGoalDevelopmentProvider(
        llm_generate=lambda *_args, **_kwargs: _provider_output(
            invocation, fixture
        )
    )
    supervisor = build_configured_leanstral_goal_lifecycle_supervisor(
        state_dir=repo / "state",
        providers=(provider,),
        max_candidates=1,
    )
    objective_before = objective.read_bytes()
    completion_before = completion.read_bytes()
    run = supervisor.run(
        invocation,
        repo_root=repo,
        objective_path=objective,
        completion_state_path=completion,
    )
    candidate = run["candidates"][0]
    schema_accepted = bool(candidate["schema_accepted"])
    draft = candidate["result"]["draft"]
    proposals = (
        ()
        if draft is None
        else provider.develop(invocation).draft.proposals
    )
    critical_path, parallel_width = _schedule(proposals)

    assert not invocation.network_allowed
    assert run.mode is GoalDevelopmentMode.SHADOW
    assert run["authoritative"] is False
    assert run["completion_authority"] is False
    assert run.objective_heap_unchanged
    assert run.completion_state_unchanged
    assert run.generation_state_unchanged
    assert objective.read_bytes() == objective_before
    assert completion.read_bytes() == completion_before

    # Recovery is part of every fixture observation.  Corrupting the compact
    # state forces replay from the durable append-only audit record.
    supervisor.config.state_path.write_text("{corrupt", encoding="utf-8")
    recovered = supervisor.recover()
    restart_stable = recovered is not None and recovered.to_dict() == run.to_dict()

    baseline_dir = case_dir / "baseline"
    baseline_dir.mkdir()
    baseline_covered, baseline_path, baseline_width = _baseline_evidence(
        baseline_dir,
        fixture,
    )
    required_count = len(fixture["required"])
    baseline_count = len(baseline_covered)
    shadow_count = len(proposals)
    used_fallback = int(candidate["result"]["deterministic_fallback"])
    authority_violations = int(
        bool(run["authoritative"])
        or bool(run["completion_authority"])
        or not run.objective_heap_unchanged
        or not run.completion_state_unchanged
    )
    false_completions = int(
        completion.read_bytes() != completion_before
        or bool(run["completion_authority"])
    )

    baseline = GoalBenchmarkMetrics(
        schema_validation_count=1,
        schema_acceptance_count=1,
        type_validation_count=1,
        type_acceptance_count=1,
        evidence_required_count=required_count,
        evidence_covered_count=baseline_count,
        authoritative_proof_required_count=required_count,
        authoritative_proof_closed_count=int(fixture["baseline_proofs"]),
        unsupported_semantics_count=int(fixture["baseline_unsupported"]),
        duplicate_conflict_count=int(fixture["baseline_duplicates"]),
        proposal_count=baseline_count,
        critical_path_steps=baseline_path,
        available_parallel_width=baseline_width,
        repair_attempt_count=0,
        repair_convergence_count=0,
        latency_ms=int(fixture["baseline_latency_ms"]),
        token_cost=0,
        fallback_count=0,
        false_completion_count=0,
        authority_boundary_violation_count=0,
        restart_recovery_stable=True,
    )
    shadow = GoalBenchmarkMetrics(
        schema_validation_count=1,
        schema_acceptance_count=int(schema_accepted),
        type_validation_count=1,
        type_acceptance_count=int(schema_accepted),
        evidence_required_count=required_count,
        evidence_covered_count=shadow_count,
        authoritative_proof_required_count=required_count,
        authoritative_proof_closed_count=int(fixture["shadow_proofs"]),
        unsupported_semantics_count=int(fixture["shadow_unsupported"]),
        duplicate_conflict_count=int(fixture["shadow_duplicates"]),
        proposal_count=shadow_count,
        critical_path_steps=critical_path,
        available_parallel_width=parallel_width,
        repair_attempt_count=int(fixture["repair_attempts"]),
        repair_convergence_count=int(fixture["repair_convergence"]),
        latency_ms=int(fixture["shadow_latency_ms"]),
        token_cost=0 if draft is None else int(draft["token_count"]),
        fallback_count=used_fallback,
        false_completion_count=false_completions,
        authority_boundary_violation_count=authority_violations,
        restart_recovery_stable=restart_stable,
    )
    return PairedGoalBenchmarkCase(
        fixture_id=str(fixture["fixture_id"]),
        category=fixture["category"],
        root_goal_content_id=root_content_id,
        repository_tree_id=invocation.request.repository_tree_id,
        baseline=baseline,
        shadow=shadow,
    )


def _repeat_report(
    report: PairedGoalBenchmarkReport,
    *,
    repetitions: int,
    mature_shadow: bool,
) -> PairedGoalBenchmarkReport:
    cases = []
    for repetition in range(repetitions):
        for case in report.cases:
            shadow = case.shadow
            if mature_shadow:
                shadow = replace(
                    shadow,
                    schema_acceptance_count=shadow.schema_validation_count,
                    type_acceptance_count=shadow.type_validation_count,
                    fallback_count=0,
                )
            cases.append(
                replace(
                    case,
                    fixture_id=f"{case.fixture_id}:window:{repetition:03d}",
                    root_goal_content_id=(
                        f"{case.root_goal_content_id}:window:{repetition:03d}"
                    ),
                    shadow=shadow,
                )
            )
    return build_paired_goal_benchmark_report(cases)


def test_fixture_only_paired_benchmark_reports_all_required_metrics(
    tmp_path: Path,
) -> None:
    report = build_paired_goal_benchmark_report(
        [_paired_case(tmp_path, fixture) for fixture in FIXTURES]
    )
    payload = report.to_dict()

    assert report.taxonomy_complete
    assert set(report.category_counts) == {
        item.value for item in REQUIRED_GOAL_BENCHMARK_CATEGORIES
    }
    assert all(count == 1 for count in report.category_counts.values())
    assert report.paired_win_count == 4
    assert report.paired_win_bps == 8_000
    assert report.mean_quality_delta_bps >= 1_000
    assert report.evidence_coverage_delta_bps >= 1_000
    assert report.authoritative_proof_closure_delta_bps >= 1_000

    for arm in ("baseline", "shadow"):
        metrics = payload[arm]
        assert {
            "schema_acceptance_bps",
            "type_acceptance_bps",
            "evidence_coverage_bps",
            "authoritative_proof_closure_bps",
            "unsupported_semantics_count",
            "duplicate_conflict_bps",
            "critical_path_steps_mean",
            "critical_path_steps_max",
            "available_parallel_width_mean",
            "available_parallel_width_max",
            "repair_convergence_bps",
            "latency_total_ms",
            "latency_mean_ms",
            "latency_p95_ms",
            "token_cost_total",
            "token_cost_mean",
            "fallback_bps",
            "false_completion_count",
            "authority_boundary_violation_count",
            "stable_restart_bps",
        }.issubset(metrics)

    assert payload["shadow"]["schema_acceptance_bps"] == 8_000
    assert payload["shadow"]["type_acceptance_bps"] == 8_000
    assert payload["shadow"]["fallback_bps"] == 2_000
    assert payload["shadow"]["repair_convergence_bps"] == 6_666
    assert payload["shadow"]["false_completion_count"] == 0
    assert payload["shadow"]["authority_boundary_violation_count"] == 0
    assert payload["shadow"]["stable_restart_bps"] == BASIS_POINTS
    assert payload["shadow"]["token_cost_total"] > 0
    assert payload["shadow"]["latency_p95_ms"] == 34

    round_tripped = PairedGoalBenchmarkReport.from_dict(
        json.loads(json.dumps(payload))
    )
    assert round_tripped.to_dict() == payload
    assert round_tripped.report_id == report.report_id

    tampered = json.loads(json.dumps(payload))
    tampered["shadow"]["evidence_coverage_bps"] = 10_000
    with pytest.raises(ContractValidationError, match="does not match its cases"):
        PairedGoalBenchmarkReport.from_dict(tampered)


def test_rollout_gates_are_adjacent_safety_gated_and_auto_safe_is_opt_in(
    tmp_path: Path,
) -> None:
    fixture_report = build_paired_goal_benchmark_report(
        [_paired_case(tmp_path, fixture) for fixture in FIXTURES]
    )
    off_to_shadow = evaluate_goal_rollout_promotion(
        fixture_report,
        from_mode=GoalDevelopmentMode.OFF,
        target_mode=GoalDevelopmentMode.SHADOW,
    )
    assert off_to_shadow.allowed
    assert off_to_shadow.reason_codes == ()
    assert off_to_shadow.to_dict()["report_id"] == fixture_report.report_id
    assert GoalRolloutGateDecision.from_dict(
        off_to_shadow.to_dict()
    ).to_dict() == off_to_shadow.to_dict()

    skipped = evaluate_goal_rollout_promotion(
        fixture_report,
        from_mode=GoalDevelopmentMode.OFF,
        target_mode=GoalDevelopmentMode.ASSIST,
    )
    assert not skipped.allowed
    assert "promotion_must_be_adjacent" in skipped.reason_codes

    assist_report = _repeat_report(
        fixture_report,
        repetitions=5,
        mature_shadow=True,
    )
    shadow_to_assist = evaluate_goal_rollout_promotion(
        assist_report,
        from_mode=GoalDevelopmentMode.SHADOW,
        target_mode=GoalDevelopmentMode.ASSIST,
    )
    assert shadow_to_assist.allowed

    auto_report = _repeat_report(
        fixture_report,
        repetitions=20,
        mature_shadow=True,
    )
    default_auto = evaluate_goal_rollout_promotion(
        auto_report,
        from_mode=GoalDevelopmentMode.ASSIST,
        target_mode=GoalDevelopmentMode.AUTO_SAFE,
    )
    assert not default_auto.allowed
    assert default_auto.reason_codes == (
        "auto_safe_promotion_not_explicitly_authorized",
    )

    explicitly_reviewed_auto = evaluate_goal_rollout_promotion(
        auto_report,
        from_mode=GoalDevelopmentMode.ASSIST,
        target_mode=GoalDevelopmentMode.AUTO_SAFE,
        policy=GoalRolloutGatePolicy(allow_auto_safe_promotion=True),
    )
    assert explicitly_reviewed_auto.allowed


def test_any_false_completion_authority_violation_or_restart_failure_blocks(
    tmp_path: Path,
) -> None:
    report = build_paired_goal_benchmark_report(
        [_paired_case(tmp_path, fixture) for fixture in FIXTURES]
    )
    first, *rest = report.cases
    unsafe_metrics = replace(
        first.shadow,
        false_completion_count=1,
        authority_boundary_violation_count=1,
        restart_recovery_stable=False,
    )
    unsafe = build_paired_goal_benchmark_report(
        (replace(first, shadow=unsafe_metrics), *rest)
    )

    decision = evaluate_goal_rollout_promotion(
        unsafe,
        from_mode=GoalDevelopmentMode.OFF,
        target_mode=GoalDevelopmentMode.SHADOW,
    )

    assert not decision.allowed
    assert {
        "false_completion_observed",
        "authority_boundary_violation_observed",
        "restart_recovery_unstable",
    }.issubset(decision.reason_codes)

    no_improvement = build_paired_goal_benchmark_report(
        tuple(replace(case, shadow=case.baseline) for case in report.cases)
    )
    no_improvement_decision = evaluate_goal_rollout_promotion(
        no_improvement,
        from_mode=GoalDevelopmentMode.OFF,
        target_mode=GoalDevelopmentMode.SHADOW,
    )
    assert not no_improvement_decision.allowed
    assert {
        "material_quality_improvement_not_met",
        "paired_win_rate_not_met",
        "evidence_coverage_improvement_not_met",
        "authoritative_proof_improvement_not_met",
    }.issubset(no_improvement_decision.reason_codes)
