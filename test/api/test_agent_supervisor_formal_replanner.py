from __future__ import annotations

import json

import pytest

from ipfs_accelerate_py.agent_supervisor.formal_counterexamples import (
    CounterexampleKind,
    RepairClass,
    normalize_counterexample,
    normalize_tdfol_contradiction,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_compiler import (
    FormalPlanCompiler,
    compile_formal_plan,
)
from ipfs_accelerate_py.agent_supervisor.formal_plan_validator import (
    FormalPlanValidator,
)
from ipfs_accelerate_py.agent_supervisor.formal_replanner import (
    BOUNDED_REFINEMENT_EVIDENCE_ID,
    CODEX_REPAIR_PACKET_SCHEMA,
    REPAIR_TRANSITION_SCHEMA,
    RESPONSIVE_REPLAN_DECISION_SCHEMA,
    CodexRepairPacket,
    FormalReplanner,
    RepairCandidateStatus,
    RepairOperation,
    RepairRuleKind,
    RepairTransition,
    ReplanLimits,
    ReplanStopReason,
    ReplannerValidationError,
    ResponsiveReplanDecision,
)


def _source() -> dict[str, object]:
    return {
        "repository_tree_id": "tree:repair",
        "objectives": [
            {
                "goal_id": "G12.S4",
                "goal_cid": "goal:cid:g12-s4",
                "acceptance_criteria": ["The intended transition is evidenced."],
            }
        ],
        "tasks": [
            {
                "task_id": "REF-BASE",
                "task_cid": "task:cid:base",
                "goal_id": "G12.S4",
                "actor_id": "agent:base",
                "changed_ast_scopes": ["symbol:base"],
                "acceptance_criteria": ["base test"],
            },
            {
                "task_id": "REF-TARGET",
                "task_cid": "task:cid:target",
                "goal_id": "G12.S4",
                "actor_id": "agent:target",
                "changed_ast_scopes": ["symbol:target", "symbol:unrelated"],
                "effects": [
                    {
                        "operation": "assign",
                        "fluent_id": "target:built",
                        "value": True,
                    },
                    {
                        "operation": "assign",
                        "fluent_id": "target:tested",
                        "value": True,
                    },
                ],
                "acceptance_criteria": ["target test"],
            },
        ],
        "policies": [
            {
                "policy_id": "policy:formal-repair",
                "fallback_checks": ["pytest baseline.py"],
            }
        ],
    }


def _counterexample(
    source: dict[str, object] | None = None,
    *,
    repair_classes: tuple[RepairClass, ...] = (RepairClass.HUMAN_REVIEW,),
):
    active_source = source or _source()
    compiled = compile_formal_plan(active_source)
    assert compiled.plan is not None
    return normalize_counterexample(
        {
            "kind": CounterexampleKind.GENERIC_FAILURE.value,
            "failure": {"code": "focused-repair-required"},
        },
        kind=CounterexampleKind.GENERIC_FAILURE,
        violated_property="goal transition must remain valid",
        bindings={
            "plan_id": compiled.plan_id,
            "task_id": "REF-TARGET",
            "ast_scope_id": "symbol:target",
        },
        repair_classes=repair_classes,
    )


def _operation(kind: RepairRuleKind, counterexample_id: str) -> RepairOperation:
    parameters = {
        RepairRuleKind.ADD_DEPENDENCY: {
            "dependency_task_id": "REF-BASE",
        },
        RepairRuleKind.SPLIT_EFFECTS: {
            "split_index": 1,
            "generated_task_id": "task:cid:target:part-2",
        },
        RepairRuleKind.TIGHTEN_AUTHORITY: {
            "actor_ids": ["agent:target"],
            "fencing_token": 9,
        },
        RepairRuleKind.ADD_EVIDENCE: {
            "evidence_kind": "test",
            "check_ids": ["pytest targeted_regression.py"],
        },
        RepairRuleKind.CONSTRAIN_SCOPE: {
            "scope_ids": ["symbol:target"],
        },
        RepairRuleKind.ADD_PREMISE: {
            "premise_ids": ["premise:dependency-ready"],
        },
        RepairRuleKind.CHANGE_RESOURCE_BOUNDS: {
            "resource_bounds": {
                "cpu": 1,
                "portfolio_width": 1,
                "deadline": 20,
            },
        },
        RepairRuleKind.HUMAN_REVIEW: {
            "reviewer_actor_id": "human:formal-reviewer",
            "scope_ids": ["symbol:target"],
            "question": "Confirm the intended effect semantics.",
        },
    }[kind]
    return RepairOperation(
        kind=kind,
        target_task_id="REF-TARGET",
        parameters=parameters,
        counterexample_id=counterexample_id,
    )


@pytest.mark.parametrize("kind", tuple(RepairRuleKind))
def test_every_typed_repair_rule_recompiles_checks_and_preserves_goal(kind) -> None:
    source = _source()
    counterexample = _counterexample(source)
    operation = _operation(kind, counterexample.semantic_id)

    result = FormalReplanner().replan(
        source,
        counterexample,
        candidate_repairs=(operation,),
    )

    assert result.stop_reason is ReplanStopReason.ADMITTED
    assert result.admitted
    assert result.selected is not None
    assert result.selected.status is RepairCandidateStatus.ADMITTED
    assert result.selected.compilation is not None
    assert result.selected.compilation.plan is not None
    assert result.selected.validation is not None
    assert result.selected.validation.consistent
    assert {
        goal.goal_id for goal in result.selected.compilation.plan.goals
    } == {"goal:cid:g12-s4"}
    transition = result.selected_transition
    assert transition is not None
    assert transition.repair.kind is kind
    assert transition.progress.improved
    assert transition.progress.changed_records <= ReplanLimits().max_changed_records
    assert transition.taskboard_records
    assert RepairTransition.from_dict(transition.to_dict()) == transition
    assert transition.to_dict()["schema"] == REPAIR_TRANSITION_SCHEMA


def test_repair_transition_deserialization_fails_closed() -> None:
    source = _source()
    counterexample = _counterexample(source)
    result = FormalReplanner().replan(
        source,
        counterexample,
        candidate_repairs=(
            _operation(
                RepairRuleKind.ADD_DEPENDENCY,
                counterexample.semantic_id,
            ),
        ),
    )
    transition = result.selected_transition
    assert transition is not None

    unsupported = transition.to_dict()
    unsupported["replanner_version"] += 1
    with pytest.raises(
        ReplannerValidationError, match="replanner version"
    ):
        RepairTransition.from_dict(unsupported)
    inconsistent = transition.to_dict()
    inconsistent.pop("transition_id")
    inconsistent["progress"]["improved"] = not transition.progress.improved
    with pytest.raises(
        ReplannerValidationError, match="progress projection"
    ):
        RepairTransition.from_dict(inconsistent)
    unknown = transition.to_dict()
    unknown["untrusted_metadata"] = {"claim": "safe"}
    with pytest.raises(
        ReplannerValidationError, match="unknown repair transition fields"
    ):
        RepairTransition.from_dict(unknown)


def test_counterexample_classes_generate_focused_dependency_and_premise_repairs() -> None:
    source = _source()
    compiled = compile_formal_plan(source)
    assert compiled.plan is not None
    counterexample = normalize_tdfol_contradiction(
        {
            "contradiction": {
                "dependency_task_id": "REF-BASE",
                "formula": "started before dependency",
            }
        },
        violated_property="dependencies complete before task start",
        bindings={
            "plan_id": compiled.plan_id,
            "task_id": "REF-TARGET",
        },
    )
    replanner = FormalReplanner()

    operations = replanner.generate_repairs(source, counterexample)
    result = replanner.replan(source, counterexample)

    assert {item.kind for item in operations} == {
        RepairRuleKind.ADD_DEPENDENCY,
        RepairRuleKind.ADD_PREMISE,
    }
    assert result.admitted
    assert result.selected_transition is not None
    assert result.selected_transition.repair.target_task_id == "REF-TARGET"
    assert len(result.candidates) == len(operations)
    assert all(item.compilation is not None for item in result.candidates)
    assert all(item.validation is not None for item in result.candidates)


class _CountingCompiler(FormalPlanCompiler):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def compile(self, *args, **kwargs):
        self.calls += 1
        return super().compile(*args, **kwargs)


class _CountingValidator(FormalPlanValidator):
    def __init__(self) -> None:
        super().__init__()
        self.calls = 0

    def validate(self, *args, **kwargs):
        self.calls += 1
        return super().validate(*args, **kwargs)


class _CountingReplanner(FormalReplanner):
    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self.replan_calls = 0
        self.generate_calls = 0

    def replan(self, *args, **kwargs):
        self.replan_calls += 1
        return super().replan(*args, **kwargs)

    def generate_repairs(self, *args, **kwargs):
        self.generate_calls += 1
        return super().generate_repairs(*args, **kwargs)


def test_all_candidates_are_compiled_and_checked_before_one_taskboard_admission() -> None:
    source = _source()
    counterexample = _counterexample(source)
    compiler = _CountingCompiler()
    validator = _CountingValidator()
    admitted: list[RepairTransition] = []
    operations = (
        _operation(RepairRuleKind.ADD_DEPENDENCY, counterexample.semantic_id),
        _operation(RepairRuleKind.ADD_EVIDENCE, counterexample.semantic_id),
        _operation(RepairRuleKind.CONSTRAIN_SCOPE, counterexample.semantic_id),
    )

    result = FormalReplanner(
        compiler=compiler,
        validator=validator,
        admission_callback=lambda transition: admitted.append(transition),
    ).replan(source, counterexample, candidate_repairs=operations)

    assert result.admitted
    assert compiler.calls == 1 + len(operations)
    assert validator.calls == 1 + len(operations)
    assert len(admitted) == 1
    assert admitted[0] == result.selected_transition
    assert all(item.validation and item.validation.consistent for item in result.candidates)


def test_semantic_dedup_retry_and_refinement_budgets_stop_infinite_repair() -> None:
    source = _source()
    counterexample = _counterexample(source)
    operation = _operation(RepairRuleKind.ADD_EVIDENCE, counterexample.semantic_id)
    replanner = FormalReplanner(
        limits=ReplanLimits(max_retry_attempts=2, max_refinement_depth=2)
    )

    first = replanner.replan(
        source, counterexample, candidate_repairs=(operation,)
    )
    duplicate = replanner.replan(
        source, counterexample, candidate_repairs=(operation,)
    )
    exhausted = replanner.replan(
        source, counterexample, candidate_repairs=(operation,)
    )
    too_deep = FormalReplanner(
        limits=ReplanLimits(max_refinement_depth=1)
    ).replan(
        source,
        counterexample,
        candidate_repairs=(operation,),
        refinement_depth=1,
    )

    assert first.admitted
    assert duplicate.candidates[0].status is RepairCandidateStatus.DUPLICATE
    assert not duplicate.admitted
    assert exhausted.stop_reason is ReplanStopReason.RETRY_BUDGET_EXHAUSTED
    assert too_deep.stop_reason is ReplanStopReason.REFINEMENT_DEPTH_EXHAUSTED
    assert operation.semantic_id in replanner.seen_semantic_ids


def test_mismatched_counterexample_never_reaches_admission() -> None:
    source = _source()
    counterexample = _counterexample(source)
    rebound = type(counterexample)(
        kind=counterexample.kind,
        property_class=counterexample.property_class,
        violated_property=counterexample.violated_property,
        summary=counterexample.summary,
        payload=counterexample.payload,
        bindings=type(counterexample.bindings)(
            plan_ids=("plan:some-other-tree",),
            task_ids=counterexample.bindings.task_ids,
            ast_scope_ids=counterexample.bindings.ast_scope_ids,
        ),
        assumption_ids=counterexample.assumption_ids,
        finite_bounds=counterexample.finite_bounds,
        repair_classes=counterexample.repair_classes,
    )
    admitted: list[RepairTransition] = []

    result = FormalReplanner(
        admission_callback=lambda transition: admitted.append(transition)
    ).replan(source, rebound)

    assert result.stop_reason is ReplanStopReason.COUNTEREXAMPLE_PLAN_MISMATCH
    assert not result.candidates
    assert not admitted


def test_codex_receives_only_selected_transition_and_bounded_redacted_capsule() -> None:
    source = _source()
    compiled = compile_formal_plan(source)
    assert compiled.plan is not None
    counterexample = normalize_counterexample(
        {
            "kind": "generic_failure",
            "failure": {"code": "semantic ambiguity"},
            "api_key": "must-never-enter-the-prompt",
            "raw_output": "repository-wide transcript",
        },
        kind=CounterexampleKind.GENERIC_FAILURE,
        violated_property="review required",
        bindings={
            "plan_id": compiled.plan_id,
            "task_id": "REF-TARGET",
            "ast_scope_id": "symbol:target",
        },
        repair_classes=(RepairClass.HUMAN_REVIEW,),
    )

    result = FormalReplanner().replan(source, counterexample)

    assert result.admitted
    packet = result.codex_packet
    assert isinstance(packet, CodexRepairPacket)
    record = packet.to_dict()
    encoded = json.dumps(record, sort_keys=True)
    assert record["schema"] == CODEX_REPAIR_PACKET_SCHEMA
    assert set(record) == {
        "schema",
        "replanner_version",
        "transition",
        "counterexample_capsule",
        "limits",
    }
    assert len(record["counterexample_capsule"]["counterexamples"]) == 1
    assert record["counterexample_capsule"]["counterexamples"][0][
        "counterexample_id"
    ] == counterexample.semantic_id
    assert packet.byte_size <= packet.max_bytes
    assert packet.estimated_tokens <= packet.max_tokens
    assert "must-never-enter-the-prompt" not in encoded
    assert "repository-wide transcript" not in encoded
    assert "original_compilation" not in encoded
    assert "rejection_reasons" not in encoded
    assert '"objectives":' not in encoded
    assert '"tasks":' not in encoded


def test_prompt_budget_is_enforced_before_taskboard_callback() -> None:
    source = _source()
    counterexample = _counterexample(source)
    admitted: list[RepairTransition] = []
    operation = _operation(RepairRuleKind.ADD_EVIDENCE, counterexample.semantic_id)

    result = FormalReplanner(
        limits=ReplanLimits(
            max_capsule_bytes=1024,
            max_prompt_bytes=1024,
            max_prompt_tokens=256,
        ),
        admission_callback=lambda transition: admitted.append(transition),
    ).replan(source, counterexample, candidate_repairs=(operation,))

    assert not result.admitted
    assert not admitted
    assert result.codex_packet is None
    assert result.selected is not None
    assert result.selected.status is RepairCandidateStatus.ADMISSION_REJECTED
    assert "packet rejected" in result.selected.rejection_reasons[-1]


def test_unchanged_counterexample_backs_off_before_compile_or_generation() -> None:
    source = _source()
    counterexample = _counterexample(source)
    compiler = _CountingCompiler()
    validator = _CountingValidator()
    replanner = _CountingReplanner(compiler=compiler, validator=validator)

    decision = replanner.replan_if_changed(
        source,
        counterexample,
        previous_counterexample_id=counterexample.semantic_id,
        backoff_attempt=4,
        base_backoff_seconds=2,
        max_backoff_seconds=20,
    )

    assert isinstance(decision, ResponsiveReplanDecision)
    assert decision.changed is False
    assert decision.refined is False
    assert decision.model_call_required is False
    assert decision.result is None
    assert (
        decision.stop_reason
        is ReplanStopReason.UNCHANGED_COUNTEREXAMPLE_BACKOFF
    )
    assert decision.backoff_attempt == 5
    assert decision.backoff_seconds == 20
    assert compiler.calls == validator.calls == 0
    assert replanner.replan_calls == replanner.generate_calls == 0
    payload = decision.to_dict()
    assert payload["schema"] == RESPONSIVE_REPLAN_DECISION_SCHEMA
    assert payload["evidence_ids"] == []
    assert payload["requirement_ids"] == [
        BOUNDED_REFINEMENT_EVIDENCE_ID
    ]
    assert BOUNDED_REFINEMENT_EVIDENCE_ID == (
        "003778425160038348524906247302938706902"
    )


def test_changed_counterexample_delegates_once_and_preserves_frozen_root() -> None:
    source = _source()
    frozen_source = json.dumps(source, sort_keys=True)
    counterexample = _counterexample(source)
    compiler = _CountingCompiler()
    replanner = _CountingReplanner(compiler=compiler)

    decision = replanner.replan_if_changed(
        source,
        counterexample,
        previous_counterexample_id="sha256:previous-counterexample",
    )

    assert decision.changed is True
    assert decision.refined is True
    assert decision.model_call_required is True
    # The wrapper routes the requirement; only the durable adaptive-refiner
    # receipt is an authoritative objective evidence producer.
    assert decision.evidence_ids == ()
    assert decision.to_dict()["requirement_ids"] == [
        BOUNDED_REFINEMENT_EVIDENCE_ID
    ]
    assert decision.result is not None
    assert decision.result.admitted
    assert decision.stop_reason is ReplanStopReason.ADMITTED
    assert decision.backoff_attempt == decision.backoff_seconds == 0
    assert replanner.replan_calls == 1
    assert replanner.generate_calls == 1
    assert compiler.calls == 1 + len(decision.result.candidates)
    assert json.dumps(source, sort_keys=True) == frozen_source
    transition = decision.result.selected_transition
    assert transition is not None
    assert set(transition.goal_ids) == {"goal:cid:g12-s4"}


def test_changed_but_unadmitted_replan_does_not_emit_objective_evidence() -> None:
    source = _source()
    counterexample = _counterexample(source)

    decision = FormalReplanner().replan_if_changed(
        source,
        counterexample,
        previous_counterexample_id="sha256:different",
        candidate_repairs=(),
    )

    assert decision.changed is True
    assert decision.refined is False
    assert decision.result is not None
    assert not decision.result.admitted
    assert decision.evidence_ids == ()
    assert decision.to_dict()["evidence_ids"] == []
