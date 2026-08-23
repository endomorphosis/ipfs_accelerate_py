from __future__ import annotations

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    ArtifactState,
    EffectClass,
    ProcedureContractError,
    RiskClass,
    TaskFamily,
    TaskFamilyBoundary,
    parse_procedure_artifact,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.experiments import (
    REQUIRED_DECLARATION_FIELDS,
    REQUIRED_PRIVACY_CLASSES,
    DecisionRule,
    DecisionRuleClass,
    ExecutionBound,
    ExperimentAction,
    ExperimentCost,
    ExperimentDeclarationError,
    ExperimentDecision,
    ExperimentEffectClass,
    ExperimentError,
    ExperimentIsolationError,
    ExperimentObservationError,
    ExperimentObservationRecord,
    ExperimentOutcome,
    ExperimentPlanner,
    ExperimentReason,
    IsolationKind,
    IsolationTarget,
    ObservationUse,
    PendingDecision,
    PrivacyClass,
    ShadowExperiment,
    ShadowExperimentRunner,
    UncertaintyQuestion,
    extract_uncertainty_questions,
    observation_may_discharge,
    value_of_experiment,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.world_model import (
    RepositoryWorldState,
    WorldProjectionStatus,
)


def _bindings() -> ArtifactBindings:
    return ArtifactBindings(
        repository_id="repo",
        repository_commit="commit-1",
        tree_id="tree-1",
        objective_id="PCPC-G000",
        task_id="PCPC-024",
        contract_revision="contract-1",
        policy_revision="policy-1",
        environment_id="environment-1",
    )


def _privacy() -> tuple[PrivacyClass, ...]:
    return tuple(PrivacyClass(item) for item in REQUIRED_PRIVACY_CLASSES)


def _rule(**changes: object) -> DecisionRule:
    values: dict[str, object] = {
        "rule_class": DecisionRuleClass.DISTINGUISH_OUTCOMES,
        "observation_binding": "proof_status_current",
        "hypothesis_option_id": "hold-shadow",
        "counterfactual_option_id": "advance-candidate",
        "hypothesis_operand": False,
        "counterfactual_operand": True,
    }
    values.update(changes)
    return DecisionRule(**values)


def _bound(**changes: object) -> ExecutionBound:
    values: dict[str, object] = {
        "max_tokens": 128,
        "max_duration_ms": 1_000,
        "max_provider_cost_micros": 0,
        "max_worktrees": 0,
        "max_output_bytes": 4_096,
        "max_steps": 4,
    }
    values.update(changes)
    return ExecutionBound(**values)


def _cost(**changes: object) -> ExperimentCost:
    values: dict[str, object] = {
        "tokens": 32,
        "duration_ms": 50,
        "provider_cost_micros": 0,
        "worktree_count": 0,
    }
    values.update(changes)
    return ExperimentCost(**values)


def _fixture_isolation(**changes: object) -> IsolationTarget:
    values: dict[str, object] = {
        "kind": IsolationKind.FIXTURE,
        "target_id": "fixture-proof-status",
        "repository_id": "repo",
        "tree_id": "tree-1",
        "disposable": True,
        "production": False,
        "policy_mutable": False,
        "authorized": False,
        "scope_paths": ("test/fixtures/procedure_compiler/proof-status.json",),
    }
    values.update(changes)
    return IsolationTarget(**values)


def _worktree_isolation(**changes: object) -> IsolationTarget:
    values: dict[str, object] = {
        "kind": IsolationKind.AUTHORIZED_DISPOSABLE_WORKTREE,
        "target_id": "worktree-shadow-1",
        "repository_id": "repo",
        "tree_id": "tree-1",
        "disposable": True,
        "production": False,
        "policy_mutable": False,
        "authorized": True,
        "admission_receipt_id": "worktree-admission-1",
        "lease_id": "lease-shadow-1",
        "fencing_token": 7,
        "scope_paths": ("ipfs_accelerate_py/agent_supervisor/procedure_compiler",),
    }
    values.update(changes)
    return IsolationTarget(**values)


def _experiment(**changes: object) -> ShadowExperiment:
    values: dict[str, object] = {
        "bindings": _bindings(),
        "experiment_id": "shadow-proof-status",
        "question_id": "world.unavailable.proof_status",
        "question": "Is the current proof-status dimension available?",
        "hypothesis_id": "proof-status-missing",
        "hypothesis": "proof status remains unavailable",
        "counterfactual_id": "proof-status-present",
        "counterfactual": "proof status is independently admitted",
        "required_data_ids": ("proof_status_current",),
        "risk_class": RiskClass.OBSERVATION_ONLY,
        "privacy_classes": _privacy(),
        "cost": _cost(),
        "decision_rule": _rule(),
        "execution_bound": _bound(),
        "isolation": _fixture_isolation(),
        "effects": (ExperimentEffectClass.OBSERVE_FIXTURE,),
        "decision_id": "promote-or-hold",
    }
    values.update(changes)
    return ShadowExperiment(**values)


def _pending(**changes: object) -> PendingDecision:
    values: dict[str, object] = {
        "decision_id": "promote-or-hold",
        "option_ids": ("hold-shadow", "advance-candidate"),
        "question_ids": ("world.unavailable.proof_status",),
        "committed_option_id": "",
    }
    values.update(changes)
    return PendingDecision(**values)


def _world(**changes: object) -> RepositoryWorldState:
    values: dict[str, object] = {
        "bindings": _bindings(),
        "world_snapshot_cid": "sha256:" + "1" * 64,
        "repository_reference": "sha256:" + "2" * 64,
        "repository_snapshot_id": "sca-repository-snapshot:sha256:" + "3" * 64,
        "analysis_head_tree_id": "git-tree-1",
        "analysis_index_tree_id": "git-index-1",
        "changed_files": ("src/a.py",),
        "changed_symbols": ("src.a:run",),
        "package_graph_id": "package-graph-1",
        "import_graph_id": "import-graph-1",
        "dependency_graph_id": "dependency-graph-1",
        "interface_graph_id": "interface-graph-1",
        "effect_graph_id": "effect-graph-1",
        "acceptance_state_id": "acceptance-1",
        "active_task_ids": ("PCPC-024",),
        "task_dependency_ids": ("PCPC-008", "PCPC-011"),
        "task_dependency_state_id": "task-dependencies-1",
        "proof_status_id": "",
        "test_status_id": "test-status-1",
        "capability_state_id": "capability-state-1",
        "provider_capacity_id": "provider-capacity-1",
        "worktree_ids": ("worktree-shadow-1",),
        "lease_ids": ("lease-shadow-1",),
        "merge_queue_id": "merge-queue-1",
        "cache_state_id": "cache-state-1",
        "artifact_pressure_id": "artifact-pressure-1",
        "token_budget_remaining": 10_000,
        "resource_budget_id": "resource-budget-1",
        "known_failure_signature_ids": ("failure-1",),
        "procedure_registry_revision": 4,
        "procedure_registry_id": "registry-4",
        "source_evidence_ids": ("receipt-1",),
        "unavailable_dimensions": ("proof_status",),
        "projection_status": WorldProjectionStatus.INCOMPLETE,
    }
    values.update(changes)
    return RepositoryWorldState(**values)


def _family(**changes: object) -> TaskFamily:
    boundary = TaskFamilyBoundary(
        positive_member_cids=("positive-a",),
        negative_example_cids=("negative-a",),
        boundary_example_cids=("boundary-a",),
        unknown_case_cids=("unknown-a",),
        risk_ceiling=RiskClass.REVERSIBLE_LOCAL,
        permitted_repositories=("repo",),
        permitted_languages=("python",),
        permitted_frameworks=("pytest",),
        permitted_effect_classes=(EffectClass.OBSERVE, EffectClass.VALIDATION),
    )
    values: dict[str, object] = {
        "bindings": _bindings(),
        "name": "IMPORT_PURITY_REPAIR",
        "goal_semantics": ("restore-import-purity",),
        "precondition_shape": ("import-side-effect-observed",),
        "affected_artifact_classes": ("python-source",),
        "effect_classes": (EffectClass.OBSERVE, EffectClass.VALIDATION),
        "required_operation_contracts": ("approved-patch-template@1", "test-runner@1"),
        "validation_structure": ("focused-tests", "postcondition-check"),
        "failure_signatures": ("import-side-effect",),
        "postcondition_shape": ("import-is-pure",),
        "rollback_structure": ("restore-exact-tree",),
        "boundary": boundary,
    }
    values.update(changes)
    return TaskFamily(**values)


def _questions() -> tuple[UncertaintyQuestion, ...]:
    return (
        UncertaintyQuestion(
            question_id="world.unavailable.proof_status",
            source="world-unavailable-dimension",
            dimension="proof_status",
        ),
    )


def _plan(experiment: ShadowExperiment | None = None, **kwargs: object) -> ExperimentDecision:
    planner = ExperimentPlanner()
    return planner.plan(
        experiment or _experiment(),
        pending_decision=kwargs.pop("pending_decision", _pending()),
        questions=kwargs.pop("questions", _questions()),
        **kwargs,
    )


def _walk_has_float(value: object) -> bool:
    if isinstance(value, float):
        return True
    if isinstance(value, dict):
        return any(_walk_has_float(item) for item in value.values())
    if isinstance(value, list):
        return any(_walk_has_float(item) for item in value)
    return False


def test_declaration_requires_question_hypothesis_counterfactual_data_risk_privacy_cost_rule_bounds() -> None:
    experiment = _experiment()
    assert set(REQUIRED_DECLARATION_FIELDS) <= {
        "question",
        "hypothesis",
        "counterfactual",
        "required_data",
        "risk",
        "privacy",
        "cost",
        "decision_rule",
        "execution_bound",
    }
    assert experiment.question
    assert experiment.hypothesis
    assert experiment.counterfactual
    assert experiment.required_data_ids
    assert experiment.risk_class is RiskClass.OBSERVATION_ONLY
    assert {item.value for item in experiment.privacy_classes} == set(REQUIRED_PRIVACY_CLASSES)
    assert experiment.cost.units > 0
    assert experiment.decision_rule.observation_binding in experiment.required_data_ids
    assert experiment.execution_bound.admits(experiment.cost)

    with pytest.raises(ProcedureContractError):
        _experiment(question="")
    with pytest.raises(ProcedureContractError):
        _experiment(hypothesis="")
    with pytest.raises(ProcedureContractError):
        _experiment(counterfactual="")
    with pytest.raises(ProcedureContractError):
        _experiment(required_data_ids=())
    with pytest.raises(ExperimentDeclarationError):
        _experiment(decision_rule=_rule(observation_binding="undeclared_binding"))
    with pytest.raises(ProcedureContractError):
        _experiment(privacy_classes=())
    with pytest.raises(ExperimentDeclarationError):
        ExecutionBound(max_worktrees=2)


def test_decision_value_skips_when_experiment_cannot_change_the_pending_decision() -> None:
    same_option = _experiment(
        decision_rule=_rule(
            hypothesis_option_id="hold-shadow",
            counterfactual_option_id="hold-shadow",
        )
    )
    value, reachable = value_of_experiment(same_option, _pending())
    assert value == 0
    assert reachable == ("hold-shadow",)

    decision = _plan(same_option)
    assert decision.action is ExperimentAction.SKIP
    assert decision.reason_code is ExperimentReason.CANNOT_CHANGE_DECISION
    assert decision.value_of_experiment == 0
    assert decision.can_authorize is False

    committed = _plan(pending_decision=_pending(committed_option_id="hold-shadow"))
    assert committed.action is ExperimentAction.SKIP
    assert committed.reason_code is ExperimentReason.ALREADY_ANSWERED

    unrelated = _plan(
        pending_decision=_pending(question_ids=("family.unknown.unknown-a",)),
    )
    assert unrelated.action is ExperimentAction.SKIP
    assert unrelated.reason_code is ExperimentReason.QUESTION_NOT_RELEVANT


def test_decision_relevant_bounded_experiment_runs_on_a_fixture() -> None:
    planned = ExperimentPlanner().plan_experiment(
        _experiment(),
        pending_decision=_pending(),
        questions=_questions(),
        world=_world(),
        remaining_budget=_cost(tokens=128, duration_ms=1_000),
    )
    decision = planned.decision
    assert decision.action is ExperimentAction.RUN
    assert decision.reason_code is ExperimentReason.DECISION_RELEVANT
    assert decision.value_of_experiment == 1
    assert decision.reachable_option_ids == ("hold-shadow", "advance-candidate")
    assert decision.isolation.kind is IsolationKind.FIXTURE
    assert decision.can_grant_authority is False
    assert not _walk_has_float(decision.to_dict())

    decoded = ExperimentDecision.from_dict(decision.to_dict())
    assert decoded == decision
    assert parse_procedure_artifact(planned.plan_artifact.to_dict()) == planned.plan_artifact
    assert planned.plan_artifact.state is ArtifactState.SHADOW
    assert planned.plan_artifact.facts["can_authorize"] is False


def test_open_questions_come_from_world_uncertainty_and_family_unknown_cases() -> None:
    questions = extract_uncertainty_questions(world=_world(), family=_family())
    ids = {item.question_id for item in questions}
    assert "world.unavailable.proof_status" in ids
    assert "world.status.incomplete" in ids
    assert "family.unknown.unknown-a" in ids

    closed = _plan(questions=(), world=_world(unavailable_dimensions=()))
    assert closed.action is ExperimentAction.SKIP
    assert closed.reason_code is ExperimentReason.QUESTION_NOT_OPEN

    family_experiment = _experiment(
        question_id="family.unknown.unknown-a",
        decision_id="family-boundary",
    )
    family_decision = _plan(
        family_experiment,
        pending_decision=_pending(
            decision_id="family-boundary",
            question_ids=("family.unknown.unknown-a",),
        ),
        questions=(),
        family=_family(),
    )
    assert family_decision.action is ExperimentAction.RUN
    assert family_decision.question_id == "family.unknown.unknown-a"


def test_cost_privacy_risk_and_bounds_are_fail_closed() -> None:
    expensive = _plan(
        remaining_budget=_cost(tokens=1, duration_ms=1),
    )
    assert expensive.action is ExperimentAction.SKIP
    assert expensive.reason_code is ExperimentReason.COST_EXCEEDS_BUDGET
    assert expensive.value_of_experiment == 1

    over_bound = _plan(_experiment(cost=_cost(tokens=10_000)))
    assert over_bound.action is ExperimentAction.REFUSE
    assert over_bound.reason_code is ExperimentReason.COST_EXCEEDS_BOUND

    risky = _plan(_experiment(risk_class=RiskClass.REPOSITORY_WRITE))
    assert risky.action is ExperimentAction.REFUSE
    assert risky.reason_code is ExperimentReason.RISK_CEILING

    private = _plan(_experiment(privacy_classes=(PrivacyClass.NO_SECRETS,)))
    assert private.action is ExperimentAction.REFUSE
    assert private.reason_code is ExperimentReason.PRIVACY_VIOLATION

    with pytest.raises(ProcedureContractError, match="forbidden secret or executable field"):
        _rule(hypothesis_operand={"api_key": "redacted"})
    with pytest.raises(ProcedureContractError):
        _experiment(cost=_cost(tokens=0.5))  # type: ignore[arg-type]


def test_disposable_worktree_runs_and_unauthorized_or_production_targets_are_refused() -> None:
    worktree_experiment = _experiment(
        isolation=_worktree_isolation(),
        effects=(ExperimentEffectClass.OBSERVE_DISPOSABLE_WORKTREE,),
        cost=_cost(worktree_count=1),
        execution_bound=_bound(max_worktrees=1),
        risk_class=RiskClass.REVERSIBLE_LOCAL,
    )
    decision = _plan(worktree_experiment, world=_world())
    assert decision.action is ExperimentAction.RUN
    assert decision.isolation.kind is IsolationKind.AUTHORIZED_DISPOSABLE_WORKTREE
    assert decision.isolation.disposable is True
    assert decision.isolation.production is False

    production = _plan(_experiment(isolation=_fixture_isolation(production=True)))
    assert production.action is ExperimentAction.REFUSE
    assert production.reason_code is ExperimentReason.PRODUCTION_MUTATION

    policy = _plan(_experiment(isolation=_fixture_isolation(policy_mutable=True)))
    assert policy.action is ExperimentAction.REFUSE
    assert policy.reason_code is ExperimentReason.POLICY_MUTATION

    shared = _plan(_experiment(isolation=_worktree_isolation(disposable=False)))
    assert shared.action is ExperimentAction.REFUSE
    assert shared.reason_code is ExperimentReason.NON_DISPOSABLE_WORKTREE

    unauthorized = _plan(
        _experiment(
            isolation=_worktree_isolation(authorized=False, admission_receipt_id=""),
            effects=(ExperimentEffectClass.OBSERVE_DISPOSABLE_WORKTREE,),
            cost=_cost(worktree_count=1),
            execution_bound=_bound(max_worktrees=1),
            risk_class=RiskClass.REVERSIBLE_LOCAL,
        )
    )
    assert unauthorized.action is ExperimentAction.REFUSE
    assert unauthorized.reason_code is ExperimentReason.UNAUTHORIZED_WORKTREE

    mutate = _plan(
        _experiment(effects=(ExperimentEffectClass.MUTATE_PRODUCTION,)),
    )
    assert mutate.action is ExperimentAction.REFUSE
    assert mutate.reason_code is ExperimentReason.FORBIDDEN_EFFECT


def test_runner_persists_observations_that_cannot_authorize() -> None:
    experiment = _experiment()
    planned = ExperimentPlanner().plan_experiment(
        experiment,
        pending_decision=_pending(),
        questions=_questions(),
        world=_world(),
    )
    result = ShadowExperimentRunner().run_experiment(
        planned.decision,
        experiment,
        observed_facts={"proof_status_current": True},
        emitted_at_ms=10,
    )
    observation = result.observation
    assert observation.outcome is ExperimentOutcome.COUNTERFACTUAL
    assert observation.selected_option_id == "advance-candidate"
    assert observation.hypothesis_supported is False
    assert observation.state is ArtifactState.CANDIDATE
    assert observation.can_authorize is False
    assert observation.can_grant_authority is False
    assert observation.can_promote is False
    assert observation.can_establish_completion is False
    assert observation.can_establish_proof is False
    assert observation.can_establish_postcondition is False
    for forbidden in (
        ObservationUse.AUTHORITY,
        ObservationUse.POLICY,
        ObservationUse.PROMOTION,
        ObservationUse.PROOF,
        ObservationUse.POSTCONDITION,
        ObservationUse.COMPLETION,
        ObservationUse.VALIDATION_SUPPRESSION,
        ObservationUse.HUMAN_REVIEW_SUPPRESSION,
    ):
        assert observation.allows_use(forbidden) is False
        assert observation_may_discharge(forbidden) is False
        assert planned.decision.allows_use(forbidden) is False
    assert observation.allows_use(ObservationUse.PLANNING_OBSERVATION)

    artifact = result.observation_artifact
    assert parse_procedure_artifact(artifact.to_dict()) == artifact
    assert artifact.state is ArtifactState.CANDIDATE
    assert artifact.facts["can_authorize"] is False
    assert result.evaluation_artifact.facts["mutated_production"] is False
    assert result.evaluation_artifact.facts["mutated_policy"] is False
    assert result.evaluation_artifact.facts["can_promote"] is False
    assert result.evaluation_artifact.state is ArtifactState.CANDIDATE


def test_runner_refuses_skipped_experiments_and_unsafe_isolation() -> None:
    runner = ShadowExperimentRunner()
    skipped = _plan(
        _experiment(
            decision_rule=_rule(
                hypothesis_option_id="hold-shadow",
                counterfactual_option_id="hold-shadow",
            )
        )
    )
    with pytest.raises(ExperimentIsolationError, match="skipped or refused"):
        runner.run(skipped, _experiment(), observed_facts={"proof_status_current": True})

    production_experiment = _experiment(isolation=_fixture_isolation(production=True))
    refused = _plan(production_experiment)
    with pytest.raises(ExperimentIsolationError, match="production-mutation"):
        runner.run(refused, production_experiment, observed_facts={"proof_status_current": True})

    decision = _plan()
    with pytest.raises(ExperimentObservationError, match="required experiment data"):
        runner.run(decision, _experiment(), observed_facts={})
    with pytest.raises(ExperimentError, match="forbidden privacy field"):
        runner.run(
            decision,
            _experiment(),
            observed_facts={"proof_status_current": True, "private_prompt": "hidden"},
        )


def test_integer_threshold_and_closed_membership_rules_stay_integer_and_bounded() -> None:
    threshold = _experiment(
        required_data_ids=("coverage_count",),
        decision_rule=_rule(
            rule_class=DecisionRuleClass.INTEGER_THRESHOLD,
            observation_binding="coverage_count",
            hypothesis_operand=2,
            counterfactual_operand=0,
        ),
    )
    decision = _plan(threshold)
    result = ShadowExperimentRunner().run_experiment(
        decision,
        threshold,
        observed_facts={"coverage_count": 3},
    )
    assert result.observation.outcome is ExperimentOutcome.HYPOTHESIS
    assert result.observation.selected_option_id == "hold-shadow"

    membership = _experiment(
        required_data_ids=("family_id",),
        decision_rule=_rule(
            rule_class=DecisionRuleClass.CLOSED_MEMBERSHIP,
            observation_binding="family_id",
            hypothesis_operand=("IMPORT_PURITY_REPAIR",),
            counterfactual_operand="",
        ),
    )
    membership_decision = _plan(membership)
    observed = ShadowExperimentRunner().run(
        membership_decision,
        membership,
        observed_facts={"family_id": "OTHER_FAMILY"},
    )
    assert observed.outcome is ExperimentOutcome.COUNTERFACTUAL
    assert not _walk_has_float(observed.to_facts())


def test_forged_authorization_flag_on_a_decision_is_rejected() -> None:
    decision = _plan()
    payload = decision.to_dict()
    payload["can_authorize"] = True
    with pytest.raises(ExperimentError, match="cannot authorize"):
        ExperimentDecision.from_dict(payload)

    payload = decision.to_dict()
    payload["callback"] = "not-allowed"
    with pytest.raises(ProcedureContractError, match="unsupported fields"):
        ExperimentDecision.from_dict(payload)

    with pytest.raises(ExperimentObservationError, match="cannot authorize"):
        ExperimentObservationRecord(
            bindings=_bindings(),
            experiment_id="shadow-proof-status",
            question_id="world.unavailable.proof_status",
            outcome=ExperimentOutcome.HYPOTHESIS,
            selected_option_id="hold-shadow",
            observed_facts={"proof_status_current": False},
            isolation=_fixture_isolation(),
            can_authorize=True,
        )

    with pytest.raises(ExperimentObservationError, match="verified or promoted"):
        ExperimentObservationRecord(
            bindings=_bindings(),
            experiment_id="shadow-proof-status",
            question_id="world.unavailable.proof_status",
            outcome=ExperimentOutcome.HYPOTHESIS,
            selected_option_id="hold-shadow",
            observed_facts={"proof_status_current": False},
            isolation=_fixture_isolation(),
            state=ArtifactState.PROMOTED,
        )


def test_generic_experiment_plan_envelope_is_unchanged() -> None:
    from ipfs_accelerate_py.agent_supervisor.procedure_compiler import contracts

    record = contracts.ExperimentPlan(
        bindings=_bindings(),
        state=ArtifactState.SHADOW,
        subject_cid="experiment-cid",
        facts={"maximum_executions": 2, "dry_run": True},
    )
    decoded = parse_procedure_artifact(record.to_dict())
    assert decoded == record
    assert isinstance(decoded, contracts.BoundedArtifact)
