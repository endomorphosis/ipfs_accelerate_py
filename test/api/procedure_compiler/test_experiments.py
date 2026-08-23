from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import pytest
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.contracts import (
    ArtifactBindings,
    EffectClass,
    ProcedureBoundsError,
    ProcedureContractError,
    ProcedureIdentityError,
    RiskClass,
    TaskFamily,
    TaskFamilyBoundary,
)
from ipfs_accelerate_py.agent_supervisor.procedure_compiler.experiments import (
    DecisionRuleKind,
    DisposableWorktreeGrant,
    ExperimentAction,
    ExperimentAuthorityError,
    ExperimentCost,
    ExperimentDecision,
    ExperimentDecisionRule,
    ExperimentEffect,
    ExperimentError,
    ExperimentExecutionBounds,
    ExperimentFixture,
    ExperimentIsolation,
    ExperimentIsolationError,
    ExperimentIsolationKind,
    ExperimentOutcomeStatus,
    ExperimentPlanner,
    ExperimentPrivacyClass,
    ExperimentPrivacyError,
    ExperimentReasonCode,
    InMemoryDisposableWorktreePort,
    InMemoryShadowObservationStore,
    ObservedSupport,
    PendingDecision,
    ShadowExperimentObservation,
    ShadowExperimentRunner,
    ShadowExperimentSpec,
    UncertaintySource,
    questions_from_family,
    questions_from_world,
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
        "task_dependency_ids": ("PCPC-011",),
        "task_dependency_state_id": "task-dependencies-1",
        "proof_status_id": "proof-status-1",
        "test_status_id": "test-status-1",
        "capability_state_id": "capability-state-1",
        "provider_capacity_id": "provider-capacity-1",
        "worktree_ids": ("worktree-1",),
        "lease_ids": ("lease-1",),
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
    }
    values.update(changes)
    return RepositoryWorldState(**values)


def _boundary(**changes: object) -> TaskFamilyBoundary:
    values: dict[str, object] = {
        "positive_member_cids": ("positive-a",),
        "negative_example_cids": ("negative-a",),
        "boundary_example_cids": ("boundary-a",),
        "unknown_case_cids": ("unknown-a",),
        "risk_ceiling": RiskClass.REVERSIBLE_LOCAL,
        "permitted_repositories": ("repo",),
        "permitted_languages": ("python",),
        "permitted_frameworks": ("pytest",),
        "permitted_effect_classes": (EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
    }
    values.update(changes)
    return TaskFamilyBoundary(**values)


def _family(**changes: object) -> TaskFamily:
    values: dict[str, object] = {
        "bindings": _bindings(),
        "name": "IMPORT_PURITY_REPAIR",
        "goal_semantics": ("restore-import-purity",),
        "precondition_shape": ("import-side-effect-observed",),
        "affected_artifact_classes": ("python-source",),
        "effect_classes": (EffectClass.REPOSITORY_WRITE, EffectClass.VALIDATION),
        "required_operation_contracts": ("approved-patch-template@1", "test-runner@1"),
        "validation_structure": ("focused-tests", "postcondition-check"),
        "failure_signatures": ("import-side-effect",),
        "postcondition_shape": ("import-is-pure",),
        "rollback_structure": ("restore-exact-tree",),
        "boundary": _boundary(),
    }
    values.update(changes)
    return TaskFamily(**values)


def _bounds(**changes: object) -> ExperimentExecutionBounds:
    values: dict[str, object] = {
        "wall_time_ms": 1_000,
        "cpu_time_ms": 1_000,
        "memory_bytes": 1_048_576,
        "disk_bytes": 1_048_576,
        "token_limit": 100,
        "step_limit": 4,
        "observation_limit": 8,
        "path_limit": 8,
    }
    values.update(changes)
    return ExperimentExecutionBounds(**values)


def _rule(**changes: object) -> ExperimentDecisionRule:
    values: dict[str, object] = {
        "kind": DecisionRuleKind.CHANGES_PENDING_DECISION,
        "pending_decision_id": "promote-or-hold",
        "hypothesis_action": "evaluate-further",
        "counterfactual_action": "hold",
        "expected_decision_change_bps": 5_000,
    }
    values.update(changes)
    return ExperimentDecisionRule(**values)


def _pending(**changes: object) -> PendingDecision:
    values: dict[str, object] = {
        "decision_id": "promote-or-hold",
        "alternatives": ("hold", "evaluate-further"),
        "default_action": "hold",
        "open": True,
    }
    values.update(changes)
    return PendingDecision(**values)


def _isolation(**changes: object) -> ExperimentIsolation:
    values: dict[str, object] = {
        "kind": ExperimentIsolationKind.FIXTURE,
        "authorized": True,
        "disposable": True,
        "fixture_id": "unknown-case-fixture",
        "scope_paths": ("fixtures/unknown-case.json",),
    }
    values.update(changes)
    return ExperimentIsolation(**values)


def _spec(family: TaskFamily, **changes: object) -> ShadowExperimentSpec:
    values: dict[str, object] = {
        "bindings": _bindings(),
        "question_id": "family.unknown.unknown-a",
        "question": "is unknown-a inside the family?",
        "hypothesis": "unknown-a is a negative example",
        "counterfactual": "unknown-a is a positive member",
        "required_data_ids": ("unknown-case-trace",),
        "risk_class": RiskClass.OBSERVATION_ONLY,
        "privacy_class": ExperimentPrivacyClass.PUBLIC_FIXTURE,
        "cost": ExperimentCost(),
        "decision_rule": _rule(),
        "bounds": _bounds(),
        "isolation_kind": ExperimentIsolationKind.FIXTURE,
        "scope_paths": ("fixtures/unknown-case.json",),
        "family_cid": family.content_id,
        "fixture_id": "unknown-case-fixture",
        "hypothesis_expected": "negative",
        "counterfactual_expected": "positive",
    }
    values.update(changes)
    return ShadowExperimentSpec(**values)


def _plan(
    spec: ShadowExperimentSpec,
    family: TaskFamily,
    *,
    world: RepositoryWorldState | None = None,
    isolation: ExperimentIsolation | None = None,
    pending: PendingDecision | None = None,
) -> ExperimentDecision:
    return ExperimentPlanner().plan(
        spec,
        isolation=isolation or _isolation(),
        world=world,
        family=family,
        pending_decisions=(_pending() if pending is None else pending,),
    )


def test_world_and_family_boundaries_expose_explicit_questions() -> None:
    world = _world(
        unavailable_dimensions=("proof_status", "test_status"),
        projection_status=WorldProjectionStatus.INCOMPLETE,
    )
    family = _family()

    world_questions = questions_from_world(world)
    family_questions = questions_from_family(family)
    planner_questions = ExperimentPlanner().exposed_questions(world=world, family=family)

    assert {item.question_id for item in world_questions} == {
        "world.unavailable.proof_status",
        "world.unavailable.test_status",
        "world.projection.incomplete",
    }
    assert any(item.source is UncertaintySource.FAMILY_UNKNOWN_CASE for item in family_questions)
    assert "family.unknown.unknown-a" in {item.question_id for item in family_questions}
    assert "family.boundary.boundary-a" in {item.question_id for item in family_questions}
    assert planner_questions == world_questions + family_questions


def test_spec_requires_question_hypothesis_counterfactual_data_risk_privacy_cost_rule_and_bounds() -> None:
    family = _family()
    with pytest.raises(ProcedureContractError, match="question is required"):
        _spec(family, question="")
    with pytest.raises(ProcedureContractError, match="hypothesis is required"):
        _spec(family, hypothesis="")
    with pytest.raises(ProcedureContractError, match="counterfactual is required"):
        _spec(family, counterfactual="")
    with pytest.raises(ProcedureContractError, match="required_data_ids must not be empty"):
        _spec(family, required_data_ids=())
    with pytest.raises(ProcedureContractError, match="risk_class must be one of"):
        _spec(family, risk_class="unbounded")
    with pytest.raises(ProcedureContractError, match="privacy_class must be one of"):
        _spec(family, privacy_class="secret")
    with pytest.raises(ProcedureContractError, match="cost must be ExperimentCost"):
        _spec(family, cost=None)
    with pytest.raises(ProcedureContractError, match="decision_rule must be ExperimentDecisionRule"):
        _spec(family, decision_rule=None)
    with pytest.raises(ProcedureContractError, match="bounds must be ExperimentExecutionBounds"):
        _spec(family, bounds=None)
    with pytest.raises(ProcedureContractError, match="wall_time_ms must be positive"):
        _bounds(wall_time_ms=0)


def test_privacy_rejects_secret_or_executable_data_identifiers() -> None:
    family = _family()
    with pytest.raises(ExperimentPrivacyError, match="forbidden secret"):
        _spec(family, required_data_ids=("api_key",))
    with pytest.raises(ExperimentPrivacyError, match="forbidden secret"):
        ExperimentFixture(fixture_id="leaky", values={"private_key": "abcd"})


def test_execution_bounds_forbid_network_subprocesses_and_unbounded_limits() -> None:
    with pytest.raises(ProcedureBoundsError, match="network_request_limit"):
        _bounds(network_request_limit=1)
    with pytest.raises(ProcedureBoundsError, match="subprocess_limit"):
        _bounds(subprocess_limit=1)
    with pytest.raises(ProcedureContractError, match="step_limit must be positive"):
        _bounds(step_limit=0)
    with pytest.raises(ProcedureBoundsError, match="token_limit"):
        _bounds(token_limit=10**9)


def test_decision_relevant_experiment_runs_only_on_authorized_fixture() -> None:
    family = _family()
    spec = _spec(family)
    decision = _plan(spec, family)

    assert decision.action is ExperimentAction.RUN
    assert decision.decision_relevant is True
    assert decision.reason_code is ExperimentReasonCode.DECISION_RELEVANT
    assert decision.matched_question_id == "family.unknown.unknown-a"
    assert decision.pending_decision_id == "promote-or-hold"
    assert decision.expected_decision_change_bps == 5_000
    assert decision.observation_only is True
    assert decision.can_grant_authority is False
    assert decision.can_promote is False
    assert decision.can_establish_proof is False

    runner = ShadowExperimentRunner()
    result = runner.run(
        spec,
        decision,
        isolation=_isolation(),
        fixture=ExperimentFixture(
            fixture_id="unknown-case-fixture",
            values={"unknown-case-trace": "negative"},
        ),
    )

    assert result.status is ExperimentOutcomeStatus.OBSERVED
    assert result.observation is not None
    assert result.observation.support is ObservedSupport.HYPOTHESIS
    assert result.observation.observation_only is True
    assert result.can_grant_authority is False
    assert result.is_authoritative is False
    assert runner.store.get(spec.content_id) == result.observation
    with pytest.raises(ExperimentAuthorityError, match="cannot authorize"):
        result.authorize()
    with pytest.raises(ExperimentAuthorityError, match="cannot authorize"):
        result.observation.authorize()


def test_experiment_is_skipped_when_it_cannot_change_a_pending_decision() -> None:
    family = _family()
    same_action = _spec(
        family,
        decision_rule=_rule(hypothesis_action="hold", counterfactual_action="hold"),
    )
    same_text = _spec(family, hypothesis="same", counterfactual="same")
    closed = _spec(family)
    already = _pending(open=False, admitted_evidence_ids=("admission-1",))
    no_pending = _spec(family, decision_rule=_rule(pending_decision_id="other-decision"))
    low_value = _spec(family, decision_rule=_rule(expected_decision_change_bps=0))

    planner = ExperimentPlanner()
    isolation = _isolation()
    pending = _pending()

    assert (
        planner.plan(
            same_action, isolation=isolation, family=family, pending_decisions=(pending,)
        ).reason_code
        is ExperimentReasonCode.HYPOTHESIS_EQUALS_COUNTERFACTUAL
    )
    assert (
        planner.plan(
            same_text, isolation=isolation, family=family, pending_decisions=(pending,)
        ).action
        is ExperimentAction.SKIP
    )
    assert (
        planner.plan(
            closed, isolation=isolation, family=family, pending_decisions=(already,)
        ).reason_code
        is ExperimentReasonCode.ALREADY_DECIDED
    )
    assert (
        planner.plan(
            no_pending, isolation=isolation, family=family, pending_decisions=(pending,)
        ).reason_code
        is ExperimentReasonCode.NO_PENDING_DECISION
    )
    assert (
        planner.plan(
            low_value, isolation=isolation, family=family, pending_decisions=(pending,)
        ).reason_code
        is ExperimentReasonCode.CANNOT_CHANGE_DECISION
    )


def test_question_not_exposed_by_world_or_family_is_skipped() -> None:
    family = _family()
    world = _world(unavailable_dimensions=())
    spec = _spec(family, question_id="family.unknown.missing-case")
    decision = ExperimentPlanner().plan(
        spec,
        isolation=_isolation(),
        world=world,
        family=family,
        pending_decisions=(_pending(),),
    )
    assert decision.action is ExperimentAction.SKIP
    assert decision.reason_code is ExperimentReasonCode.QUESTION_NOT_EXPOSED
    assert decision.decision_relevant is False


def test_no_explicit_uncertainty_questions_are_skipped() -> None:
    family = _family()
    spec = _spec(family)
    world = _world(unavailable_dimensions=())
    decision = ExperimentPlanner().plan(
        spec,
        isolation=_isolation(),
        world=world,
        pending_decisions=(_pending(),),
    )
    assert decision.reason_code is ExperimentReasonCode.NO_EXPLICIT_QUESTION
    assert decision.action is ExperimentAction.SKIP


def test_world_uncertainty_question_can_authorize_only_an_observation_run() -> None:
    family = _family()
    world = _world()
    spec = _spec(
        family,
        question_id="world.unavailable.proof_status",
        question="is proof_status available?",
        world_state_id=world.content_id,
    )
    decision = _plan(spec, family, world=world)
    assert decision.action is ExperimentAction.RUN
    assert decision.matched_question_id == "world.unavailable.proof_status"
    assert decision.can_grant_authority is False


def test_cost_above_fixed_bounds_is_refused() -> None:
    family = _family()
    spec = _spec(family, cost=ExperimentCost(tokens=500), bounds=_bounds(token_limit=10))
    decision = _plan(spec, family)
    assert decision.action is ExperimentAction.REFUSE
    assert decision.reason_code is ExperimentReasonCode.BOUND_EXCEEDED


def test_authority_or_repository_write_risk_is_refused() -> None:
    family = _family()
    high_risk = _spec(family, risk_class=RiskClass.AUTHORITY_OR_SECURITY)
    write_risk = _spec(family, risk_class=RiskClass.REPOSITORY_WRITE)
    tight_family = _family(boundary=_boundary(risk_ceiling=RiskClass.OBSERVATION_ONLY))
    above_family = _spec(
        tight_family,
        family_cid=tight_family.content_id,
        risk_class=RiskClass.REVERSIBLE_LOCAL,
    )

    assert _plan(high_risk, family).reason_code is ExperimentReasonCode.RISK_CEILING
    assert _plan(write_risk, family).reason_code is ExperimentReasonCode.RISK_CEILING
    assert (
        ExperimentPlanner()
        .plan(
            above_family,
            isolation=_isolation(),
            family=tight_family,
            pending_decisions=(_pending(),),
        )
        .reason_code
        is ExperimentReasonCode.RISK_CEILING
    )


def test_production_and_policy_targets_are_refused_and_cannot_run() -> None:
    family = _family()
    spec = _spec(family)
    production = _plan(spec, family, isolation=_isolation(production=True))
    policy = _plan(spec, family, isolation=_isolation(policy_mutable=True))
    policy_path = _plan(
        _spec(family, scope_paths=("config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json",)),
        family,
        isolation=_isolation(
            scope_paths=("config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json",)
        ),
    )
    architecture = _plan(
        _spec(
            family,
            scope_paths=("docs/architecture/agent_supervisor_procedure_compiler.todo.md",),
        ),
        family,
        isolation=_isolation(
            scope_paths=("docs/architecture/agent_supervisor_procedure_compiler.todo.md",)
        ),
    )

    assert production.reason_code is ExperimentReasonCode.PRODUCTION_MUTATION
    assert policy.reason_code is ExperimentReasonCode.POLICY_MUTATION
    assert policy_path.reason_code is ExperimentReasonCode.POLICY_MUTATION
    assert architecture.reason_code is ExperimentReasonCode.POLICY_MUTATION

    runner = ShadowExperimentRunner()
    run_decision = _plan(spec, family)
    with pytest.raises(ExperimentIsolationError, match="production"):
        runner.run(
            spec,
            run_decision,
            isolation=_isolation(),
            fixture=ExperimentFixture(
                fixture_id="unknown-case-fixture",
                values={"unknown-case-trace": "negative"},
                production=True,
            ),
        )
    with pytest.raises(ExperimentIsolationError, match="refused"):
        runner.run(spec, production, isolation=_isolation(production=True))


def test_unauthorized_or_non_disposable_worktree_is_refused() -> None:
    family = _family()
    spec = _spec(
        family,
        isolation_kind=ExperimentIsolationKind.DISPOSABLE_WORKTREE,
        fixture_id="",
        scope_paths=("src/module.py",),
        cost=ExperimentCost(worktree_count=1),
    )
    unauthorized = ExperimentPlanner().plan(
        spec,
        isolation=ExperimentIsolation(
            kind=ExperimentIsolationKind.DISPOSABLE_WORKTREE,
            authorized=False,
            disposable=True,
            scope_paths=("src/module.py",),
        ),
        family=family,
        pending_decisions=(_pending(),),
    )
    sticky = ExperimentPlanner().plan(
        spec,
        isolation=ExperimentIsolation(
            kind=ExperimentIsolationKind.DISPOSABLE_WORKTREE,
            authorized=True,
            disposable=False,
            scope_paths=("src/module.py",),
        ),
        family=family,
        pending_decisions=(_pending(),),
    )
    assert unauthorized.reason_code is ExperimentReasonCode.UNAUTHORIZED_WORKTREE
    assert sticky.reason_code is ExperimentReasonCode.NON_DISPOSABLE_WORKTREE


def test_disposable_worktree_runner_uses_existing_authority_and_never_touches_production(
    tmp_path,
) -> None:
    family = _family()
    spec = _spec(
        family,
        isolation_kind=ExperimentIsolationKind.DISPOSABLE_WORKTREE,
        fixture_id="",
        scope_paths=("src/module.py",),
        required_data_ids=("src/module.py",),
        hypothesis_expected="observed-negative",
        counterfactual_expected="observed-positive",
        cost=ExperimentCost(worktree_count=1),
    )
    isolation = ExperimentIsolation(
        kind=ExperimentIsolationKind.DISPOSABLE_WORKTREE,
        authorized=True,
        disposable=True,
        scope_paths=("src/module.py",),
    )
    decision = ExperimentPlanner().plan(
        spec,
        isolation=isolation,
        family=family,
        pending_decisions=(_pending(),),
    )
    assert decision.action is ExperimentAction.RUN

    production = tmp_path / "production"
    policy = production / "config"
    policy.mkdir(parents=True)
    policy_file = policy / "authority_policy.json"
    policy_file.write_text("live-policy", encoding="utf-8")

    port = InMemoryDisposableWorktreePort(tmp_path / "worktrees")
    runner = ShadowExperimentRunner(worktree_port=port)
    result = runner.run(
        spec,
        decision,
        isolation=isolation,
        fixture=ExperimentFixture(
            fixture_id="worktree-data",
            values={"src/module.py": "observed-negative"},
        ),
    )

    assert result.status is ExperimentOutcomeStatus.OBSERVED
    assert result.observation is not None
    assert result.observation.support is ObservedSupport.HYPOTHESIS
    assert result.observation.worktree_id.startswith("wt-")
    assert result.observation.isolation_kind is ExperimentIsolationKind.DISPOSABLE_WORKTREE
    assert result.can_grant_authority is False
    assert any(
        effect.effect_class is EffectClass.WORKTREE_CREATE
        for effect in result.observation.effects
    )
    assert policy_file.read_text(encoding="utf-8") == "live-policy"
    assert list(production.rglob("observations")) == []
    assert port.acquired and port.released
    written = list((tmp_path / "worktrees").rglob("*.json"))
    assert written
    assert all(tmp_path / "worktrees" in path.parents for path in written)


def test_fixture_root_persists_observation_inside_isolation_only(tmp_path) -> None:
    family = _family()
    spec = _spec(family)
    decision = _plan(spec, family)
    fixture_root = tmp_path / "fixture"
    fixture_root.mkdir()
    runner = ShadowExperimentRunner()
    result = runner.run(
        spec,
        decision,
        isolation=_isolation(),
        fixture=ExperimentFixture(
            fixture_id="unknown-case-fixture",
            values={"unknown-case-trace": "positive"},
            root_path=str(fixture_root),
        ),
    )
    assert result.observation is not None
    assert result.observation.support is ObservedSupport.COUNTERFACTUAL
    stored = fixture_root / "observations" / (spec.content_id + ".json")
    assert stored.is_file()
    replayed = ShadowExperimentObservation.from_dict(result.observation.to_dict())
    assert replayed == result.observation
    assert replayed.can_grant_authority is False


def test_skipped_experiment_does_not_run_or_persist() -> None:
    family = _family()
    spec = _spec(family, decision_rule=_rule(expected_decision_change_bps=0))
    decision = _plan(spec, family)
    assert decision.action is ExperimentAction.SKIP
    store = InMemoryShadowObservationStore()
    runner = ShadowExperimentRunner(store=store)
    result = runner.run(
        spec,
        decision,
        isolation=_isolation(),
        fixture=ExperimentFixture(
            fixture_id="unknown-case-fixture",
            values={"unknown-case-trace": "negative"},
        ),
    )
    assert result.status is ExperimentOutcomeStatus.SKIPPED
    assert result.observation is None
    assert store.records() == ()


def test_canonical_round_trip_and_unknown_field_rejection() -> None:
    family = _family()
    spec = _spec(family)
    decision = _plan(spec, family)
    observation = ShadowExperimentObservation(
        bindings=_bindings(),
        experiment_id=spec.content_id,
        decision_id=decision.content_id,
        question_id=spec.question_id,
        producer_contract="shadow-experiment-observation@1",
        observed_values={"unknown-case-trace": "negative"},
        support=ObservedSupport.HYPOTHESIS,
        isolation_kind=ExperimentIsolationKind.FIXTURE,
        isolation_root="fixtures/unknown-case-fixture",
        fixture_id="unknown-case-fixture",
        effects=(
            ExperimentEffect(
                effect_id="observe",
                effect_class=EffectClass.OBSERVE,
                targets=("fixtures/unknown-case.json",),
            ),
        ),
    )

    assert ShadowExperimentSpec.from_dict(spec.to_dict()) == spec
    assert ExperimentDecision.from_dict(decision.to_dict()) == decision
    assert ShadowExperimentObservation.from_dict(observation.to_dict()) == observation
    assert observation.observation_only is True
    with pytest.raises(FrozenInstanceError):
        spec.question = "mutated"  # type: ignore[misc]

    forged = {**spec.to_dict(), "unknown_normative_field": True}
    with pytest.raises(ProcedureContractError, match="unsupported fields"):
        ShadowExperimentSpec.from_dict(forged)
    identity = spec.to_dict()
    identity["content_id"] = "forged"
    with pytest.raises(ProcedureIdentityError):
        ShadowExperimentSpec.from_dict(identity)
    with pytest.raises(ExperimentAuthorityError, match="observation-only"):
        ExperimentDecision.from_dict({**decision.to_dict(), "observation_only": False})


def test_experiment_effects_cannot_write_production_or_claim_authority() -> None:
    with pytest.raises(ExperimentAuthorityError, match="only observe or create"):
        ExperimentEffect(
            effect_id="write",
            effect_class=EffectClass.REPOSITORY_WRITE,
            targets=("src/a.py",),
        )
    with pytest.raises(ExperimentIsolationError, match="production or policy"):
        ExperimentEffect(
            effect_id="observe-policy",
            effect_class=EffectClass.OBSERVE,
            targets=("config/agent_supervisor_proof_carrying_procedure_compiler_scheduler.json",),
        )
    with pytest.raises(ExperimentIsolationError, match="production or policy"):
        DisposableWorktreeGrant(
            worktree_id="wt-1",
            reservation_id="res-1",
            receipt_cid="receipt-1",
            scope_paths=("docs/architecture/procedure_compiler_inventory/baseline.json",),
        )


def test_binding_mismatch_and_family_cid_mismatch_are_refused() -> None:
    family = _family()
    other = replace(_bindings(), tree_id="other-tree")
    mismatched_world = _world(bindings=other)
    spec = _spec(family, world_state_id="not-the-world")
    world = _world()
    assert (
        ExperimentPlanner()
        .plan(
            spec,
            isolation=_isolation(),
            world=world,
            family=family,
            pending_decisions=(_pending(),),
        )
        .reason_code
        is ExperimentReasonCode.BINDING_MISMATCH
    )
    assert (
        ExperimentPlanner()
        .plan(
            _spec(family),
            isolation=_isolation(),
            world=mismatched_world,
            family=family,
            pending_decisions=(_pending(),),
        )
        .reason_code
        is ExperimentReasonCode.BINDING_MISMATCH
    )
    assert (
        ExperimentPlanner()
        .plan(
            _spec(family, family_cid="other-family"),
            isolation=_isolation(),
            family=family,
            pending_decisions=(_pending(),),
        )
        .reason_code
        is ExperimentReasonCode.FAMILY_BOUNDARY_MISMATCH
    )


def test_runner_requires_matching_decision_and_rejects_authority_claims() -> None:
    family = _family()
    spec = _spec(family)
    other = _spec(family, question="other question text")
    decision = _plan(spec, family)
    other_decision = _plan(other, family)
    runner = ShadowExperimentRunner()
    with pytest.raises(ExperimentError, match="does not bind"):
        runner.run(spec, other_decision, isolation=_isolation())
    with pytest.raises(ExperimentIsolationError, match="fixture"):
        runner.run(spec, decision, isolation=_isolation())
