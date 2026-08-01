"""Contract tests for create/steer, plan-delta/revision, and task v2 (PDR-020)."""

from __future__ import annotations

from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    CompletionAuthority,
    DeltaEffectClass,
    DirtyTreePolicy,
    FallbackPolicy,
    LifecycleState,
    MergeStrategyKind,
    PlanAuthorityRoots,
    PlanCompletionRule,
    PlanConflictContract,
    PlanCreateRequest,
    PlanDelta,
    PlanDeltaItem,
    PlanDeltaOperation,
    PlanLeaseContract,
    PlanMergeStrategy,
    PlanOrigin,
    PlanPopulationDigest,
    PlanProviderContract,
    PlanRequestBudget,
    PlanResourceContract,
    PlanRetryContract,
    PlanRevision,
    PlanRevisionContractError,
    PlanRevisionIdentityError,
    PlanRevisionLifecycleError,
    PlanRevisionPathError,
    PlanRevisionSecretError,
    PlanRevisionStaleRootError,
    PlanSteerRequest,
    PlanValidationNode,
    PlanWorktreeContract,
    PopulationKind,
    TaskSourceKind,
    assert_population_history_intact,
    closed_delta_operations,
    is_history_immutable,
    plan_revision_cid,
)
from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    PromptAcceptanceRecord,
    PromptGoalRecord,
    PromptGoalRecordV2,
    PromptOutputRecord,
    PromptParallelContract,
    PromptTaskRecord,
    PromptTaskRecordV2,
    PromptValidationRecord,
    RecordStatus,
    conservative_parallel_contract,
    prompt_workflow_cid,
    read_goal_record,
    read_task_record,
    upgrade_goal_record_v1_to_v2,
    upgrade_task_record_v1_to_v2,
)


def _cid(name: str) -> str:
    return plan_revision_cid({"fixture": name})


def _roots(**changes: object) -> PlanAuthorityRoots:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:test",
        "repository_root_cid": _cid("repo-root"),
        "dirty_worktree_root": _cid("dirty"),
        "task_source_id": "task-source:markdown:board",
        "task_source_revision": _cid("ts-rev-1"),
        "policy_root": _cid("policy"),
        "intent_ir_root": _cid("intent"),
        "legal_ir_root": _cid("legal"),
        "security_ir_root": _cid("security"),
        "program_root": _cid("program"),
        "capability_catalog_root": _cid("capability"),
        "provider_catalog_root": _cid("provider-catalog"),
        "usage_policy_root": _cid("usage"),
        "configuration_root": _cid("config"),
    }
    values.update(changes)
    return PlanAuthorityRoots(**values)


def _budget(**changes: int) -> PlanRequestBudget:
    values = {
        "max_goals": 16,
        "max_tasks": 64,
        "max_graph_depth": 8,
        "max_output_paths": 128,
        "max_ready_width": 1,
        "max_repair_rounds": 2,
        "max_scan_bytes": 8 * 1024 * 1024,
        "max_analysis_operations": 16,
        "max_evidence_items": 64,
        "max_logic_families": 4,
        "max_model_calls": 2,
        "max_latency_ms": 60_000,
        "max_provider_tokens": 8_192,
        "max_cost_micros": 0,
    }
    values.update(changes)
    return PlanRequestBudget(**values)


def _population(
    kind: PopulationKind, *members: str
) -> PlanPopulationDigest:
    return PlanPopulationDigest(kind=kind, member_cids=members)


def _create_request(**changes: object) -> PlanCreateRequest:
    values: dict[str, object] = {
        "prompt_source_cid": _cid("prompt"),
        "repository_id": "repository:sha256:test",
        "repository_root": "/workspace/repository",
        "scope_paths": ("ipfs_accelerate_py/agent_supervisor",),
        "dirty_tree_policy": DirtyTreePolicy.OBSERVE_AND_BIND,
        "task_source_kind": TaskSourceKind.BOTH,
        "board_namespace": "agent-supervisor-test",
        "alias_prefix": "PDR",
        "roots": _roots(),
        "budget": _budget(),
        "required_analysis_operations": ("symbol_impact", "graph_rag_retrieval"),
        "optional_analysis_operations": ("premise_selection",),
        "required_logic_families": (),
        "optional_logic_families": ("contradiction_search",),
        "fallback_policy": FallbackPolicy.FAIL_CLOSED,
        "caller": "principal:test",
        "idempotency_key": "create/attempt-1",
    }
    values.update(changes)
    return PlanCreateRequest(**values)


def _steer_request(**changes: object) -> PlanSteerRequest:
    claimed = _population(PopulationKind.CLAIMED, _cid("task-running"))
    accepted = _population(PopulationKind.ACCEPTED, _cid("task-done"))
    status = _population(
        PopulationKind.UNSTARTED, _cid("task-ready"), _cid("task-running")
    )
    values: dict[str, object] = {
        "directive_cid": _cid("directive"),
        "base_admitted_plan_root": _cid("admitted-plan"),
        "base_materialized_plan_root": _cid("materialized-plan"),
        "plan_revision": 2,
        "parent_revision": 1,
        "roots": _roots(),
        "event_cursor": _cid("cursor-1"),
        "status_population": status,
        "claimed_population": claimed,
        "accepted_population": accepted,
        "accepted_evidence_root": _cid("accepted-evidence"),
        "completion_revision": _cid("completion-1"),
        "allowed_delta_operations": (
            PlanDeltaOperation.ADD_TASK.value,
            PlanDeltaOperation.ATTACH_EVIDENCE.value,
            PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK.value,
        ),
        "budget": _budget(),
        "caller": "principal:test",
        "idempotency_key": "steer/attempt-1",
        "lease_id": "lease:1",
        "fencing_epoch": 3,
    }
    values.update(changes)
    return PlanSteerRequest(**values)


def _delta_item(**changes: object) -> PlanDeltaItem:
    values: dict[str, object] = {
        "item_key": "delta:add-task-1",
        "operation": PlanDeltaOperation.ADD_TASK,
        "target_cid": "",
        "expected_target_lifecycle": LifecycleState.UNSTARTED,
        "expected_target_spec_revision": "",
        "before_digest": "",
        "after_record_cid": _cid("new-task"),
        "effect_class": DeltaEffectClass.MATERIALIZABLE_NOW,
        "rationale": "Add a successor validation task.",
        "provenance": {"source": "deterministic-fallback"},
        "affected_goal_cids": (_cid("goal-root"),),
        "affected_task_cids": (_cid("new-task"),),
        "affected_paths": ("ipfs_accelerate_py/agent_supervisor/planning/x.py",),
        "expected_effects": ("append-task",),
    }
    values.update(changes)
    return PlanDeltaItem(**values)


def _delta(**changes: object) -> PlanDelta:
    values: dict[str, object] = {
        "base_plan_root": _cid("materialized-plan"),
        "base_plan_revision": 1,
        "request_cid": _cid("steer-request"),
        "roots": _roots(),
        "items": (_delta_item(),),
        "expected_effects": ("append-task",),
        "claimed_population_digest": _cid("claimed-pop"),
        "accepted_population_digest": _cid("accepted-pop"),
        "scan_receipt_cid": _cid("scan"),
        "evidence_bundle_cid": _cid("evidence"),
        "admission_receipt_cid": _cid("admission"),
    }
    values.update(changes)
    return PlanDelta(**values)


def _revision(**changes: object) -> PlanRevision:
    values: dict[str, object] = {
        "plan_root_cid": _cid("plan-root-1"),
        "semantic_revision": 1,
        "parent_plan_root": "",
        "origin": PlanOrigin.CREATE,
        "roots": _roots(),
        "request_cid": _cid("create-request"),
        "delta_cid": "",
        "scan_receipt_cid": _cid("scan"),
        "query_plan_cid": _cid("query"),
        "evidence_bundle_cid": _cid("evidence"),
        "admission_receipt_cid": _cid("admission"),
        "execution_plan_cid": _cid("exec-plan"),
        "goal_population": _population(PopulationKind.RETAINED, _cid("goal-1")),
        "task_population": _population(PopulationKind.RETAINED, _cid("task-1")),
        "added_population": _population(PopulationKind.ADDED, _cid("goal-1"), _cid("task-1")),
        "superseded_population": _population(PopulationKind.SUPERSEDED),
        "retained_population": _population(PopulationKind.RETAINED),
        "deferred_population": _population(PopulationKind.DEFERRED),
        "claimed_population": _population(PopulationKind.CLAIMED),
        "completed_population": _population(PopulationKind.COMPLETED),
        "blocked_population": _population(PopulationKind.BLOCKED),
        "resource_contract": PlanResourceContract(),
        "provider_contract": PlanProviderContract(),
        "lease_contract": PlanLeaseContract(),
        "retry_contract": PlanRetryContract(),
        "worktree_contract": PlanWorktreeContract(),
        "merge_strategy": PlanMergeStrategy(kind=MergeStrategyKind.SERIAL),
        "conflict_contract": PlanConflictContract(
            predicted_files=("ipfs_accelerate_py/agent_supervisor/planning/x.py",),
        ),
        "completion_rule": PlanCompletionRule(
            authority=CompletionAuthority.VALIDATION_GATE,
        ),
        "validation_dag": (
            PlanValidationNode(
                validation_key="validation:pytest",
                argv=("python", "-m", "pytest", "-q"),
            ),
        ),
        "event_cursor": _cid("cursor-0"),
    }
    values.update(changes)
    return PlanRevision(**values)


# ---------------------------------------------------------------------------
# Authority roots / create / steer
# ---------------------------------------------------------------------------


def test_authority_roots_bind_all_required_identities_and_round_trip() -> None:
    roots = _roots()
    assert roots.repository_id.startswith("repository:")
    assert roots.policy_root == _cid("policy")
    assert roots.capability_catalog_root == _cid("capability")
    assert PlanAuthorityRoots.from_dict(roots.to_record()) == roots
    assert roots.content_id.startswith("b")


def test_create_request_binds_roots_budget_and_is_body_free() -> None:
    request = _create_request()
    encoded = request.to_json()
    assert "prompt_text" not in encoded
    assert request.prompt_source_cid in encoded
    assert request.roots.repository_root_cid in encoded
    assert request.budget.max_ready_width == 1
    assert PlanCreateRequest.from_json(encoded).content_id == request.content_id

    changed = replace(request, caller="principal:other")
    assert changed.content_id != request.content_id


def test_create_request_rejects_repository_identity_mismatch() -> None:
    with pytest.raises(PlanRevisionContractError, match="repository_id"):
        _create_request(repository_id="repository:other")


def test_steer_request_binds_parent_revision_and_populations() -> None:
    request = _steer_request()
    assert request.plan_revision == 2
    assert request.parent_revision == 1
    assert request.claimed_population.kind is PopulationKind.CLAIMED
    assert request.accepted_population.kind is PopulationKind.ACCEPTED
    assert PlanSteerRequest.from_dict(request.to_record()) == request


def test_steer_request_fails_stale_roots_cursor_and_populations() -> None:
    request = _steer_request()
    with pytest.raises(PlanRevisionStaleRootError, match="stale"):
        request.require_fresh(
            roots=replace(_roots(), dirty_worktree_root=_cid("dirty-2")),
            plan_revision=2,
            event_cursor=request.event_cursor,
            claimed_digest=request.claimed_population.digest,
            accepted_evidence_root=request.accepted_evidence_root,
        )
    with pytest.raises(PlanRevisionStaleRootError, match="cursor"):
        request.require_fresh(
            roots=request.roots,
            plan_revision=2,
            event_cursor=_cid("cursor-stale"),
            claimed_digest=request.claimed_population.digest,
            accepted_evidence_root=request.accepted_evidence_root,
        )
    with pytest.raises(PlanRevisionStaleRootError, match="claimed"):
        request.require_fresh(
            roots=request.roots,
            plan_revision=2,
            event_cursor=request.event_cursor,
            claimed_digest=_cid("wrong-claimed"),
            accepted_evidence_root=request.accepted_evidence_root,
        )


def test_steer_rejects_parent_revision_not_less_than_plan() -> None:
    with pytest.raises(PlanRevisionContractError, match="parent_revision"):
        _steer_request(plan_revision=1, parent_revision=1)


# ---------------------------------------------------------------------------
# Closed delta language / lifecycle immutability
# ---------------------------------------------------------------------------


def test_closed_delta_operations_are_stable_and_complete() -> None:
    ops = closed_delta_operations()
    assert "add_goal" in ops
    assert "add_task" in ops
    assert "request_lifecycle_action" in ops
    assert "delete_task" not in ops
    assert "edit_completed_task" not in ops
    assert len(ops) == len(PlanDeltaOperation)


def test_delta_round_trip_and_unique_item_keys() -> None:
    delta = _delta()
    assert PlanDelta.from_dict(delta.to_record()) == delta
    assert delta.delta_cid.startswith("b")
    with pytest.raises(PlanRevisionContractError, match="unique"):
        _delta(items=(_delta_item(), _delta_item()))


def test_delta_rejects_mutation_of_completed_and_claimed_history() -> None:
    with pytest.raises(PlanRevisionLifecycleError, match="completed|history"):
        _delta_item(
            operation=PlanDeltaOperation.SUPERSEDE_UNSTARTED_TASK,
            target_cid=_cid("task-done"),
            expected_target_lifecycle=LifecycleState.COMPLETED,
            before_digest=_cid("before"),
            after_record_cid=_cid("after"),
        )
    with pytest.raises(PlanRevisionLifecycleError, match="claimed|history|running"):
        _delta_item(
            item_key="delta:mutate-claimed",
            operation=PlanDeltaOperation.AMEND_UNSTARTED_GOAL,
            target_cid=_cid("task-running"),
            expected_target_lifecycle=LifecycleState.CLAIMED,
            before_digest=_cid("before"),
            after_record_cid=_cid("after"),
        )


def test_delta_allows_safe_attach_on_completed_and_successor_on_claimed() -> None:
    attach = _delta_item(
        item_key="delta:attach",
        operation=PlanDeltaOperation.ATTACH_EVIDENCE,
        target_cid=_cid("task-done"),
        expected_target_lifecycle=LifecycleState.COMPLETED,
        after_record_cid=_cid("evidence-ref"),
        effect_class=DeltaEffectClass.EVIDENCE_ONLY,
    )
    successor = _delta_item(
        item_key="delta:successor",
        operation=PlanDeltaOperation.ADD_TASK,
        target_cid=_cid("task-running"),
        expected_target_lifecycle=LifecycleState.RUNNING,
        after_record_cid=_cid("successor-task"),
        effect_class=DeltaEffectClass.DEFERRED,
    )
    delta = _delta(items=(attach, successor))
    assert len(delta.items) == 2


def test_population_history_cannot_shrink_or_delete() -> None:
    completed = {_cid("t1"), _cid("t2")}
    accepted = {_cid("t1")}
    claimed = {_cid("t3")}
    assert_population_history_intact(
        prior_completed=completed,
        prior_accepted=accepted,
        prior_claimed=claimed,
        next_completed=completed | {_cid("t3")},
        next_accepted=accepted,
        next_claimed=set(),
    )
    with pytest.raises(PlanRevisionLifecycleError, match="deleted"):
        assert_population_history_intact(
            prior_completed=completed,
            prior_accepted=accepted,
            prior_claimed=claimed,
            next_completed=completed,
            next_accepted=accepted,
            next_claimed=claimed,
            deleted_cids=(_cid("t1"),),
        )
    with pytest.raises(PlanRevisionLifecycleError, match="accepted"):
        assert_population_history_intact(
            prior_completed=completed,
            prior_accepted=accepted,
            prior_claimed=claimed,
            next_completed=completed,
            next_accepted=set(),
            next_claimed=claimed,
        )


def test_history_immutable_predicate() -> None:
    assert is_history_immutable(LifecycleState.COMPLETED)
    assert is_history_immutable("claimed")
    assert is_history_immutable(LifecycleState.RUNNING)
    assert not is_history_immutable(LifecycleState.UNSTARTED)
    assert not is_history_immutable(LifecycleState.PROPOSED)


# ---------------------------------------------------------------------------
# Plan revision ancestry / contracts
# ---------------------------------------------------------------------------


def test_create_revision_requires_revision_one_without_parent() -> None:
    revision = _revision()
    assert revision.origin is PlanOrigin.CREATE
    assert revision.semantic_revision == 1
    assert revision.parent_plan_root == ""
    assert PlanRevision.from_dict(revision.to_record()) == revision

    with pytest.raises(PlanRevisionContractError, match="semantic_revision"):
        _revision(semantic_revision=2)
    with pytest.raises(PlanRevisionContractError, match="parent_plan_root"):
        _revision(parent_plan_root=_cid("parent"))


def test_steer_revision_requires_parent_and_delta() -> None:
    revision = _revision(
        plan_root_cid=_cid("plan-root-2"),
        semantic_revision=2,
        parent_plan_root=_cid("plan-root-1"),
        origin=PlanOrigin.STEER,
        delta_cid=_cid("delta-1"),
        request_cid=_cid("steer-request"),
        added_population=_population(PopulationKind.ADDED, _cid("task-new")),
        retained_population=_population(
            PopulationKind.RETAINED, _cid("goal-1"), _cid("task-1")
        ),
    )
    assert revision.origin is PlanOrigin.STEER
    assert revision.parent_plan_root != revision.plan_root_cid
    assert revision.delta_cid
    assert PlanRevision.from_json(revision.to_json()).revision_cid == revision.revision_cid

    with pytest.raises(PlanRevisionContractError, match="parent_plan_root"):
        _revision(
            semantic_revision=2,
            origin=PlanOrigin.STEER,
            parent_plan_root="",
            delta_cid=_cid("delta"),
        )
    with pytest.raises(PlanRevisionIdentityError, match="differ"):
        _revision(
            plan_root_cid=_cid("same"),
            semantic_revision=2,
            origin=PlanOrigin.STEER,
            parent_plan_root=_cid("same"),
            delta_cid=_cid("delta"),
        )


def test_revision_binds_resources_leases_retries_worktrees_merge_and_validation_dag() -> None:
    revision = _revision(
        resource_contract=PlanResourceContract(
            resource_class="cpu-medium",
            cpu_slots=2,
            memory_bytes=512 * 1024 * 1024,
        ),
        lease_contract=PlanLeaseContract(
            lease_scope="lane",
            lease_duration_ms=300_000,
            fencing_epoch=7,
        ),
        retry_contract=PlanRetryContract(max_retries=2, backoff_ms=1_000),
        worktree_contract=PlanWorktreeContract(
            policy="isolated",
            isolation_required=True,
            expected_base_revision=_cid("base"),
        ),
        merge_strategy=PlanMergeStrategy(
            kind=MergeStrategyKind.MERGE_TRAIN,
            merge_group="train-a",
            post_merge_validation_cids=(_cid("post-merge"),),
        ),
        validation_dag=(
            PlanValidationNode(
                validation_key="validation:unit",
                argv=("python", "-m", "pytest", "test/unit", "-q"),
            ),
            PlanValidationNode(
                validation_key="validation:integration",
                dependency_keys=("validation:unit",),
                argv=("python", "-m", "pytest", "test/api", "-q"),
            ),
        ),
    )
    assert revision.resource_contract.cpu_slots == 2
    assert revision.lease_contract.fencing_epoch == 7
    assert revision.merge_strategy.kind is MergeStrategyKind.MERGE_TRAIN
    assert len(revision.validation_dag) == 2


def test_validation_dag_rejects_cycles_and_unknown_deps() -> None:
    with pytest.raises(PlanRevisionContractError, match="cycle"):
        _revision(
            validation_dag=(
                PlanValidationNode(
                    validation_key="a",
                    dependency_keys=("b",),
                    argv=("true",),
                ),
                PlanValidationNode(
                    validation_key="b",
                    dependency_keys=("a",),
                    argv=("true",),
                ),
            )
        )
    with pytest.raises(PlanRevisionContractError, match="unknown dependency"):
        _revision(
            validation_dag=(
                PlanValidationNode(
                    validation_key="a",
                    dependency_keys=("missing",),
                    argv=("true",),
                ),
            )
        )


# ---------------------------------------------------------------------------
# Fail-closed: unknown fields, secrets, floats, paths, CID, identity
# ---------------------------------------------------------------------------


def test_unknown_fields_rejected() -> None:
    payload = _roots().to_record()
    payload["extra_field"] = "nope"
    with pytest.raises(PlanRevisionContractError, match="unsupported fields"):
        PlanAuthorityRoots.from_dict(payload)


def test_secrets_and_source_bodies_rejected() -> None:
    with pytest.raises(PlanRevisionSecretError):
        _create_request(redacted_source_metadata={"api_key": "secret"})
    with pytest.raises(PlanRevisionSecretError):
        _delta_item(provenance={"source_body": "def evil(): pass"})
    with pytest.raises(PlanRevisionSecretError):
        PlanDeltaItem(
            item_key="x",
            operation=PlanDeltaOperation.ADD_TASK,
            target_cid="",
            expected_target_lifecycle=LifecycleState.UNSTARTED,
            expected_target_spec_revision="",
            before_digest="",
            after_record_cid=_cid("t"),
            effect_class=DeltaEffectClass.MATERIALIZABLE_NOW,
            rationale="sk-" + "x" * 24,
        )


def test_floats_rejected_in_mappings() -> None:
    with pytest.raises(PlanRevisionContractError, match="float"):
        _create_request(redacted_source_metadata={"score": 1.5})


def test_path_errors_fail_closed() -> None:
    with pytest.raises(PlanRevisionPathError):
        _create_request(repository_root="relative/path")
    with pytest.raises(PlanRevisionPathError):
        _create_request(scope_paths=("../escape",))
    with pytest.raises(PlanRevisionPathError):
        PlanConflictContract(predicted_files=("/absolute.py",))


def test_cid_and_identity_tampering_fail() -> None:
    with pytest.raises(PlanRevisionIdentityError):
        _create_request(prompt_source_cid="not-a-cid")
    roots = _roots()
    payload = roots.to_record()
    payload["content_id"] = "baguqeeraaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    with pytest.raises(PlanRevisionIdentityError):
        PlanAuthorityRoots.from_dict(payload)
    pop = _population(PopulationKind.CLAIMED, _cid("t1"))
    with pytest.raises(PlanRevisionIdentityError, match="digest"):
        PlanPopulationDigest(
            kind=PopulationKind.CLAIMED,
            member_cids=(_cid("t1"),),
            digest=_cid("forged-digest"),
        )


# ---------------------------------------------------------------------------
# Prompt goal/task v2 + conservative v1 adapter
# ---------------------------------------------------------------------------


def _acceptance(evidence_cid: str) -> PromptAcceptanceRecord:
    return PromptAcceptanceRecord(
        criterion_key="criterion:tests",
        criterion="Focused contract tests pass.",
        evidence_cids=(evidence_cid,),
        validation_keys=("validation:pytest",),
    )


def _validation() -> PromptValidationRecord:
    return PromptValidationRecord(
        validation_key="validation:pytest",
        argv=("python", "-m", "pytest", "-q"),
        cwd=".",
        policy_cid=prompt_workflow_cid({"validation": "policy"}),
    )


def _v1_goal(**changes: object) -> PromptGoalRecord:
    evidence = prompt_workflow_cid({"evidence": "scan"})
    values: dict[str, object] = {
        "goal_key": "goal:root",
        "parent_goal_cid": "",
        "dependency_goal_cids": (),
        "title": "Root goal",
        "objective": "Define contracts.",
        "rationale": "Required by PDR-020.",
        "scope_paths": ("ipfs_accelerate_py/agent_supervisor",),
        "acceptance": (_acceptance(evidence),),
        "evidence_cids": (evidence,),
    }
    values.update(changes)
    return PromptGoalRecord(**values)


def _v1_task(goal_cid: str, **changes: object) -> PromptTaskRecord:
    evidence = prompt_workflow_cid({"evidence": "scan"})
    values: dict[str, object] = {
        "task_key": "task:contracts",
        "goal_cid": goal_cid,
        "dependency_task_cids": (),
        "objective": "Implement plan revision contracts.",
        "rationale": "Leaf goal needs a producer.",
        "scope_paths": (
            "ipfs_accelerate_py/agent_supervisor/planning/plan_revision_contracts.py",
        ),
        "outputs": (
            PromptOutputRecord(
                path=(
                    "ipfs_accelerate_py/agent_supervisor/planning/"
                    "plan_revision_contracts.py"
                ),
                effect="create",
                media_type="text/x-python",
            ),
        ),
        "validations": (_validation(),),
        "acceptance": (_acceptance(evidence),),
        "evidence_cids": (evidence,),
        "policy_roots": (
            prompt_workflow_cid({"policy": 1}),
            prompt_workflow_cid({"policy": 2}),
        ),
        "predicted_files": (
            "ipfs_accelerate_py/agent_supervisor/planning/plan_revision_contracts.py",
        ),
        "parallel_lane": "pdr-planner",
        "resource_class": "cpu-small",
    }
    values.update(changes)
    return PromptTaskRecord(**values)


def test_v1_goal_and_task_identity_preserved() -> None:
    goal = _v1_goal()
    task = _v1_task(goal.goal_cid)
    # Status/timestamps do not affect identity (existing v1 contract).
    assert (
        replace(goal, status=RecordStatus.COMPLETED, created_at_ms=9).goal_cid
        == goal.goal_cid
    )
    assert (
        replace(task, status=RecordStatus.RUNNING, updated_at_ms=9).task_cid
        == task.task_cid
    )


def test_v1_reads_upgrade_with_conservative_non_parallel_defaults() -> None:
    goal = _v1_goal()
    task = _v1_task(goal.goal_cid)

    upgraded_goal = read_goal_record(goal.to_dict())
    assert isinstance(upgraded_goal, PromptGoalRecordV2)
    assert upgraded_goal.record_schema_version == 2
    assert upgraded_goal.completion_authority == "validation_gate"
    assert upgraded_goal.closed_producer_population
    assert upgraded_goal.goal_key == goal.goal_key

    upgraded_task = read_task_record(task.to_dict())
    assert isinstance(upgraded_task, PromptTaskRecordV2)
    assert upgraded_task.record_schema_version == 2
    assert upgraded_task.parallel_contract.parallel_ready is False
    assert upgraded_task.parallel_contract.max_ready_width == 1
    assert upgraded_task.parallel_contract.merge_strategy == "serial"
    # Lane string alone never implies parallel readiness.
    assert upgraded_task.parallel_lane == "pdr-planner"
    assert upgraded_task.validation_dag
    assert upgraded_task.validation_dag[0].validation_key == "validation:pytest"


def test_v2_goal_requires_producers_for_leaves() -> None:
    goal = _v1_goal()
    with pytest.raises(Exception, match="producing_task_cids|closed_producer"):
        PromptGoalRecordV2(
            goal_key=goal.goal_key,
            parent_goal_cid="",
            dependency_goal_cids=(),
            title=goal.title,
            objective=goal.objective,
            rationale=goal.rationale,
            scope_paths=goal.scope_paths,
            acceptance=goal.acceptance,
            child_goal_cids=(),
            producing_task_cids=(),
            closed_producer_population="",
        )


def test_v2_task_parallel_ready_requires_width() -> None:
    with pytest.raises(Exception, match="max_ready_width"):
        PromptParallelContract(parallel_ready=True, max_ready_width=1)

    contract = conservative_parallel_contract(resource_class="cpu-small")
    assert contract.parallel_ready is False
    assert contract.max_ready_width == 1


def test_v2_records_round_trip() -> None:
    goal_v1 = _v1_goal()
    task_v1 = _v1_task(goal_v1.goal_cid)
    goal = upgrade_goal_record_v1_to_v2(
        goal_v1,
        producing_task_cids=(task_v1.task_cid,),
        plan_root_cid=prompt_workflow_cid({"plan": 1}),
        plan_revision=1,
    )
    task = upgrade_task_record_v1_to_v2(
        task_v1,
        plan_root_cid=prompt_workflow_cid({"plan": 1}),
        plan_revision=1,
        board_namespace="agent-supervisor-test",
    )
    assert PromptGoalRecordV2.from_dict(goal.to_record()) == goal
    assert PromptTaskRecordV2.from_dict(task.to_record()) == task
    assert read_goal_record(goal.to_dict()).goal_cid == goal.goal_cid
    assert read_task_record(task.to_dict()).task_cid == task.task_cid


def test_v2_native_read_does_not_force_serial_when_parallel_ready() -> None:
    goal_v1 = _v1_goal()
    task_v1 = _v1_task(goal_v1.goal_cid)
    parallel = PromptParallelContract(
        parallel_ready=True,
        max_ready_width=4,
        merge_strategy="merge_train",
        resource_class="cpu-medium",
        cpu_slots=2,
    )
    task = PromptTaskRecordV2(
        task_key=task_v1.task_key,
        goal_cid=task_v1.goal_cid,
        dependency_task_cids=(),
        objective=task_v1.objective,
        rationale=task_v1.rationale,
        scope_paths=task_v1.scope_paths,
        outputs=task_v1.outputs,
        validations=task_v1.validations,
        acceptance=task_v1.acceptance,
        evidence_cids=task_v1.evidence_cids,
        policy_roots=task_v1.policy_roots,
        parallel_contract=parallel,
        predicted_files=task_v1.predicted_files,
        resource_class="cpu-medium",
    )
    assert task.parallel_contract.parallel_ready is True
    assert task.parallel_contract.max_ready_width == 4
    assert PromptTaskRecordV2.from_json(task.to_json()) == task
