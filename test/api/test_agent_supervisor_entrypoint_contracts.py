"""ASE-003 prompt-entrypoint contract and storage-boundary tests."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    DEFAULT_PARQUET_PARTITIONS,
    REQUIRED_TARGET_DECISION_FIELDS,
    ContinuationAction,
    ContractBoundsError,
    ContractIdentityError,
    CoordinationShardBinding,
    DecisionEffect,
    EntrypointContractError,
    ExpectedEffect,
    InvocationBudget,
    InvocationMode,
    InvocationStatus,
    LaunchPlan,
    OutputMode,
    ProviderFallbackReason,
    ProviderRouteProvenance,
    ProviderSelection,
    ReplicationBinding,
    ReplicationMode,
    ResolutionDisposition,
    ResolutionSource,
    ResolvedSupervisorProfile,
    ResourceBudget,
    RevalidationRule,
    RunHandle,
    RunHealth,
    RunState,
    SecretBearingRecordError,
    SupervisorInvocationRequest,
    SupervisorInvocationResult,
    TargetCandidate,
    TargetInferenceDecision,
    TargetResolutionReceipt,
    TaskSourceKind,
    UnknownContractFieldError,
    WorktreeStrategy,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_dag_json,
)

PROMPT = "Improve the validation cache without leaking this prompt."


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": label})


def _route(
    selected: ProviderSelection = ProviderSelection.GROK,
) -> ProviderRouteProvenance:
    fallback = selected is ProviderSelection.CODEX
    return ProviderRouteProvenance(
        preferred_provider="grok",
        fallback_provider="codex",
        selected_provider=selected,
        fallback_reason=(
            ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED
            if fallback
            else ProviderFallbackReason.NONE
        ),
        fallback_receipt_cid=_cid("fallback") if fallback else "",
        observed_capability_cid=_cid("capability"),
        usage_evidence_cid=_cid("usage"),
        budget_cid=_cid("provider-budget"),
        task_revision_cid=_cid("task-revision"),
        attempt_cid=_cid(f"{selected.value}-attempt"),
        worktree_cid=_cid("provider-worktree"),
        maximum_fallback_dispatches=1,
        independent_review_required=True,
    )


def _resources() -> ResourceBudget:
    return ResourceBudget(
        max_lanes=4,
        max_processes=8,
        max_validation_workers=4,
        cpu_millis=8_000,
        memory_bytes=8 * 1024**3,
        provider_request_limit=100,
        deadline_ms=3_600_000,
    )


def _coordination(root: Path) -> CoordinationShardBinding:
    return CoordinationShardBinding(
        backend="duckdb",
        database_path=str(root / "state" / "coordination.duckdb"),
        shard_id="repo-run-0",
        shard_count=2,
        shard_index=0,
        owner_principal_ref="did:key:local-owner",
        coordinator_cid=_cid("coordinator"),
        lease_namespace="repo-run",
        fencing_generation=7,
        writable=True,
    )


def _replication(root: Path) -> ReplicationBinding:
    return ReplicationBinding(
        mode=ReplicationMode.PARQUET_IPLD_IPFS,
        parquet_dataset_path=str(root / "state" / "epochs"),
        parquet_schema_cid=_cid("parquet-schema"),
        partition_keys=DEFAULT_PARQUET_PARTITIONS,
        ipld_manifest_schema_cid=_cid("ipld-manifest-schema"),
        ipld_codec="dag-json",
        cid_profile="cidv1-base32-sha2-256",
        links_must_be_verified=True,
        car_export=True,
        ipfs_publish=True,
        ipfs_backend_handle="ipfs-kit:development",
        pin=True,
        max_events_per_epoch=10_000,
    )


def _decision(
    field_name: str,
    *,
    value: str | None = None,
    disposition: ResolutionDisposition = ResolutionDisposition.UNIQUE,
) -> TargetInferenceDecision:
    selected_value = value or f"value:{field_name}"
    source = (
        ResolutionSource.AUTHENTICATED_TRANSPORT
        if field_name
        in {"policy", "principal", "authority_source", "effect_ceiling"}
        else ResolutionSource.DISCOVERY
    )
    evidence = _cid(f"evidence-{field_name}")
    if disposition is ResolutionDisposition.AMBIGUOUS:
        candidates = (
            TargetCandidate(
                field_name=field_name,
                value=selected_value + ":a",
                source=source,
                source_precedence=60,
                evidence_cid=evidence,
                confidence_ppm=500_000,
            ),
            TargetCandidate(
                field_name=field_name,
                value=selected_value + ":b",
                source=source,
                source_precedence=60,
                evidence_cid=_cid(f"evidence-{field_name}-b"),
                confidence_ppm=500_000,
            ),
        )
        return TargetInferenceDecision(
            field_name=field_name,
            disposition=disposition,
            selected_value="",
            selected_source=source,
            source_precedence=60,
            evidence_cid=evidence,
            candidates=candidates,
            reason_codes=("multiple_viable_candidates",),
            effect=DecisionEffect.CONFIGURATION,
            override_accepted=False,
            fresh_until_ms=0,
            revalidation_rule=RevalidationRule.IMMUTABLE,
        )
    candidate = TargetCandidate(
        field_name=field_name,
        value=selected_value,
        source=source,
        source_precedence=60,
        evidence_cid=evidence,
    )
    return TargetInferenceDecision(
        field_name=field_name,
        disposition=disposition,
        selected_value=selected_value,
        selected_source=source,
        source_precedence=60,
        evidence_cid=evidence,
        candidates=(candidate,),
        reason_codes=(),
        effect=(
            DecisionEffect.REQUIRES_AUTHORITY
            if field_name
            in {"policy", "principal", "authority_source", "effect_ceiling"}
            else DecisionEffect.CONFIGURATION
        ),
        override_accepted=False,
        fresh_until_ms=0,
        revalidation_rule=RevalidationRule.IMMUTABLE,
    )


def _decisions(
    *,
    values: dict[str, str] | None = None,
    ambiguous_fields: frozenset[str] = frozenset(),
) -> tuple[TargetInferenceDecision, ...]:
    values = values or {}
    return tuple(
        _decision(
            field_name,
            value=values.get(field_name),
            disposition=(
                ResolutionDisposition.AMBIGUOUS
                if field_name in ambiguous_fields
                else ResolutionDisposition.UNIQUE
            ),
        )
        for field_name in REQUIRED_TARGET_DECISION_FIELDS
    )


def _invocation() -> SupervisorInvocationRequest:
    return SupervisorInvocationRequest.from_prompt(
        PROMPT,
        prompt_ref="prompt-broker:fixture",
        mode=InvocationMode.WORKTREE,
        budget=InvocationBudget(
            max_prompt_bytes=16_384,
            max_actions=64,
            max_lanes=4,
            timeout_ms=3_600_000,
            max_result_bytes=1024 * 1024,
        ),
        repository_hint="/srv/repo",
        scope_hint="/srv/repo/ipfs_accelerate_py",
        profile_hint="local-worktree",
        output_mode_hint=OutputMode.BOTH.value,
        lane_ceiling_hint=4,
    )


def _receipt(
    root: Path,
    *,
    invocation: SupervisorInvocationRequest | None = None,
    ambiguous_field: str = "",
    route: ProviderRouteProvenance | None = None,
) -> TargetResolutionReceipt:
    invocation = invocation or _invocation()
    repository_root = str(root)
    repository_id = "repository:fixture"
    checkout_id = "checkout:fixture"
    scope_path = str(root / "ipfs_accelerate_py")
    head_tree_cid = _cid("head-tree")
    dirty_overlay_cid = _cid("dirty-overlay")
    submodule_population_cid = _cid("submodules")
    nested_repository_population_cid = _cid("nested-repositories")
    state_root = str(root / "state")
    run_namespace = "fixture-run"
    objective_cid = _cid("objective")
    plan_cid = _cid("plan")
    task_source_cid = _cid("task-source")
    policy_cid = _cid("policy")
    principal_ref = "did:key:local-owner"
    authority_source_ref = "authority:local-worktree-profile"
    effect_ceiling_cid = _cid("effect-ceiling")
    output_mode = OutputMode.BOTH
    provider_route = route or _route()
    resource_budget_cid = _resources().content_id
    lane_ceiling = 4
    merge_target = "main"
    worktree_strategy = WorktreeStrategy.ISOLATED
    validation_profile_cid = _cid("validation-profile")
    coordination = _coordination(root)
    replication = _replication(root)

    ambiguous_fields: set[str] = set()
    if ambiguous_field:
        ambiguous_fields.add(ambiguous_field)
        if ambiguous_field == "repository_root":
            ambiguous_fields.update(
                {
                    "repository_id",
                    "checkout_id",
                    "scope",
                    "tree_id",
                    "dirty_overlay",
                    "submodules",
                    "nested_repositories",
                }
            )
            repository_root = ""
            repository_id = ""
            checkout_id = ""
            scope_path = ""
            head_tree_cid = ""
            dirty_overlay_cid = ""
            submodule_population_cid = ""
            nested_repository_population_cid = ""
        coordination = replace(coordination, writable=False)
        replication = replace(
            replication,
            mode=ReplicationMode.PARQUET_IPLD,
            ipfs_publish=False,
            ipfs_backend_handle="",
            pin=False,
        )
        worktree_strategy = WorktreeStrategy.NONE

    projections = {
        "repository_root": repository_root or str(root),
        "state_root": state_root,
        "repository_id": repository_id or "repository:fixture",
        "checkout_id": checkout_id or "checkout:fixture",
        "scope": scope_path or str(root / "ipfs_accelerate_py"),
        "tree_id": head_tree_cid or _cid("head-tree"),
        "dirty_overlay": dirty_overlay_cid or _cid("dirty-overlay"),
        "submodules": submodule_population_cid or _cid("submodules"),
        "nested_repositories": (
            nested_repository_population_cid or _cid("nested-repositories")
        ),
        "run_namespace": run_namespace,
        "objective": objective_cid,
        "plan": plan_cid,
        "task_source": task_source_cid,
        "policy": policy_cid,
        "principal": principal_ref,
        "authority_source": authority_source_ref,
        "effect_ceiling": effect_ceiling_cid,
        "output": output_mode.value,
        "provider": provider_route.content_id,
        "resources": resource_budget_cid,
        "lane_ceiling": str(lane_ceiling),
        "merge_target": merge_target,
        "worktree_strategy": worktree_strategy.value,
        "validation": validation_profile_cid,
        "coordination": coordination.content_id,
        "replication": replication.content_id,
    }
    decisions = _decisions(
        values=projections,
        ambiguous_fields=frozenset(ambiguous_fields),
    )
    return TargetResolutionReceipt(
        invocation_cid=invocation.content_id,
        prompt_cid=invocation.prompt_cid,
        repository_root=repository_root,
        repository_id=repository_id,
        checkout_id=checkout_id,
        scope_path=scope_path,
        head_tree_cid=head_tree_cid,
        dirty_overlay_cid=dirty_overlay_cid,
        submodule_population_cid=submodule_population_cid,
        nested_repository_population_cid=nested_repository_population_cid,
        state_root=state_root,
        run_namespace=run_namespace,
        objective_cid=objective_cid,
        objective_revision_cid=_cid("objective-revision"),
        plan_cid=plan_cid,
        task_source_cid=task_source_cid,
        task_source_revision_cid=_cid("task-source-revision"),
        task_source_kind=TaskSourceKind.DUAL,
        policy_cid=policy_cid,
        principal_ref=principal_ref,
        authority_source_ref=authority_source_ref,
        effect_ceiling_cid=effect_ceiling_cid,
        output_mode=output_mode,
        markdown_path=str(root / "state" / "plan.todo.md"),
        duckdb_path=str(root / "state" / "tasks.duckdb"),
        provider_route=provider_route,
        capability_report_cid=_cid("capability-report"),
        resource_budget_cid=resource_budget_cid,
        lane_ceiling=lane_ceiling,
        merge_target=merge_target,
        worktree_strategy=worktree_strategy,
        validation_profile_cid=validation_profile_cid,
        coordination_shard=coordination,
        replication=replication,
        configuration_root_cid=_cid("configuration-root"),
        capability_catalog_cid=_cid("capability-catalog"),
        decisions=decisions,
        unresolved_fields=tuple(sorted(ambiguous_fields)),
        resolved_at_ms=1_000,
        fresh_until_ms=2_000,
        is_authorization=False,
    )


def _profile(
    root: Path,
    receipt: TargetResolutionReceipt,
    *,
    route: ProviderRouteProvenance | None = None,
) -> ResolvedSupervisorProfile:
    return ResolvedSupervisorProfile(
        profile_name="local-worktree",
        profile_source_cid=_cid("profile-source"),
        target_resolution_receipt_cid=receipt.content_id,
        mode=InvocationMode.WORKTREE,
        repository_root=str(root),
        state_root=str(root / "state"),
        run_namespace="fixture-run",
        policy_cid=_cid("policy"),
        principal_ref="did:key:local-owner",
        effect_ceiling_cid=_cid("effect-ceiling"),
        task_source_kind=TaskSourceKind.DUAL,
        task_source_path=str(root / "state" / "tasks.duckdb"),
        task_source_cid=_cid("task-source"),
        output_mode=OutputMode.BOTH,
        markdown_path=str(root / "state" / "plan.todo.md"),
        duckdb_path=str(root / "state" / "tasks.duckdb"),
        provider_route=route or receipt.provider_route,
        resource_budget=_resources(),
        validation_profile_cid=_cid("validation-profile"),
        lifecycle_health_contract_cid=_cid("health-contract"),
        coordination_shard=_coordination(root),
        replication=_replication(root),
        supervisor_argv=(
            "python",
            "-m",
            "ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_supervisor",
            "--state-dir",
            str(root / "state"),
        ),
        daemon_argv=(
            "python",
            "-m",
            "ipfs_accelerate_py.agent_supervisor.todo_daemon."
            "implementation_daemon",
            "--todo-path",
            str(root / "state" / "plan.todo.md"),
        ),
        environment_names=("CODEX_HOME", "GROK_API_KEY"),
        credential_handles=("env:GROK_API_KEY",),
        expected_effects=(
            ExpectedEffect.INSPECT_REPOSITORY,
            ExpectedEffect.WRITE_SUPERVISOR_STATE,
            ExpectedEffect.CREATE_ISOLATED_WORKTREE,
            ExpectedEffect.EDIT_ISOLATED_WORKTREE,
            ExpectedEffect.RUN_VALIDATION,
            ExpectedEffect.LAUNCH_LOCAL_PROCESS,
        ),
        worktree_strategy=WorktreeStrategy.ISOLATED,
        merge_target="main",
    )


def _launch(
    root: Path,
    invocation: SupervisorInvocationRequest,
    receipt: TargetResolutionReceipt,
    profile: ResolvedSupervisorProfile,
) -> LaunchPlan:
    return LaunchPlan(
        invocation_cid=invocation.content_id,
        target_resolution_receipt_cid=receipt.content_id,
        resolved_profile_cid=profile.content_id,
        working_directory=str(root),
        state_path=str(root / "state" / "run.json"),
        task_source_path=str(root / "state" / "tasks.duckdb"),
        supervisor_argv=profile.supervisor_argv,
        daemon_argv=profile.daemon_argv,
        environment_names=profile.environment_names,
        provider_route_cid=profile.provider_route.content_id,
        resource_budget_cid=profile.resource_budget.content_id,
        validation_profile_cid=profile.validation_profile_cid,
        lifecycle_profile_cid=_cid("lifecycle-profile"),
        coordination_shard=profile.coordination_shard,
        replication=profile.replication,
        expected_effects=profile.expected_effects,
        idempotency_key="invocation:fixture",
        adoption_key="adoption:fixture",
        lease_required=True,
        authorization_required=True,
        dry_run=False,
    )


def _handle(
    invocation: SupervisorInvocationRequest,
    receipt: TargetResolutionReceipt,
) -> RunHandle:
    return RunHandle(
        run_id=_cid("run"),
        run_revision=3,
        target_resolution_receipt_cid=receipt.content_id,
        invocation_cid=invocation.content_id,
        prompt_cid=invocation.prompt_cid,
        workflow_cid=_cid("workflow"),
        scan_cid=_cid("scan"),
        plan_cid=_cid("plan"),
        materialization_cid=_cid("materialization"),
        task_source_cid=_cid("task-source"),
        task_source_revision_cid=_cid("task-source-revision"),
        lifecycle_profile_cid=_cid("lifecycle-profile"),
        process_cid=_cid("process"),
        objective_cid=_cid("objective"),
        objective_revision_cid=_cid("objective-revision"),
        lease_id="lease:fixture",
        fencing_generation=7,
        state=RunState.RUNNING,
        health=RunHealth.HEALTHY,
        state_revision_cid=_cid("state-revision"),
        health_revision_cid=_cid("health-revision"),
        event_cursor="event:42",
        continuation_action=ContinuationAction.MONITOR,
        pending_approval_cid="",
        ambiguity_cid="",
        created_at_ms=1_000,
        updated_at_ms=2_000,
    )


def test_invocation_is_closed_canonical_and_prompt_body_free() -> None:
    invocation = _invocation()
    record = invocation.to_dict()
    encoded = invocation.to_json()

    assert invocation.transient_prompt_body == PROMPT.encode()
    assert "transient_prompt_body" not in record
    assert PROMPT not in encoded
    assert "GROK_API_KEY" not in encoded
    assert record["schema"].endswith("/invocation-request@1")
    assert record["content_id"] == invocation.content_id
    assert SupervisorInvocationRequest.from_json(encoded) == replace(
        invocation, transient_prompt_body=None
    )

    unknown = dict(record)
    unknown["prompt_body"] = PROMPT
    with pytest.raises(UnknownContractFieldError):
        SupervisorInvocationRequest.from_dict(unknown)
    with pytest.raises(ContractIdentityError):
        replace(invocation, prompt_cid=_cid("wrong-prompt"))


def test_decisions_record_selected_alternatives_and_ambiguity() -> None:
    selected = _decision("repository_root")
    assert selected.disposition is ResolutionDisposition.UNIQUE
    assert selected.candidates[0].value == selected.selected_value
    assert TargetInferenceDecision.from_json(selected.to_json()) == selected

    ambiguous = _decision(
        "repository_root",
        disposition=ResolutionDisposition.AMBIGUOUS,
    )
    assert ambiguous.unresolved
    assert len(ambiguous.candidates) == 2
    assert ambiguous.selected_value == ""

    with pytest.raises(EntrypointContractError, match="at least two"):
        TargetInferenceDecision(
            field_name="repository_root",
            disposition=ResolutionDisposition.AMBIGUOUS,
            selected_value="",
            selected_source=ResolutionSource.DISCOVERY,
            source_precedence=60,
            evidence_cid=_cid("evidence"),
            candidates=(),
            reason_codes=("multiple_viable_candidates",),
            effect=DecisionEffect.CONFIGURATION,
            override_accepted=False,
            fresh_until_ms=0,
            revalidation_rule=RevalidationRule.IMMUTABLE,
        )


def test_resolution_receipt_binds_every_field_but_is_not_authority(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    receipt = _receipt(root)
    record = receipt.to_dict()

    assert {item.field_name for item in receipt.decisions} == set(
        REQUIRED_TARGET_DECISION_FIELDS
    )
    assert receipt.unresolved_fields == ()
    assert receipt.authorizes_effects is False
    assert record["is_authorization"] is False
    assert record["coordination_shard"]["backend"] == "duckdb"
    assert record["replication"]["ipld_codec"] == "dag-json"
    assert TargetResolutionReceipt.from_json(receipt.to_json()) == receipt

    ambiguous = _receipt(root, ambiguous_field="repository_root")
    assert set(ambiguous.unresolved_fields) == {
        "checkout_id",
        "dirty_overlay",
        "nested_repositories",
        "repository_id",
        "repository_root",
        "scope",
        "submodules",
        "tree_id",
    }
    assert ambiguous.repository_root == ""
    assert ambiguous.scope_path == ""
    assert ambiguous.coordination_shard.writable is False
    assert ambiguous.replication.ipfs_publish is False
    assert ambiguous.worktree_strategy is WorktreeStrategy.NONE
    with pytest.raises(EntrypointContractError, match="not authorization"):
        replace(receipt, is_authorization=True)
    with pytest.raises(EntrypointContractError, match="missing"):
        replace(receipt, decisions=receipt.decisions[:-1])


def test_grok_route_codex_fallback_and_storage_roles_are_typed(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    grok = _route(ProviderSelection.GROK)
    codex = _route(ProviderSelection.CODEX)
    coordination = _coordination(root)
    replication = _replication(root)

    assert grok.preferred_provider == "grok"
    assert grok.fallback_provider == "codex"
    assert grok.fallback_reason is ProviderFallbackReason.NONE
    assert codex.fallback_receipt_cid
    assert codex.fallback_reason is (
        ProviderFallbackReason.PREFERRED_QUOTA_EXHAUSTED
    )
    assert codex.maximum_fallback_dispatches == 1
    assert coordination.backend == "duckdb"
    assert coordination.write_model == "single_writer_transactional_cas"
    assert coordination.remote_access == "owner_rpc"
    assert replication.grants_authority is False
    assert replication.partition_keys == DEFAULT_PARQUET_PARTITIONS
    assert replication.links_must_be_verified is True

    with pytest.raises(EntrypointContractError, match="committed fallback"):
        replace(codex, fallback_receipt_cid="")
    with pytest.raises(EntrypointContractError, match="must be duckdb"):
        replace(coordination, backend="parquet")
    with pytest.raises(EntrypointContractError, match="requires publication"):
        replace(replication, ipfs_publish=False)


def test_profile_and_launch_plan_bind_behavior_and_round_trip(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    invocation = _invocation()
    receipt = _receipt(root, invocation=invocation)
    profile = _profile(root, receipt)
    launch = _launch(root, invocation, receipt, profile)

    assert ResolvedSupervisorProfile.from_json(profile.to_json()) == profile
    assert LaunchPlan.from_json(launch.to_json()) == launch
    assert launch.coordination_shard.content_id == (
        profile.coordination_shard.content_id
    )
    assert launch.replication.content_id == profile.replication.content_id
    assert launch.provider_route_cid == profile.provider_route.content_id
    assert profile.profile_cid != replace(
        profile,
        merge_target="release",
    ).profile_cid
    assert launch.launch_plan_cid != replace(
        launch,
        adoption_key="adoption:other",
    ).launch_plan_cid

    with pytest.raises(SecretBearingRecordError):
        replace(
            profile,
            supervisor_argv=(*profile.supervisor_argv, "--token=secret"),
        )
    with pytest.raises(EntrypointContractError, match="lease and authorization"):
        replace(launch, lease_required=False)


def test_run_handle_content_id_tracks_bytes_and_semantic_id_ignores_timestamps(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    invocation = _invocation()
    receipt = _receipt(root, invocation=invocation)
    handle = _handle(invocation, receipt)
    later = replace(handle, created_at_ms=9_000, updated_at_ms=10_000)

    assert handle.handle_cid != later.handle_cid
    assert handle.semantic_id == later.semantic_id
    assert handle.to_dict() != later.to_dict()
    assert RunHandle.from_json(handle.to_json()) == handle
    assert replace(handle, run_revision=4).handle_cid != handle.handle_cid
    assert replace(handle, event_cursor="event:43").handle_cid != handle.handle_cid

    with pytest.raises(EntrypointContractError, match="needs_input"):
        replace(
            handle,
            state=RunState.NEEDS_INPUT,
            health=RunHealth.UNKNOWN,
            process_cid="",
            continuation_action=ContinuationAction.ASK_INPUT,
            ambiguity_cid="",
        )


def test_result_round_trip_and_closed_failure_states(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    invocation = _invocation()
    receipt = _receipt(root, invocation=invocation)
    profile = _profile(root, receipt)
    launch = _launch(root, invocation, receipt, profile)
    handle = _handle(invocation, receipt)
    result = SupervisorInvocationResult(
        invocation_cid=invocation.content_id,
        status=InvocationStatus.STARTED,
        target_resolution_receipt_cid=receipt.content_id,
        launch_plan_cid=launch.content_id,
        run_handle=handle,
        reason_codes=(),
        questions=(),
        continuation_action=ContinuationAction.MONITOR,
        effect_receipt_cids=(_cid("launch-effect"),),
        event_cursor="event:42",
        error_code="",
    )

    assert result.succeeded
    assert SupervisorInvocationResult.from_json(result.to_json()) == result
    assert PROMPT not in result.to_json()

    needs_input = SupervisorInvocationResult(
        invocation_cid=invocation.content_id,
        status=InvocationStatus.NEEDS_INPUT,
        target_resolution_receipt_cid=receipt.content_id,
        launch_plan_cid="",
        run_handle=None,
        reason_codes=("ambiguous_target",),
        questions=("select_repository_root",),
        continuation_action=ContinuationAction.ASK_INPUT,
        effect_receipt_cids=(),
        event_cursor="event:1",
        error_code="",
    )
    assert not needs_input.succeeded

    with pytest.raises(EntrypointContractError, match="require error_code"):
        replace(needs_input, status=InvocationStatus.DENIED, questions=())


def test_bounds_secrets_and_identity_tampering_fail_closed() -> None:
    over_bound = 1024 * 1024 + 1
    with pytest.raises(ContractBoundsError):
        InvocationBudget(max_prompt_bytes=over_bound)

    with pytest.raises(SecretBearingRecordError):
        SupervisorInvocationRequest.from_prompt(
            PROMPT,
            prompt_ref="Bearer secret-token",
        )

    invocation = _invocation()
    tampered = invocation.to_dict()
    tampered["repository_hint"] = "/different"
    with pytest.raises(ContractIdentityError):
        SupervisorInvocationRequest.from_dict(tampered)

    noncanonical = json.dumps(invocation.to_dict(), indent=2, sort_keys=True)
    with pytest.raises(EntrypointContractError, match="not the exact canonical"):
        SupervisorInvocationRequest.from_json(noncanonical)


@pytest.mark.parametrize(
    "source",
    [
        ResolutionSource.CANONICAL_REQUEST,
        ResolutionSource.EXPLICIT_OVERRIDE,
        ResolutionSource.REPOSITORY_HINT,
        ResolutionSource.DISCOVERY,
        ResolutionSource.BUILTIN_DEFAULT,
    ],
)
def test_authority_decisions_reject_untrusted_sources(
    source: ResolutionSource,
) -> None:
    candidate = TargetCandidate(
        field_name="principal",
        value="did:key:untrusted",
        source=source,
        source_precedence=1,
        evidence_cid=_cid(f"untrusted-{source.value}"),
    )
    with pytest.raises(EntrypointContractError, match="requires authenticated"):
        TargetInferenceDecision(
            field_name="principal",
            disposition=ResolutionDisposition.UNIQUE,
            selected_value=candidate.value,
            selected_source=source,
            source_precedence=1,
            evidence_cid=candidate.evidence_cid,
            candidates=(candidate,),
            reason_codes=(),
            effect=DecisionEffect.REQUIRES_AUTHORITY,
            override_accepted=source is ResolutionSource.EXPLICIT_OVERRIDE,
            fresh_until_ms=0,
            revalidation_rule=RevalidationRule.IMMUTABLE,
        )


def test_prompt_and_secret_material_cannot_enter_durable_text() -> None:
    invocation = _invocation()
    record = invocation.to_dict()
    record.pop("content_id")
    record["objective_hint"] = PROMPT
    with pytest.raises(SecretBearingRecordError, match="never persist"):
        SupervisorInvocationRequest.from_dict(record)

    with pytest.raises(SecretBearingRecordError, match="never persist"):
        SupervisorInvocationRequest.from_prompt(
            PROMPT,
            prompt_ref="prompt-broker:fixture",
            objective_hint=f"prefix:{PROMPT}",
        )
    with pytest.raises(SecretBearingRecordError, match="never persist"):
        SupervisorInvocationRequest.from_prompt("fix", prompt_ref="fix")
    with pytest.raises(SecretBearingRecordError):
        TargetCandidate(
            field_name="provider",
            value="sk-abcdefghijklmnopqrstuvwxyz",
            source=ResolutionSource.DISCOVERY,
            source_precedence=1,
            evidence_cid=_cid("secret-candidate"),
        )
    with pytest.raises(EntrypointContractError):
        TargetCandidate(
            field_name="objective",
            value=PROMPT,
            source=ResolutionSource.DISCOVERY,
            source_precedence=1,
            evidence_cid=_cid("prompt-candidate"),
        )


def test_receipt_projection_containment_and_owner_binding_fail_closed(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    invocation = _invocation()
    receipt = _receipt(root, invocation=invocation)

    with pytest.raises(EntrypointContractError, match="repository_root decision"):
        replace(receipt, repository_root="/tmp/different-repository")
    with pytest.raises(EntrypointContractError, match="contained"):
        replace(receipt, markdown_path="/tmp/outside-state/plan.todo.md")

    outside_scope = "/tmp/outside-scope"
    scope_decisions = tuple(
        _decision("scope", value=outside_scope)
        if item.field_name == "scope"
        else item
        for item in receipt.decisions
    )
    with pytest.raises(EntrypointContractError, match="scope_path"):
        replace(
            receipt,
            scope_path=outside_scope,
            decisions=scope_decisions,
        )

    profile = _profile(root, receipt)
    with pytest.raises(EntrypointContractError, match="owner principal"):
        replace(profile, principal_ref="did:key:not-the-owner")


def test_only_quota_fallback_requires_independent_review() -> None:
    codex = _route(ProviderSelection.CODEX)
    with pytest.raises(EntrypointContractError, match="independent reviewer"):
        replace(codex, independent_review_required=False)
    with pytest.raises(EntrypointContractError, match="quota exhaustion"):
        replace(
            codex,
            fallback_reason=ProviderFallbackReason.PREFERRED_PRE_EFFECT_FAILURE,
        )


def test_contract_cid_codecs_are_semantically_typed() -> None:
    invocation = _invocation()
    with pytest.raises(ContractIdentityError):
        replace(invocation, prompt_cid=_cid("structured-not-prompt"))
    with pytest.raises(ContractIdentityError):
        replace(
            _route(),
            observed_capability_cid=invocation.prompt_cid,
        )


def test_launch_handle_and_result_cross_links_fail_closed(tmp_path: Path) -> None:
    root = (tmp_path / "repo").resolve()
    invocation = _invocation()
    receipt = _receipt(root, invocation=invocation)
    profile = _profile(root, receipt)
    launch = _launch(root, invocation, receipt, profile)
    handle = _handle(invocation, receipt)
    result = SupervisorInvocationResult(
        invocation_cid=invocation.content_id,
        status=InvocationStatus.RUNNING,
        target_resolution_receipt_cid=receipt.content_id,
        launch_plan_cid=launch.content_id,
        run_handle=handle,
        reason_codes=(),
        questions=(),
        continuation_action=ContinuationAction.MONITOR,
        effect_receipt_cids=(_cid("running-effect"),),
        event_cursor=handle.event_cursor,
        error_code="",
    )

    with pytest.raises(EntrypointContractError, match="present together"):
        replace(handle, lease_id="")
    with pytest.raises(EntrypointContractError, match="status"):
        replace(result, status=InvocationStatus.COMPLETED)
    with pytest.raises(EntrypointContractError, match="links must match"):
        replace(
            result,
            run_handle=replace(handle, invocation_cid=_cid("other-invocation")),
        )
    with pytest.raises(EntrypointContractError, match="writable DuckDB owner"):
        replace(
            launch,
            coordination_shard=replace(
                launch.coordination_shard,
                writable=False,
            ),
        )


def test_malformed_nested_input_and_aggregate_bounds_are_stable(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "repo").resolve()
    receipt = _receipt(root)
    malformed = receipt.to_dict()
    malformed.pop("content_id")
    malformed["decisions"] = 1
    with pytest.raises(EntrypointContractError, match="sequence"):
        TargetResolutionReceipt.from_dict(malformed)

    profile = _profile(root, receipt)
    reversed_effects = replace(
        profile,
        expected_effects=tuple(reversed(profile.expected_effects)),
    )
    assert reversed_effects.content_id == profile.content_id

    huge_argv = ("x" * 4096,) * 256
    with pytest.raises(ContractBoundsError, match="record byte bound"):
        replace(
            profile,
            supervisor_argv=huge_argv,
            daemon_argv=huge_argv,
        )


def test_candidate_alternatives_have_canonical_order() -> None:
    decision = _decision(
        "repository_root",
        value="/srv/repository",
        disposition=ResolutionDisposition.AMBIGUOUS,
    )
    permuted = replace(decision, candidates=tuple(reversed(decision.candidates)))
    assert permuted.candidates == decision.candidates
    assert permuted.content_id == decision.content_id
