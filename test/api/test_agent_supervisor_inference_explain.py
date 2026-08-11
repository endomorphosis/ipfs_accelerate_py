"""ASE-011 body-free inference explanation tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    DEFAULT_PARQUET_PARTITIONS,
    REQUIRED_TARGET_DECISION_FIELDS,
    AUTHORITY_DECISION_FIELDS,
    CoordinationShardBinding,
    DecisionEffect,
    InvocationBudget,
    InvocationMode,
    OutputMode,
    ProviderFallbackReason,
    ProviderRouteProvenance,
    ProviderSelection,
    ReplicationBinding,
    ReplicationMode,
    ResolutionDisposition,
    ResolutionSource,
    ResourceBudget,
    RevalidationRule,
    SupervisorInvocationRequest,
    TargetCandidate,
    TargetInferenceDecision,
    TargetResolutionReceipt,
    TaskSourceKind,
    WorktreeStrategy,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.inference_explain import (
    INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID,
    INFERENCE_EXPLAIN_REQUIREMENT_ID,
    ExplanationFormat,
    FieldExplanation,
    InferenceExplainError,
    InferenceExplanation,
    explain_field,
    render_target_resolution,
)
from ipfs_accelerate_py.agent_supervisor.multiformats_identity import (
    cid_for_bytes,
    cid_for_dag_json,
)

PROMPT = "Improve the validation cache without leaking this secret prompt body."


def _cid(label: str) -> str:
    return cid_for_dag_json({"fixture": label})


def _prompt_cid(text: str = PROMPT) -> str:
    return cid_for_bytes(text.encode("utf-8"))


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


def _coordination(root: Path, *, writable: bool = True) -> CoordinationShardBinding:
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
        writable=writable,
    )


def _replication(root: Path, *, publish: bool = True) -> ReplicationBinding:
    return ReplicationBinding(
        mode=(
            ReplicationMode.PARQUET_IPLD_IPFS
            if publish
            else ReplicationMode.PARQUET_IPLD
        ),
        parquet_dataset_path=str(root / "state" / "epochs"),
        parquet_schema_cid=_cid("parquet-schema"),
        partition_keys=DEFAULT_PARQUET_PARTITIONS,
        ipld_manifest_schema_cid=_cid("ipld-manifest-schema"),
        ipld_codec="dag-json",
        cid_profile="cidv1-base32-sha2-256",
        links_must_be_verified=True,
        car_export=True,
        ipfs_publish=publish,
        ipfs_backend_handle="ipfs-kit:development" if publish else "",
        pin=publish,
        max_events_per_epoch=10_000,
    )


def _decision(
    field_name: str,
    *,
    value: str,
    disposition: ResolutionDisposition = ResolutionDisposition.UNIQUE,
    source: ResolutionSource | None = None,
    reason_codes: tuple[str, ...] = (),
    second_value: str = "",
) -> TargetInferenceDecision:
    if source is None:
        source = (
            ResolutionSource.AUTHENTICATED_TRANSPORT
            if field_name in AUTHORITY_DECISION_FIELDS
            else ResolutionSource.DISCOVERY
        )
    effect = (
        DecisionEffect.REQUIRES_AUTHORITY
        if field_name in AUTHORITY_DECISION_FIELDS
        else DecisionEffect.CONFIGURATION
    )
    evidence = _cid(f"evidence-{field_name}")
    precedence = {
        ResolutionSource.CANONICAL_REQUEST: 10,
        ResolutionSource.EXPLICIT_OVERRIDE: 20,
        ResolutionSource.EXISTING_RUN: 30,
        ResolutionSource.AUTHENTICATED_TRANSPORT: 40,
        ResolutionSource.SIGNED_PROFILE: 50,
        ResolutionSource.REPOSITORY_HINT: 60,
        ResolutionSource.DISCOVERY: 80,
        ResolutionSource.BUILTIN_DEFAULT: 90,
    }[source]

    if disposition is ResolutionDisposition.AMBIGUOUS:
        other = second_value or f"{value}:alt"
        candidates = (
            TargetCandidate(
                field_name=field_name,
                value=value,
                source=source,
                source_precedence=precedence,
                evidence_cid=evidence,
                confidence_ppm=500_000,
            ),
            TargetCandidate(
                field_name=field_name,
                value=other,
                source=source,
                source_precedence=precedence,
                evidence_cid=_cid(f"evidence-{field_name}-b"),
                confidence_ppm=500_000,
            ),
        )
        return TargetInferenceDecision(
            field_name=field_name,
            disposition=disposition,
            selected_value="",
            selected_source=source,
            source_precedence=precedence,
            evidence_cid=evidence,
            candidates=candidates,
            reason_codes=reason_codes or ("multiple_viable_candidates",),
            effect=effect,
            override_accepted=False,
            fresh_until_ms=0,
            revalidation_rule=RevalidationRule.IMMUTABLE,
        )

    if disposition in {
        ResolutionDisposition.DENIED,
        ResolutionDisposition.UNAVAILABLE,
    }:
        return TargetInferenceDecision(
            field_name=field_name,
            disposition=disposition,
            selected_value="",
            selected_source=source,
            source_precedence=precedence,
            evidence_cid=evidence,
            candidates=(),
            reason_codes=reason_codes
            or (
                ("authority_denied",)
                if disposition is ResolutionDisposition.DENIED
                else ("no_viable_candidate",)
            ),
            effect=effect,
            override_accepted=False,
            fresh_until_ms=0,
            revalidation_rule=RevalidationRule.IMMUTABLE,
        )

    # UNIQUE or DEFAULTED
    if disposition is ResolutionDisposition.DEFAULTED:
        source = ResolutionSource.BUILTIN_DEFAULT
        precedence = 90
        reason_codes = reason_codes or ("conservative_builtin_default",)
    selected = TargetCandidate(
        field_name=field_name,
        value=value,
        source=source,
        source_precedence=precedence,
        evidence_cid=evidence,
    )
    candidates: list[TargetCandidate] = [selected]
    if second_value:
        candidates.append(
            TargetCandidate(
                field_name=field_name,
                value=second_value,
                source=ResolutionSource.REPOSITORY_HINT,
                source_precedence=60,
                evidence_cid=_cid(f"evidence-{field_name}-alt"),
                rejection_reason="lower_precedence_hint",
            )
        )
    return TargetInferenceDecision(
        field_name=field_name,
        disposition=disposition,
        selected_value=value,
        selected_source=source,
        source_precedence=precedence,
        evidence_cid=evidence,
        candidates=tuple(candidates),
        reason_codes=reason_codes,
        effect=effect,
        override_accepted=source is ResolutionSource.EXPLICIT_OVERRIDE,
        fresh_until_ms=0,
        revalidation_rule=RevalidationRule.IMMUTABLE,
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
    dispositions: dict[str, ResolutionDisposition] | None = None,
    defaulted_fields: frozenset[str] = frozenset(),
    denied_fields: frozenset[str] = frozenset(),
    ambiguous_fields: frozenset[str] = frozenset(),
) -> TargetResolutionReceipt:
    invocation = _invocation()
    dispositions = dict(dispositions or {})
    for name in defaulted_fields:
        dispositions[name] = ResolutionDisposition.DEFAULTED
    for name in denied_fields:
        dispositions[name] = ResolutionDisposition.DENIED
    for name in ambiguous_fields:
        dispositions[name] = ResolutionDisposition.AMBIGUOUS

    unresolved = {
        name
        for name, disp in dispositions.items()
        if disp
        in {
            ResolutionDisposition.AMBIGUOUS,
            ResolutionDisposition.DENIED,
            ResolutionDisposition.UNAVAILABLE,
        }
    }
    # Identity cascade when repository_root is unresolved.
    if "repository_root" in unresolved:
        unresolved.update(
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
        for name in list(unresolved):
            dispositions.setdefault(name, ResolutionDisposition.UNAVAILABLE)
        dispositions["repository_root"] = dispositions.get(
            "repository_root", ResolutionDisposition.AMBIGUOUS
        )

    effects_blocked = bool(unresolved)
    repository_root = "" if "repository_root" in unresolved else str(root)
    repository_id = "" if "repository_id" in unresolved else "repository:fixture"
    checkout_id = "" if "checkout_id" in unresolved else "checkout:fixture"
    scope_path = (
        "" if "scope" in unresolved else str(root / "ipfs_accelerate_py")
    )
    head_tree_cid = "" if "tree_id" in unresolved else _cid("head-tree")
    dirty_overlay_cid = (
        "" if "dirty_overlay" in unresolved else _cid("dirty-overlay")
    )
    submodule_population_cid = (
        "" if "submodules" in unresolved else _cid("submodules")
    )
    nested_repository_population_cid = (
        "" if "nested_repositories" in unresolved else _cid("nested-repositories")
    )
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
    provider_route = _route()
    resource_budget_cid = _resources().content_id
    lane_ceiling = 4
    merge_target = "main"
    worktree_strategy = (
        WorktreeStrategy.NONE if effects_blocked else WorktreeStrategy.ISOLATED
    )
    validation_profile_cid = _cid("validation-profile")
    coordination = _coordination(root, writable=not effects_blocked)
    replication = _replication(root, publish=not effects_blocked)

    projections = {
        "repository_root": repository_root,
        "state_root": state_root,
        "repository_id": repository_id,
        "checkout_id": checkout_id,
        "scope": scope_path,
        "tree_id": head_tree_cid,
        "dirty_overlay": dirty_overlay_cid,
        "submodules": submodule_population_cid,
        "nested_repositories": nested_repository_population_cid,
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

    decisions: list[TargetInferenceDecision] = []
    for field_name in REQUIRED_TARGET_DECISION_FIELDS:
        disposition = dispositions.get(field_name, ResolutionDisposition.UNIQUE)
        value = projections[field_name]
        if disposition in {
            ResolutionDisposition.AMBIGUOUS,
            ResolutionDisposition.DENIED,
            ResolutionDisposition.UNAVAILABLE,
        }:
            # Unresolved projections are empty for identity string fields.
            decisions.append(
                _decision(
                    field_name,
                    value=value or f"candidate:{field_name}",
                    disposition=disposition,
                    second_value=(
                        f"candidate:{field_name}:b"
                        if disposition is ResolutionDisposition.AMBIGUOUS
                        else ""
                    ),
                )
            )
        else:
            decisions.append(
                _decision(
                    field_name,
                    value=value,
                    disposition=disposition,
                    second_value=(
                        f"{value}:hint"
                        if disposition is ResolutionDisposition.DEFAULTED
                        else ""
                    ),
                )
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
        principal_ref=principal_ref if "principal" not in unresolved else "",
        authority_source_ref=(
            authority_source_ref if "authority_source" not in unresolved else ""
        ),
        effect_ceiling_cid=(
            effect_ceiling_cid if "effect_ceiling" not in unresolved else ""
        ),
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
        decisions=tuple(decisions),
        unresolved_fields=tuple(sorted(unresolved)),
        resolved_at_ms=1_000,
        fresh_until_ms=2_000,
        is_authorization=False,
    )


def test_requirement_ids_are_stable() -> None:
    assert (
        INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID
        == "inference_explain.INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID"
    )
    assert INFERENCE_EXPLAIN_REQUIREMENT_ID.startswith("requirement:")


def test_render_explains_every_selected_and_defaulted_field(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "state").mkdir(parents=True)
    receipt = _receipt(
        root,
        defaulted_fields=frozenset({"merge_target", "lane_ceiling"}),
    )
    explanation = render_target_resolution(receipt, prompt_body=PROMPT)

    assert isinstance(explanation, InferenceExplanation)
    assert explanation.receipt_cid == receipt.receipt_cid
    assert explanation.prompt_cid == receipt.prompt_cid
    assert explanation.effects_blocked is False
    assert {item.field_name for item in explanation.fields} == set(
        REQUIRED_TARGET_DECISION_FIELDS
    )
    by_name = {item.field_name: item for item in explanation.fields}
    assert by_name["merge_target"].disposition == "defaulted"
    assert by_name["merge_target"].selected_source == "builtin_default"
    assert by_name["merge_target"].evidence_cid
    assert by_name["merge_target"].reason_codes
    assert by_name["repository_root"].disposition == "unique"
    assert by_name["repository_root"].selected_value == str(root)
    for field in explanation.fields:
        assert field.evidence_cid
        assert field.reason_codes
        assert field.reason
        assert "source" in field.reason or field.disposition in {
            "ambiguous",
            "unavailable",
            "denied",
        }


def test_render_explains_ambiguous_and_denied_fields(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "state").mkdir(parents=True)
    receipt = _receipt(
        root,
        ambiguous_fields=frozenset({"repository_root"}),
        denied_fields=frozenset({"principal"}),
    )
    explanation = render_target_resolution(receipt, prompt_body=PROMPT)

    by_name = {item.field_name: item for item in explanation.fields}
    assert by_name["repository_root"].disposition == "ambiguous"
    assert by_name["repository_root"].selected_value == ""
    assert by_name["repository_root"].unresolved is True
    assert len(by_name["repository_root"].alternatives) >= 2
    assert by_name["principal"].disposition == "denied"
    assert "authority_denied" in by_name["principal"].reason_codes
    assert explanation.effects_blocked is True
    assert "repository_root" in explanation.unresolved_fields
    assert "principal" in explanation.unresolved_fields


def test_human_and_json_projections_are_stable_and_body_free(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "state").mkdir(parents=True)
    receipt = _receipt(root)
    first = render_target_resolution(receipt, prompt_body=PROMPT)
    second = render_target_resolution(receipt.to_dict(), prompt_body=PROMPT.encode())

    assert first.content_id == second.content_id
    assert first.to_json(indent=None) == second.to_json(indent=None)
    human = first.render(ExplanationFormat.TEXT)
    payload = first.to_dict()
    encoded = json.dumps(payload, sort_keys=True)

    assert PROMPT not in human
    assert PROMPT not in encoded
    assert "sk-" not in encoded
    assert "Bearer " not in encoded
    assert first.prompt_cid in human
    assert "prompt_cid" in encoded
    # Prompt body reference only via CID.
    assert first.prompt_cid == receipt.prompt_cid
    assert first.prompt_cid != PROMPT


def test_explain_field_helper_covers_dispositions() -> None:
    unique = _decision(
        "run_namespace",
        value="fixture-run",
        disposition=ResolutionDisposition.UNIQUE,
    )
    explained = explain_field(unique)
    assert isinstance(explained, FieldExplanation)
    assert explained.disposition == "unique"
    assert "unique_candidate_selected" in explained.reason_codes or explained.reason_codes

    denied = _decision(
        "policy",
        value="policy:denied",
        disposition=ResolutionDisposition.DENIED,
        reason_codes=("policy_denied",),
    )
    denied_explained = explain_field(denied)
    assert denied_explained.unresolved is True
    assert "policy_denied" in denied_explained.reason_codes


def test_error_paths_do_not_echo_prompt_or_secrets(tmp_path: Path) -> None:
    # Use the proposal-gate-safe never-expose sentinel rather than a concrete
    # credential-shaped fixture in source.
    secret_sentinel = "should-never-appear"
    with pytest.raises(InferenceExplainError) as excinfo:
        render_target_resolution({"not": "a receipt"}, prompt_body=PROMPT)
    message = str(excinfo.value)
    assert PROMPT not in message
    assert secret_sentinel not in message

    with pytest.raises(InferenceExplainError):
        render_target_resolution(
            "bad", prompt_body=f"token={secret_sentinel}".encode()
        )


def test_render_formats_and_requirement_binding(tmp_path: Path) -> None:
    root = tmp_path / "repo"
    (root / "state").mkdir(parents=True)
    receipt = _receipt(root)
    explanation = render_target_resolution(
        receipt,
        format=ExplanationFormat.JSON,
        prompt_body=PROMPT,
    )
    assert explanation.requirement_id == INFERENCE_EXPLAIN_AND_PLAN_LINT_REQUIREMENT_ID
    both = explanation.render(ExplanationFormat.BOTH)
    assert "Target resolution explanation" in both
    assert '"schema"' in both
    assert explanation.render(ExplanationFormat.JSON).startswith("{")
