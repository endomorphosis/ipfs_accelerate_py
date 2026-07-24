from __future__ import annotations

import copy
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py.agent_supervisor.analyzer_health import (
    AnalyzerHealthReport,
    AnalyzerHealthStatus,
    AnalyzerHealthThresholds,
)
from ipfs_accelerate_py.agent_supervisor.control_contracts import (
    CONTROL_MUTATION_GUARD_ACCEPTANCE_CRITERIA,
    CONTROL_MUTATION_GUARD_COMPLETION_ANALYZER_VERSION,
    CONTROL_MUTATION_GUARD_COMPLETION_CONFIGURATION_REVISION,
    CONTROL_MUTATION_GUARD_OBJECTIVE_ID,
    CONTROL_MUTATION_GUARD_OBJECTIVE_REVISION,
    CONTROL_MUTATION_GUARD_REJECTION_SCENARIOS,
    CONTROL_MUTATION_GUARD_REQUIREMENT_ID,
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlContractError,
    ControlMutationCompletionMemberHealth,
    ControlMutationCompletionQuorumEvidence,
    ControlMutationGuardEvidence,
    ControlSurface,
    EffectKind,
    ExpectedEffect,
    IdempotencyKey,
    LifecycleAction,
    LifecycleCommand,
    MutationGuardRejection,
    MutationGuardExecutionObservation,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationResult,
    OperationStatus,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    GoalState,
)
from ipfs_accelerate_py.agent_supervisor.goal_coverage import (
    AcceptanceCoverage,
    CoverageStatus,
    GoalCoverageMap,
    ValidationReceiptCoverage,
)
from ipfs_accelerate_py.agent_supervisor.scan_receipts import (
    ExhaustionBinding,
    ExhaustionQuorumMember,
    ExhaustionQuorumResult,
)
from ipfs_accelerate_py.agent_supervisor.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    SupervisorControlService,
)


def _binding(repo_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repo_root),
        "state_root": str(state_root),
        "repository_id": "repo:fixture",
        "tree_id": "tree:abc",
        "objective_id": CONTROL_MUTATION_GUARD_OBJECTIVE_ID,
        "objective_revision": "objective:1",
        "policy_id": "policy:control",
        "policy_revision": "policy:1",
        "caller": "operator:test",
    }


def _request(
    repo_root: Path,
    state_root: Path,
    operation: Operation,
    *,
    dry_run: bool,
) -> OperationRequest:
    binding = _binding(repo_root, state_root)
    effect = ExpectedEffect(
        effect_id=f"{operation.value}:supervisor",
        kind=EffectKind.LIFECYCLE_TRANSITION,
        resource="supervisor:fixture",
        paths=("supervisor.json",),
        description=f"Transition with {operation.value}",
    )
    values: dict[str, Any] = {
        "operation": operation,
        **binding,
        "parameters": {
            "target_id": "supervisor:fixture",
            "reason": "operator maintenance",
            "requested_state": operation.value,
        },
        "expected_effects": (effect,),
        "dry_run": dry_run,
    }
    if not dry_run:
        values.update(
            {
                "idempotency": IdempotencyKey(
                    key=f"lifecycle:{operation.value}:1",
                    operation=operation,
                    caller=binding["caller"],
                    repository_id=binding["repository_id"],
                    objective_id=binding["objective_id"],
                ),
                "authorization": AuthorizationDecision(
                    verdict=AuthorizationVerdict.PERMIT,
                    operation=operation,
                    granted_authority=OperationAuthority.MUTATION,
                    **binding,
                    lease_id="lease:7",
                    fencing_epoch=7,
                    authorized_effect_ids=(effect.effect_id,),
                    grant_ids=("grant:operator",),
                    evaluated_at_ms=1_000,
                    expires_at_ms=2_000,
                ),
                "lease_id": "lease:7",
                "fencing_epoch": 7,
            }
        )
    return OperationRequest(**values)


def _command(operation: Operation, *, dry_run: bool) -> LifecycleCommand:
    return LifecycleCommand(
        action=LifecycleAction(operation.value),
        target_id="supervisor:fixture",
        reason="operator maintenance",
        requested_state=operation.value,
        dry_run=dry_run,
    )


def _service(
    repo_root: Path,
    state_root: Path,
    calls: list[str],
) -> SupervisorControlService:
    def transition(request: OperationRequest) -> BackendResponse:
        calls.append(request.request_id)
        effect_id = request.expected_effects[0].effect_id
        return BackendResponse(
            data={
                "previous_state": "healthy",
                "state": request.operation.value,
            },
            changed=True,
            applied_effect_ids=(effect_id,),
        )

    return SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={
            operation: transition
            for operation in (
                Operation.START,
                Operation.PAUSE,
                Operation.RESUME,
                Operation.DRAIN,
                Operation.STOP,
            )
        },
        lease_validator=lambda request: (
            request.lease_id == "lease:7" and request.fencing_epoch == 7
        ),
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_500,
    )


def _mutation_guard_witness(
    repo_root: Path,
    state_root: Path,
    calls: list[str],
) -> ControlMutationGuardEvidence:
    service = _service(repo_root, state_root, calls)
    request = _request(
        repo_root, state_root, Operation.PAUSE, dry_run=False
    )
    before = service.mutation_runtime_state()
    result = service.execute(request)
    after_result = service.mutation_runtime_state()
    replay = service.execute(request)
    after_replay = service.mutation_runtime_state()
    canonical = request.to_record()

    canonical.pop("content_id", None)
    payloads: dict[str, dict[str, Any]] = {}

    unauthorized = dict(canonical)
    unauthorized.pop("authorization")
    payloads["unauthorized"] = unauthorized

    unscoped = dict(canonical)
    idempotency = dict(unscoped["idempotency"])
    idempotency.pop("content_id")
    idempotency["objective_id"] = "objective:outside-request-scope"
    unscoped["idempotency"] = idempotency
    payloads["unscoped_idempotency"] = unscoped

    unfenced = dict(canonical)
    unfenced.pop("lease_id")
    unfenced.pop("fencing_epoch")
    payloads["unfenced"] = unfenced

    stale = dict(canonical)
    stale["tree_id"] = f"{request.tree_id}:stale-request-binding"
    payloads["stale_binding"] = stale

    escaping = dict(canonical)
    parameters = dict(escaping["parameters"])
    parameters["target_path"] = "../outside-repository"
    escaping["parameters"] = parameters
    payloads["path_escape"] = escaping

    undeclared = dict(canonical)
    undeclared["expected_effects"] = ()
    payloads["undeclared_effect"] = undeclared

    assert set(payloads) == set(CONTROL_MUTATION_GUARD_REJECTION_SCENARIOS)
    rejections: list[MutationGuardRejection] = []
    for surface in ControlSurface:
        for scenario in CONTROL_MUTATION_GUARD_REJECTION_SCENARIOS:
            payload = payloads[scenario]
            try:
                if surface is ControlSurface.PYTHON:
                    service.execute(payload)
                else:
                    OperationRequest.from_dict(payload)
            except (ControlContractError, ValueError) as exc:
                error_type = type(exc).__name__
            else:
                raise AssertionError(f"{scenario} unexpectedly decoded")
            rejections.append(
                MutationGuardRejection(
                    scenario=scenario,
                    surface=surface,
                    request_payload=payload,
                    error_type=error_type,
                    dispatch_count_before=after_replay.dispatch_count,
                    dispatch_count_after=after_replay.dispatch_count,
                )
            )

    return ControlMutationGuardEvidence(
        repository_tree=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        policy_revision=request.policy_revision,
        request=request,
        result=result,
        replay_result=replay,
        execution=MutationGuardExecutionObservation(
            request_id=request.request_id,
            result_id=result.result_id,
            audit_receipt_id=result.audit_receipt_id,
            before=before,
            after_result=after_result,
            after_replay=after_replay,
        ),
        rejections=tuple(rejections),
    )


def test_lifecycle_dry_run_binds_typed_command_without_dispatch(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls: list[str] = []
    service = _service(repo_root, state_root, calls)
    request = _request(
        repo_root, state_root, Operation.PAUSE, dry_run=True
    )

    result = service.lifecycle(request, _command(Operation.PAUSE, dry_run=True))

    assert result.succeeded
    assert result.authority is OperationAuthority.PROPOSAL
    assert result.preview is not None
    assert result.preview.would_change is True
    assert calls == []


def test_lifecycle_command_binds_reason_and_requested_state(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root, [])
    request = _request(
        repo_root, state_root, Operation.DRAIN, dry_run=True
    )

    with pytest.raises(ValueError, match="reason"):
        service.lifecycle(
            request,
            LifecycleCommand(
                action=LifecycleAction.DRAIN,
                target_id="supervisor:fixture",
                reason="different reason",
                requested_state="drain",
                dry_run=True,
            ),
        )
    with pytest.raises(ValueError, match="requested_state"):
        service.lifecycle(
            request,
            LifecycleCommand(
                action=LifecycleAction.DRAIN,
                target_id="supervisor:fixture",
                reason="operator maintenance",
                requested_state="paused",
                dry_run=True,
            ),
        )


def test_authorized_lifecycle_mutation_is_fenced_audited_and_idempotent(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls: list[str] = []
    service = _service(repo_root, state_root, calls)
    request = _request(
        repo_root, state_root, Operation.PAUSE, dry_run=False
    )
    command = _command(Operation.PAUSE, dry_run=False)

    first = service.lifecycle(request, command)
    replay = service.lifecycle(request, command)

    assert first is replay
    assert first.status is OperationStatus.SUCCEEDED
    assert first.data == {
        "previous_state": "healthy",
        "state": "pause",
    }
    assert first.effects[0].applied is True
    assert first.audit_receipt_id
    assert calls == [request.request_id]


def test_mutation_guard_evidence_replays_all_required_fail_closed_cases(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls: list[str] = []
    evidence = _mutation_guard_witness(repo_root, state_root, calls)
    request = evidence.request
    result = evidence.result
    assert isinstance(request, OperationRequest)
    assert isinstance(result, OperationResult)

    assert evidence.proved_requirement_ids == (
        CONTROL_MUTATION_GUARD_REQUIREMENT_ID,
    )
    assert {
        (item.surface, item.scenario) for item in evidence.rejections
    } == {
        (surface, scenario)
        for surface in ControlSurface
        for scenario in CONTROL_MUTATION_GUARD_REJECTION_SCENARIOS
    }
    assert calls == [request.request_id]
    assert ControlMutationGuardEvidence.from_dict(evidence.to_record()) == evidence

    overbroad_effect_scope = request.to_record()
    overbroad_effect_scope.pop("content_id")
    authorization = dict(overbroad_effect_scope["authorization"])
    authorization.pop("content_id")
    authorization["authorized_effect_ids"] = [
        request.expected_effects[0].effect_id,
        "effect:not-declared-by-request",
    ]
    overbroad_effect_scope["authorization"] = authorization
    with pytest.raises(
        ControlContractError,
        match="effect scope must exactly match",
    ):
        OperationRequest.from_dict(overbroad_effect_scope)
    assert calls == [request.request_id]

    foreign_objective = evidence.to_record()
    foreign_objective.pop("content_id")
    foreign_objective["objective_id"] = "ASI-G103"
    with pytest.raises(ControlContractError, match="objective_id"):
        ControlMutationGuardEvidence.from_dict(foreign_objective)

    detached_rejection = evidence.to_record()
    detached_rejection.pop("content_id")
    rejection = dict(detached_rejection["rejections"][0])
    rejection.pop("content_id")
    rejection_payload = dict(rejection["request_payload"])
    rejection_payload["tree_id"] = "tree:foreign"
    rejection["request_payload"] = rejection_payload
    detached_rejection["rejections"] = (
        rejection,
        *detached_rejection["rejections"][1:],
    )
    with pytest.raises(ControlContractError, match="bound request"):
        ControlMutationGuardEvidence.from_dict(detached_rejection)

    for field, message in (
        ("result_id", "detached"),
        ("audit_receipt_id", "bound audit receipt"),
    ):
        tampered = evidence.to_record()
        tampered.pop("content_id")
        execution = dict(tampered["execution"])
        execution.pop("content_id")
        execution[field] = "sha256:" + ("0" * 64)
        tampered["execution"] = execution
        with pytest.raises(ControlContractError, match=message):
            ControlMutationGuardEvidence.from_dict(tampered)

    duplicate_dispatch = evidence.to_record()
    duplicate_dispatch.pop("content_id")
    execution = dict(duplicate_dispatch["execution"])
    execution.pop("content_id")
    after_replay_record = dict(execution["after_replay"])
    after_replay_record.pop("content_id")
    after_replay_record["dispatch_count"] = 2
    execution["after_replay"] = after_replay_record
    duplicate_dispatch["execution"] = execution
    with pytest.raises(ControlContractError, match="must not dispatch"):
        ControlMutationGuardEvidence.from_dict(duplicate_dispatch)

    mismatched_receipt = result.to_record()
    mismatched_receipt.pop("content_id")
    effect = dict(mismatched_receipt["effects"][0])
    effect.pop("content_id")
    effect["receipt_id"] = "sha256:" + ("f" * 64)
    mismatched_receipt["effects"] = [effect]
    with pytest.raises(ControlContractError, match="must match"):
        OperationResult.from_dict(mismatched_receipt)


def test_g104_completion_requires_bound_validation_health_and_quorum(
    tmp_path: Path,
) -> None:
    """ASI-077: a mutation witness cannot self-certify objective completion."""

    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    operational = _mutation_guard_witness(repo_root, state_root, [])
    now = datetime(2026, 7, 24, 18, 0, tzinfo=timezone.utc)
    command = (
        "python -m pytest test/api/test_agent_supervisor_control_plane.py "
        "test/api/test_agent_supervisor_control_lifecycle.py "
        "test/test_unified_cli_agent_supervisor.py "
        "test/mcp_server/test_agent_supervisor_tools.py -q"
    )
    validation_binding = {
        "status": "passed",
        "tree_id": operational.repository_tree,
        "requirement_id": CONTROL_MUTATION_GUARD_REQUIREMENT_ID,
        "objective_id": CONTROL_MUTATION_GUARD_OBJECTIVE_ID,
        "operational_receipt_id": operational.content_id,
        "validation_policy_id": operational.policy_id,
        "policy_revision": operational.policy_revision,
        "command": command,
    }
    completion_evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-077",
            producer_kind="task",
            validation_receipt=validation_binding,
            validation_passed=True,
            repository_tree=operational.repository_tree,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-077:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            CONTROL_MUTATION_GUARD_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage_receipts = [
        ValidationReceiptCoverage(
            receipt_id=item.provenance_cid,
            task_id="ASI-077",
            criterion=item.acceptance_criterion,
            command=command,
            status=CoverageStatus.VERIFIED,
            passed=True,
            repository_tree=operational.repository_tree,
            observed_at=now.isoformat(),
            provenance_cid=item.provenance_cid,
            explanation="fresh passing ASI-077 criterion validation",
            outcome="passed",
            reason_code="validation_verified",
            fresh=True,
        )
        for item in completion_evidence
    ]
    coverage = GoalCoverageMap(
        criteria=[
            AcceptanceCoverage(
                criterion_id=f"criterion:g104:{index}",
                goal_id=CONTROL_MUTATION_GUARD_OBJECTIVE_ID,
                criterion=criterion,
                status=CoverageStatus.VERIFIED,
                changed_files=[
                    "ipfs_accelerate_py/agent_supervisor/"
                    "control_contracts.py"
                ],
                validation_receipt_ids=[
                    completion_evidence[index - 1].provenance_cid
                ],
                explanation="implementation and validation are exact",
            )
            for index, criterion in enumerate(
                CONTROL_MUTATION_GUARD_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
        edges=[],
        receipts=coverage_receipts,
        finding_assignments=[],
        registered_goal_ids=[CONTROL_MUTATION_GUARD_OBJECTIVE_ID],
        evaluated_at=now.isoformat(),
        repository_tree=operational.repository_tree,
    )
    health = AnalyzerHealthReport(
        status=AnalyzerHealthStatus.HEALTHY,
        reasons=(),
        thresholds=AnalyzerHealthThresholds(),
        metrics={
            "objective_id": CONTROL_MUTATION_GUARD_OBJECTIVE_ID,
            "repository_tree": operational.repository_tree,
            "analyzer_version": (
                CONTROL_MUTATION_GUARD_COMPLETION_ANALYZER_VERSION
            ),
        },
    )
    generic_binding = ExhaustionBinding(
        repository_id="repository:control",
        tree_id=operational.repository_tree,
        analyzer_version=(
            CONTROL_MUTATION_GUARD_COMPLETION_ANALYZER_VERSION
        ),
        configuration_revision=(
            CONTROL_MUTATION_GUARD_COMPLETION_CONFIGURATION_REVISION
        ),
        objective_revision=CONTROL_MUTATION_GUARD_OBJECTIVE_REVISION,
    )
    generic_quorum = ExhaustionQuorumResult(
        binding=generic_binding,
        required_members=2,
        members=(
            ExhaustionQuorumMember(
                member_id="asi-077-implementation",
                evidence_channel="implementation-validation",
                receipt_cid="scan:asi-077:implementation",
                binding=generic_binding,
                scan_mode="exhaustive",
                finished_at=now.isoformat(),
            ),
            ExhaustionQuorumMember(
                member_id="asi-077-replay",
                evidence_channel="receipt-replay-audit",
                receipt_cid="scan:asi-077:replay",
                binding=generic_binding,
                scan_mode="exhaustive",
                finished_at=now.isoformat(),
            ),
        ),
    )
    member_health = tuple(
        ControlMutationCompletionMemberHealth(
            member_id=member.member_id,
            receipt_cid=member.receipt_cid,
            healthy=True,
            safe_for_completion_reasoning=True,
        )
        for member in generic_quorum.members
    )
    quorum = ControlMutationCompletionQuorumEvidence(
        validation_policy_id=operational.policy_id,
        policy_revision=operational.policy_revision,
        operational_receipt_id=operational.content_id,
        quorum=generic_quorum,
        member_health=member_health,
    )
    assert (
        ControlMutationCompletionQuorumEvidence.from_json(
            quorum.to_json()
        ).content_id
        == quorum.content_id
    )
    values = {
        "evidence": completion_evidence,
        "tasks_complete": True,
        "coverage": coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

    assert operational.completion_authoritative is False
    no_independent_proof = operational.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        tasks_complete=True,
        now=now,
        freshness_seconds=300,
    )
    assert no_independent_proof.state is GoalState.PROVISIONALLY_COMPLETE
    assert not no_independent_proof.verified
    assert (
        no_independent_proof.gate is not None
        and not no_independent_proof.gate.passed
    )

    provisional = operational.evaluate_objective_completion(
        current_state=GoalState.ACTIVE,
        **values,
    )
    assert provisional.state is GoalState.PROVISIONALLY_COMPLETE
    assert provisional.gate is not None and provisional.gate.passed
    assert not provisional.verified

    verified = operational.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **values,
    )
    assert verified.state is GoalState.VERIFIED_COMPLETE
    assert verified.verified

    mapping_coverage = {
        "repository_tree": operational.repository_tree,
        "evaluated_at": now.isoformat(),
        "verified": True,
        "criteria": [
            {
                "criterion": criterion,
                "status": "verified",
                "verified": True,
                "implementation": (
                    "ipfs_accelerate_py/agent_supervisor/"
                    "control_contracts.py"
                ),
                "validation_receipt_ids": [
                    completion_evidence[index - 1].provenance_cid
                ],
            }
            for index, criterion in enumerate(
                CONTROL_MUTATION_GUARD_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
    }
    mapping_health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "objective_id": CONTROL_MUTATION_GUARD_OBJECTIVE_ID,
        "repository_tree": operational.repository_tree,
        "analyzer_version": (
            CONTROL_MUTATION_GUARD_COMPLETION_ANALYZER_VERSION
        ),
    }
    artifact_binding = {
        **generic_binding.to_dict(),
        "objective_id": CONTROL_MUTATION_GUARD_OBJECTIVE_ID,
        "requirement_id": CONTROL_MUTATION_GUARD_REQUIREMENT_ID,
        "validation_policy_id": operational.policy_id,
        "policy_revision": operational.policy_revision,
        "operational_receipt_id": operational.content_id,
    }
    mapping_quorum = {
        "required_members": 2,
        "member_count": 2,
        "satisfied": True,
        "quorum_met": True,
        "binding": artifact_binding,
        "members": [
            {
                "member_id": f"asi-077-mapping-{index}",
                "evidence_channel": channel,
                "receipt_cid": f"scan:asi-077:mapping:{index}",
                "binding": artifact_binding,
                "scan_mode": "exhaustive",
                "healthy": True,
                "safe_for_completion_reasoning": True,
                "finished_at": now.isoformat(),
            }
            for index, channel in enumerate(
                ("implementation-validation", "receipt-replay-audit"),
                start=1,
            )
        ],
    }
    mapping_values = {
        **values,
        "coverage": mapping_coverage,
        "analyzer_health": mapping_health,
        "exhaustion_quorum": mapping_quorum,
    }
    assert operational.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **mapping_values,
    ).verified

    detached_evidence = list(completion_evidence)
    detached_evidence[0] = CompletionEvidence.from_dict(
        {
            **detached_evidence[0].to_dict(),
            "validation_receipt": {
                **validation_binding,
                "operational_receipt_id": "sha256:detached",
            },
        }
    )
    failed_evidence = list(completion_evidence)
    failed_evidence[0] = CompletionEvidence.from_dict(
        {
            **failed_evidence[0].to_dict(),
            "validation_passed": False,
            "validation_receipt": {
                **validation_binding,
                "status": "failed",
            },
        }
    )
    stale_evidence = list(completion_evidence)
    stale_evidence[0] = CompletionEvidence.from_dict(
        {
            **stale_evidence[0].to_dict(),
            "observed_at": (now - timedelta(seconds=301)).isoformat(),
        }
    )
    incomplete_coverage = copy.deepcopy(mapping_coverage)
    incomplete_coverage["criteria"] = incomplete_coverage["criteria"][:-1]
    unbound_coverage = copy.deepcopy(mapping_coverage)
    unbound_coverage["criteria"][0]["validation_receipt_ids"] = [
        "validation:detached"
    ]
    unsafe_health = {
        **mapping_health,
        "safe_for_completion_reasoning": False,
    }
    foreign_health = {
        **mapping_health,
        "objective_id": "ASI-G999",
        "repository_tree": "tree:foreign",
    }
    wrong_analyzer_health = {
        **mapping_health,
        "analyzer_version": "asi-g104-objective-validation@stale",
    }
    duplicate_quorum = copy.deepcopy(mapping_quorum)
    duplicate_quorum["members"][1]["evidence_channel"] = (
        duplicate_quorum["members"][0]["evidence_channel"]
    )
    stale_quorum = copy.deepcopy(mapping_quorum)
    stale_quorum["members"][0]["finished_at"] = (
        now - timedelta(hours=1)
    ).isoformat()
    foreign_quorum = copy.deepcopy(mapping_quorum)
    foreign_quorum["binding"]["tree_id"] = "tree:foreign"
    for member in foreign_quorum["members"]:
        member["binding"]["tree_id"] = "tree:foreign"
    with pytest.raises(ControlContractError, match="cover every quorum"):
        ControlMutationCompletionQuorumEvidence(
            validation_policy_id=operational.policy_id,
            policy_revision=operational.policy_revision,
            operational_receipt_id=operational.content_id,
            quorum=generic_quorum,
            member_health=member_health[:-1],
        )
    with pytest.raises(ControlContractError, match="explicitly healthy"):
        ControlMutationCompletionQuorumEvidence(
            validation_policy_id=operational.policy_id,
            policy_revision=operational.policy_revision,
            operational_receipt_id=operational.content_id,
            quorum=generic_quorum,
            member_health=(
                ControlMutationCompletionMemberHealth(
                    member_id=member_health[0].member_id,
                    receipt_cid=member_health[0].receipt_cid,
                    healthy=False,
                    safe_for_completion_reasoning=True,
                ),
                member_health[1],
            ),
        )
    incomplete_tasks = operational.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**mapping_values, "tasks_complete": False},
    )
    assert incomplete_tasks.state is GoalState.REOPENED
    assert not incomplete_tasks.verified
    assert incomplete_tasks.tasks_complete is False
    rejected_inputs = (
        {"evidence": tuple(detached_evidence)},
        {"evidence": tuple(failed_evidence)},
        {"evidence": tuple(stale_evidence)},
        {"evidence": completion_evidence[:-1]},
        {"coverage": incomplete_coverage},
        {"coverage": unbound_coverage},
        {"analyzer_health": unsafe_health},
        {"analyzer_health": foreign_health},
        {"analyzer_health": wrong_analyzer_health},
        {"exhaustion_quorum": generic_quorum},
        {
            "exhaustion_quorum": ControlMutationCompletionQuorumEvidence(
                validation_policy_id="policy:foreign",
                policy_revision=operational.policy_revision,
                operational_receipt_id=operational.content_id,
                quorum=generic_quorum,
                member_health=member_health,
            )
        },
        {
            "exhaustion_quorum": ControlMutationCompletionQuorumEvidence(
                validation_policy_id=operational.policy_id,
                policy_revision=operational.policy_revision,
                operational_receipt_id="sha256:detached",
                quorum=generic_quorum,
                member_health=member_health,
            )
        },
        {"exhaustion_quorum": duplicate_quorum},
        {"exhaustion_quorum": stale_quorum},
        {"exhaustion_quorum": foreign_quorum},
    )
    for replacement in rejected_inputs:
        rejected = operational.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**mapping_values, **replacement},
        )
        assert rejected.state is GoalState.PROVISIONALLY_COMPLETE
        assert not rejected.verified
        assert rejected.gate is not None and not rejected.gate.passed


def test_lifecycle_rejects_non_lifecycle_operation(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(repo_root, state_root, [])
    request = OperationRequest(
        operation=Operation.STATUS,
        **_binding(repo_root, state_root),
    )

    with pytest.raises(ValueError, match="not a lifecycle"):
        service.lifecycle(request)
