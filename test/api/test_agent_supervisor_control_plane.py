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
    CONTROL_DISCOVERY_SAFETY_ACCEPTANCE_CRITERIA,
    CONTROL_DISCOVERY_SAFETY_COMPLETION_ANALYZER_VERSION,
    CONTROL_DISCOVERY_SAFETY_COMPLETION_CONFIGURATION_REVISION,
    CONTROL_DISCOVERY_SAFETY_OBJECTIVE_ID,
    CONTROL_DISCOVERY_SAFETY_OBJECTIVE_REVISION,
    CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID,
    CONTROL_SURFACE_PARITY_REQUIREMENT_ID,
    CONTROL_SURFACE_PARITY_ACCEPTANCE_CRITERIA,
    CONTROL_SURFACE_PARITY_COMPLETION_ANALYZER_VERSION,
    CONTROL_SURFACE_PARITY_COMPLETION_CONFIGURATION_REVISION,
    CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
    CONTROL_SURFACE_PARITY_OBJECTIVE_REVISION,
    AuthorizationDecision,
    AuthorizationVerdict,
    ControlBounds,
    ControlContractError,
    ControlDiscoveryCompletionMemberHealth,
    ControlDiscoveryCompletionQuorumEvidence,
    ControlDiscoveryManifest,
    ControlDiscoveryObservation,
    ControlDiscoveryRuntimeState,
    ControlDiscoverySafetyEvidence,
    ControlSurface,
    ControlSurfaceParityCase,
    ControlSurfaceParityCompletionMemberHealth,
    ControlSurfaceParityCompletionQuorumEvidence,
    ControlSurfaceParityEvidence,
    EffectKind,
    ErrorCode,
    ExpectedEffect,
    IdempotencyKey,
    MUTATION_OPERATIONS,
    PROPOSAL_OPERATIONS,
    READ_OPERATIONS,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationResult,
    OperationStatus,
    operation_request_json_schema,
    operation_result_json_schema,
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
    JsonlControlStateStore,
    RepositorySupervisorBackend,
    StaleLeaseError,
    SupervisorClient,
    SupervisorControlService,
    SupervisorTarget,
    capture_control_discovery_runtime_state,
)


def _binding(
    repo_root: Path,
    state_root: Path,
    *,
    objective_id: str = "ASI-G070",
) -> dict[str, Any]:
    return {
        "repository_root": str(repo_root),
        "state_root": str(state_root),
        "repository_id": "repo:fixture",
        "tree_id": "tree:abc",
        "objective_id": objective_id,
        "objective_revision": "objective:1",
        "policy_id": "policy:supervisor",
        "policy_revision": "policy:1",
        "caller": "operator:alice",
    }


def _effect(operation: Operation) -> ExpectedEffect:
    return ExpectedEffect(
        effect_id=f"{operation.value}:target",
        kind=(
            EffectKind.EXECUTE_VALIDATION
            if operation is Operation.VALIDATION_REPLAY
            else EffectKind.WRITE_STATE
        ),
        resource="supervisor:fixture",
        paths=("data/agent_supervisor",),
        description=f"Apply {operation.value}",
    )


def _mutation_request(
    repo_root: Path,
    state_root: Path,
    operation: Operation = Operation.PAUSE,
    *,
    key: str = "request:one",
    parameters: dict[str, Any] | None = None,
    dry_run: bool = False,
    objective_id: str = "ASI-G070",
) -> OperationRequest:
    binding = _binding(
        repo_root,
        state_root,
        objective_id=objective_id,
    )
    effect = _effect(operation)
    values: dict[str, Any] = {
        "operation": operation,
        **binding,
        "expected_effects": (effect,),
        "parameters": parameters or {"target_id": "supervisor:fixture"},
        "dry_run": dry_run,
    }
    if not dry_run:
        values.update(
            {
                "idempotency": IdempotencyKey(
                    key=key,
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


def _read_request(
    repo_root: Path,
    state_root: Path,
    operation: Operation,
    parameters: dict[str, Any] | None = None,
    *,
    bounds: ControlBounds | None = None,
    objective_id: str = "ASI-G070",
) -> OperationRequest:
    return OperationRequest(
        operation=operation,
        **_binding(
            repo_root,
            state_root,
            objective_id=objective_id,
        ),
        parameters=parameters or {},
        bounds=bounds or ControlBounds(),
    )


def _service(
    repo_root: Path,
    state_root: Path,
    *,
    handlers: dict[Operation, Any] | None = None,
    lease_validator: Any = lambda _request: True,
    state_store: InMemoryControlStateStore | None = None,
) -> SupervisorControlService:
    return SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers=handlers,
        lease_validator=lease_validator,
        state_store=state_store or InMemoryControlStateStore(),
        clock_ms=lambda: 1_500,
    )


def _parity_service(
    repo_root: Path,
    state_root: Path,
) -> SupervisorControlService:
    def operation_handler(request: OperationRequest) -> BackendResponse:
        return BackendResponse(
            data={"operation": request.operation.value},
            changed=bool(request.expected_effects),
            applied_effect_ids=tuple(
                effect.effect_id for effect in request.expected_effects
            ),
        )

    return _service(
        repo_root,
        state_root,
        handlers={
            operation: operation_handler
            for operation in Operation
            if operation not in READ_OPERATIONS
        }
        | {Operation.STATUS: operation_handler},
    )


def _parity_cases(
    service: SupervisorControlService,
    repo_root: Path,
    state_root: Path,
) -> tuple[ControlSurfaceParityCase, ...]:
    requests = (
        (
            "read_success",
            _read_request(
                repo_root,
                state_root,
                Operation.STATUS,
                objective_id=CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
            ),
        ),
        (
            "proposal_success",
            _mutation_request(
                repo_root,
                state_root,
                Operation.PAUSE,
                dry_run=True,
                objective_id=CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
            ),
        ),
        (
            "stable_failure",
            _read_request(
                repo_root,
                state_root,
                Operation.HEALTH,
                {"health_path": "missing-health.json"},
                objective_id=CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
            ),
        ),
        (
            "mutation_success",
            _mutation_request(
                repo_root,
                state_root,
                Operation.PAUSE,
                key="parity:mutation",
                objective_id=CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
            ),
        ),
    )
    cases = []
    for scenario, request in requests:
        record = service.execute(request).to_record()
        cases.append(
            ControlSurfaceParityCase(
                scenario=scenario,
                request=request,
                python_result=record,
                cli_result=record,
                mcp_result=record,
            )
        )
    return tuple(cases)


def test_capabilities_are_complete_typed_and_side_effect_free(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    handlers = {
        operation: (lambda _request: {})
        for operation in Operation
        if operation not in {
            Operation.CAPABILITIES,
            *tuple(READ_OPERATIONS),
        }
    }
    service = _service(repo_root, state_root, handlers=handlers)

    report = service.capabilities()

    assert report.supported_operations == tuple(
        sorted(Operation, key=lambda item: item.value)
    )
    assert report.processes_started is False
    assert report.optional_providers_loaded is False
    for operation in Operation:
        capability = report.capability_for(operation)
        assert capability is not None
        assert capability.authority is operation.authority
        assert capability.requires_idempotency is operation.mutating
        assert capability.requires_authorization is operation.mutating


def test_python_discovery_is_cached_deterministic_and_never_dispatches(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    dispatches = 0

    def forbidden(_request: OperationRequest) -> dict[str, Any]:
        nonlocal dispatches
        dispatches += 1
        raise AssertionError("discovery dispatched a backend")

    handlers = {
        operation: forbidden
        for operation in Operation
        if operation not in {Operation.CAPABILITIES, *tuple(READ_OPERATIONS)}
    }
    service = _service(repo_root, state_root, handlers=handlers)
    before = capture_control_discovery_runtime_state()

    first = service.discovery_manifest()
    report_one = service.capability_report()
    second = service.discovery_manifest()
    report_two = service.capability_report()
    after = capture_control_discovery_runtime_state()

    assert first == second
    assert first.canonical_bytes() == second.canonical_bytes()
    assert report_one is report_two
    assert first.surface is ControlSurface.PYTHON
    assert before == after
    assert dispatches == 0


def test_discovery_safety_evidence_is_complete_content_addressed_and_strict(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    handlers = {
        operation: (lambda _request: {})
        for operation in Operation
        if operation not in {
            Operation.CAPABILITIES,
            *tuple(READ_OPERATIONS),
        }
    }
    service = _service(repo_root, state_root, handlers=handlers)
    state = ControlDiscoveryRuntimeState()
    observations = tuple(
        ControlDiscoveryObservation(
            surface=surface,
            first_manifest=ControlDiscoveryManifest(surface=surface),
            second_manifest=ControlDiscoveryManifest(surface=surface),
            before=state,
            after=state,
        )
        for surface in ControlSurface
    )
    evidence = ControlDiscoverySafetyEvidence(
        repository_tree="tree:abc",
        objective_id="ASI-G105",
        policy_id="policy:control",
        policy_revision="policy:1",
        capability_report=service.capability_report(),
        observations=observations,
    )

    assert evidence.proved_requirement_ids == (
        CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID,
    )
    assert ControlDiscoverySafetyEvidence.from_dict(
        evidence.to_record()
    ) == evidence

    changed = state.to_record()
    changed["process_start_count"] = 1
    changed.pop("content_id")
    with pytest.raises(ControlContractError, match="process_start_count"):
        ControlDiscoveryObservation(
            surface=ControlSurface.MCP,
            first_manifest=ControlDiscoveryManifest(
                surface=ControlSurface.MCP
            ),
            second_manifest=ControlDiscoveryManifest(
                surface=ControlSurface.MCP
            ),
            before=state,
            after=changed,
        )

    with pytest.raises(ControlContractError, match="Python, CLI, and MCP"):
        ControlDiscoverySafetyEvidence(
            repository_tree="tree:abc",
            objective_id="ASI-G105",
            policy_id="policy:control",
            policy_revision="policy:1",
            capability_report=service.capability_report(),
            observations=observations[:-1],
        )
    with pytest.raises(ControlContractError, match="objective_id"):
        ControlDiscoverySafetyEvidence(
            repository_tree="tree:abc",
            objective_id="ASI-G999",
            policy_id="policy:control",
            policy_revision="policy:1",
            capability_report=service.capability_report(),
            observations=observations,
        )


def test_g105_completion_requires_bound_current_tree_validation_health_and_quorum(
    tmp_path: Path,
) -> None:
    """ASI-076: operational discovery proof cannot self-certify completion."""

    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(
        repo_root,
        state_root,
        handlers={
            operation: (lambda _request: {})
            for operation in Operation
            if operation not in {
                Operation.CAPABILITIES,
                *tuple(READ_OPERATIONS),
            }
        },
    )
    runtime_state = ControlDiscoveryRuntimeState()
    operational = ControlDiscoverySafetyEvidence(
        repository_tree="tree:asi-076",
        objective_id=CONTROL_DISCOVERY_SAFETY_OBJECTIVE_ID,
        policy_id="policy:control",
        policy_revision="policy:asi-076",
        capability_report=service.capability_report(),
        observations=tuple(
            ControlDiscoveryObservation(
                surface=surface,
                first_manifest=ControlDiscoveryManifest(surface=surface),
                second_manifest=ControlDiscoveryManifest(surface=surface),
                before=runtime_state,
                after=runtime_state,
            )
            for surface in ControlSurface
        ),
    )
    now = datetime(2026, 7, 24, 17, 0, tzinfo=timezone.utc)
    command = (
        "python -m pytest test/api/test_agent_supervisor_control_plane.py "
        "test/api/test_agent_supervisor_control_lifecycle.py "
        "test/test_unified_cli_agent_supervisor.py "
        "test/mcp_server/test_agent_supervisor_tools.py -q"
    )
    validation_binding = {
        "status": "passed",
        "tree_id": operational.repository_tree,
        "requirement_id": CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID,
        "objective_id": CONTROL_DISCOVERY_SAFETY_OBJECTIVE_ID,
        "operational_receipt_id": operational.content_id,
        "validation_policy_id": operational.policy_id,
        "policy_revision": operational.policy_revision,
        "command": command,
    }
    completion_evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-076",
            producer_kind="task",
            validation_receipt=validation_binding,
            validation_passed=True,
            repository_tree=operational.repository_tree,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-076:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            CONTROL_DISCOVERY_SAFETY_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    assert operational.completion_authoritative is False
    coverage_receipts = [
        ValidationReceiptCoverage(
            receipt_id=item.provenance_cid,
            task_id="ASI-076",
            criterion=item.acceptance_criterion,
            command=command,
            status=CoverageStatus.VERIFIED,
            passed=True,
            repository_tree=operational.repository_tree,
            observed_at=now.isoformat(),
            provenance_cid=item.provenance_cid,
            explanation="fresh passing ASI-076 criterion validation",
            outcome="passed",
            reason_code="validation_verified",
            fresh=True,
        )
        for item in completion_evidence
    ]
    canonical_coverage = GoalCoverageMap(
        criteria=[
            AcceptanceCoverage(
                criterion_id=f"criterion:g105:{index}",
                goal_id=CONTROL_DISCOVERY_SAFETY_OBJECTIVE_ID,
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
                CONTROL_DISCOVERY_SAFETY_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
        edges=[],
        receipts=coverage_receipts,
        finding_assignments=[],
        registered_goal_ids=[CONTROL_DISCOVERY_SAFETY_OBJECTIVE_ID],
        evaluated_at=now.isoformat(),
        repository_tree=operational.repository_tree,
    )
    health = AnalyzerHealthReport(
        status=AnalyzerHealthStatus.HEALTHY,
        reasons=(),
        thresholds=AnalyzerHealthThresholds(),
        metrics={
            "objective_id": CONTROL_DISCOVERY_SAFETY_OBJECTIVE_ID,
            "repository_tree": operational.repository_tree,
            "analyzer_version": (
                CONTROL_DISCOVERY_SAFETY_COMPLETION_ANALYZER_VERSION
            ),
        },
    )
    binding = ExhaustionBinding(
        repository_id="repository:control",
        tree_id=operational.repository_tree,
        analyzer_version=(
            CONTROL_DISCOVERY_SAFETY_COMPLETION_ANALYZER_VERSION
        ),
        configuration_revision=(
            CONTROL_DISCOVERY_SAFETY_COMPLETION_CONFIGURATION_REVISION
        ),
        objective_revision=CONTROL_DISCOVERY_SAFETY_OBJECTIVE_REVISION,
    )
    generic_quorum = ExhaustionQuorumResult(
        binding=binding,
        required_members=2,
        members=(
            ExhaustionQuorumMember(
                member_id="asi-076-implementation",
                evidence_channel="implementation-validation",
                receipt_cid="scan:asi-076:implementation",
                binding=binding,
                scan_mode="exhaustive",
                finished_at=now.isoformat(),
            ),
            ExhaustionQuorumMember(
                member_id="asi-076-replay",
                evidence_channel="receipt-replay-audit",
                receipt_cid="scan:asi-076:replay",
                binding=binding,
                scan_mode="exhaustive",
                finished_at=now.isoformat(),
            ),
        ),
    )
    member_health = tuple(
        ControlDiscoveryCompletionMemberHealth(
            member_id=member.member_id,
            receipt_cid=member.receipt_cid,
            healthy=True,
            safe_for_completion_reasoning=True,
        )
        for member in generic_quorum.members
    )
    quorum = ControlDiscoveryCompletionQuorumEvidence(
        validation_policy_id=operational.policy_id,
        policy_revision=operational.policy_revision,
        operational_receipt_id=operational.content_id,
        quorum=generic_quorum,
        member_health=member_health,
    )
    assert (
        ControlDiscoveryCompletionQuorumEvidence.from_json(
            quorum.to_json()
        ).content_id
        == quorum.content_id
    )
    values = {
        "evidence": completion_evidence,
        "tasks_complete": True,
        "coverage": canonical_coverage,
        "analyzer_health": health,
        "exhaustion_quorum": quorum,
        "now": now,
        "freshness_seconds": 300,
    }

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

    # Mapping projections remain auditable but must carry the artifact-specific
    # objective, policy, and receipt bindings omitted from generic quorum types.
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
                CONTROL_DISCOVERY_SAFETY_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
    }
    mapping_health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "objective_id": CONTROL_DISCOVERY_SAFETY_OBJECTIVE_ID,
        "repository_tree": operational.repository_tree,
        "analyzer_version": (
            CONTROL_DISCOVERY_SAFETY_COMPLETION_ANALYZER_VERSION
        ),
    }
    artifact_binding = {
        **binding.to_dict(),
        "objective_id": CONTROL_DISCOVERY_SAFETY_OBJECTIVE_ID,
        "requirement_id": CONTROL_DISCOVERY_SAFETY_REQUIREMENT_ID,
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
                "member_id": f"asi-076-mapping-{index}",
                "evidence_channel": channel,
                "receipt_cid": f"scan:asi-076:mapping:{index}",
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
    mapped = operational.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{
            **values,
            "coverage": mapping_coverage,
            "analyzer_health": mapping_health,
            "exhaustion_quorum": mapping_quorum,
        },
    )
    assert mapped.verified

    detached_evidence_populations = []
    for field_name, detached_value in (
        ("operational_receipt_id", "sha256:detached"),
        ("objective_id", "ASI-G999"),
        ("tree_id", "tree:foreign"),
        ("validation_policy_id", "policy:foreign"),
        ("policy_revision", "policy:foreign-revision"),
    ):
        detached_evidence = list(completion_evidence)
        detached_evidence[0] = CompletionEvidence.from_dict(
            {
                **detached_evidence[0].to_dict(),
                "validation_receipt": {
                    **validation_binding,
                    field_name: detached_value,
                },
            }
        )
        detached_evidence_populations.append(tuple(detached_evidence))
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
    unsafe_health = {**mapping_health, "safe_for_completion_reasoning": False}
    mismatched_health = {
        **mapping_health,
        "analyzer_version": "foreign-analyzer@1",
    }
    detached_health = {
        **mapping_health,
        "objective_id": "ASI-G999",
        "repository_tree": "tree:foreign",
    }
    unbound_health = {
        key: value
        for key, value in mapping_health.items()
        if key not in {"objective_id", "repository_tree"}
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
        ControlDiscoveryCompletionQuorumEvidence(
            validation_policy_id=operational.policy_id,
            policy_revision=operational.policy_revision,
            operational_receipt_id=operational.content_id,
            quorum=generic_quorum,
            member_health=member_health[:-1],
        )
    with pytest.raises(ControlContractError, match="explicitly healthy"):
        ControlDiscoveryCompletionQuorumEvidence(
            validation_policy_id=operational.policy_id,
            policy_revision=operational.policy_revision,
            operational_receipt_id=operational.content_id,
            quorum=generic_quorum,
            member_health=(
                ControlDiscoveryCompletionMemberHealth(
                    member_id=member_health[0].member_id,
                    receipt_cid=member_health[0].receipt_cid,
                    healthy=False,
                    safe_for_completion_reasoning=True,
                ),
                member_health[1],
            ),
        )
    rejected_inputs = (
        *(
            {"evidence": population}
            for population in detached_evidence_populations
        ),
        {"evidence": tuple(failed_evidence)},
        {"evidence": tuple(stale_evidence)},
        {"evidence": completion_evidence[:-1]},
        {"coverage": incomplete_coverage},
        {"coverage": unbound_coverage},
        {"analyzer_health": unsafe_health},
        {"analyzer_health": mismatched_health},
        {"analyzer_health": detached_health},
        {"analyzer_health": unbound_health},
        {"exhaustion_quorum": generic_quorum},
        {
            "exhaustion_quorum": ControlDiscoveryCompletionQuorumEvidence(
                validation_policy_id="policy:foreign",
                policy_revision=operational.policy_revision,
                operational_receipt_id=operational.content_id,
                quorum=generic_quorum,
                member_health=member_health,
            )
        },
        {
            "exhaustion_quorum": ControlDiscoveryCompletionQuorumEvidence(
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
    mapping_values = {
        **values,
        "coverage": mapping_coverage,
        "analyzer_health": mapping_health,
        "exhaustion_quorum": mapping_quorum,
    }
    for replacement in rejected_inputs:
        rejected = operational.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**mapping_values, **replacement},
        )
        assert rejected.state is GoalState.PROVISIONALLY_COMPLETE
        assert not rejected.verified
        assert rejected.gate is not None and not rejected.gate.passed


def test_read_client_uses_direct_repository_apis_and_bounded_results(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    (repo_root / "objectives.md").write_text(
        "\n".join(
            (
                "# Objectives",
                "## G-1 First objective",
                "- Status: active",
                "- Acceptance: receipt",
                "## G-2 Second objective",
                "- Status: complete",
                "- Acceptance: proof",
            )
        ),
        encoding="utf-8",
    )
    (repo_root / "tasks.todo.md").write_text(
        "\n".join(
            (
                "## ASI-1 First task",
                "- Status: todo",
                "## ASI-2 Second task",
                "- Status: complete",
            )
        ),
        encoding="utf-8",
    )
    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        state_store=InMemoryControlStateStore(),
        max_query_items=2,
        clock_ms=lambda: 1_500,
    )
    client = SupervisorClient(
        service,
        target=SupervisorTarget(**_binding(repo_root, state_root)),
    )

    goals = client.goals(objective_path="objectives.md", limit=1)
    tasks = client.tasks(
        todo_path="tasks.todo.md", task_header_prefix="ASI-", limit=2
    )

    assert goals.succeeded
    assert goals.data["count"] == 1
    assert goals.data["truncated"] is True
    assert goals.data["items"][0]["goal_id"] == "G-1"
    assert tasks.data["count"] == 2
    assert [item["task_id"] for item in tasks.data["items"]] == [
        "ASI-1",
        "ASI-2",
    ]
    assert goals.audit_receipt_id.startswith("sha256:")


def test_mutation_is_authorized_fenced_audited_and_idempotent(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls: list[str] = []
    leases: list[tuple[str, int | None]] = []
    store = InMemoryControlStateStore()

    def pause(request: OperationRequest) -> BackendResponse:
        calls.append(request.request_id)
        return BackendResponse(
            data={"state": "paused"},
            changed=True,
            applied_effect_ids=("pause:target",),
        )

    def validate_lease(request: OperationRequest) -> bool:
        leases.append((request.lease_id, request.fencing_epoch))
        return request.lease_id == "lease:7" and request.fencing_epoch == 7

    service = _service(
        repo_root,
        state_root,
        handlers={Operation.PAUSE: pause},
        lease_validator=validate_lease,
        state_store=store,
    )
    request = _mutation_request(repo_root, state_root)

    before = service.mutation_runtime_state()
    first = service.pause(request)
    after_first = service.mutation_runtime_state()
    replay = service.execute(request)
    after_replay = service.mutation_runtime_state()

    assert first is replay
    assert first.status is OperationStatus.SUCCEEDED
    assert first.data["state"] == "paused"
    assert first.effects[0].applied is True
    assert first.effects[0].receipt_id == first.audit_receipt_id
    assert first.idempotency_key == "request:one"
    assert calls == [request.request_id]
    assert leases == [("lease:7", 7)]
    assert before.dispatch_count == 0
    assert before.audit_receipt_count == 0
    assert after_first.dispatch_count == 1
    assert after_first.last_dispatch_request_id == request.request_id
    assert after_first.audit_receipt_count == 1
    assert after_first.last_audit_receipt_id == first.audit_receipt_id
    assert after_replay == after_first


def test_same_idempotency_key_with_different_payload_conflicts(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _service(
        repo_root,
        state_root,
        handlers={Operation.PAUSE: lambda _request: {"state": "paused"}},
    )
    first = _mutation_request(repo_root, state_root)
    changed = _mutation_request(
        repo_root,
        state_root,
        parameters={"target_id": "supervisor:other"},
    )

    assert service.execute(first).succeeded
    conflict = service.execute(changed)

    assert conflict.status is OperationStatus.CONFLICT
    assert conflict.error is not None
    assert conflict.error.code is ErrorCode.IDEMPOTENCY_CONFLICT


def test_default_store_replays_exact_mutation_result_after_restart(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls = 0

    def pause(_request: OperationRequest) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"state": "paused"}

    request = _mutation_request(repo_root, state_root)
    first_service = _service(
        repo_root,
        state_root,
        handlers={Operation.PAUSE: pause},
        state_store=JsonlControlStateStore(),
    )
    first = first_service.execute(request)
    restarted = _service(
        repo_root,
        state_root,
        handlers={Operation.PAUSE: pause},
        state_store=JsonlControlStateStore(),
    )

    replay = restarted.execute(request)

    assert replay == first
    assert replay.result_id == first.result_id
    assert calls == 1


def test_dry_run_never_calls_mutation_or_requires_a_live_lease(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls = 0

    def forbidden_call(_request: OperationRequest) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"unexpected": True}

    service = _service(
        repo_root,
        state_root,
        handlers={Operation.QUARANTINE: forbidden_call},
        lease_validator=lambda _request: (_ for _ in ()).throw(
            AssertionError("dry run checked a live lease")
        ),
    )
    request = _mutation_request(
        repo_root,
        state_root,
        operation=Operation.QUARANTINE,
        dry_run=True,
    )

    result = service.quarantine(request)

    assert result.succeeded
    assert result.authority is OperationAuthority.PROPOSAL
    assert result.preview is not None
    assert result.preview.would_change is True
    assert result.effects == ()
    assert result.preview.expected_effects == request.expected_effects
    assert calls == 0


def test_allowlists_bounds_and_paths_fail_with_stable_errors(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    other_repo = tmp_path / "other"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    other_repo.mkdir()
    state_root.mkdir()
    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        state_store=InMemoryControlStateStore(),
        max_query_items=2,
        clock_ms=lambda: 1_500,
    )

    denied = service.execute(
        _read_request(
            other_repo,
            state_root,
            Operation.STATUS,
            {"status_path": "status.json"},
        )
    )
    bounded = service.execute(
        _read_request(
            repo_root,
            state_root,
            Operation.EVENTS,
            {"events_path": "events.jsonl", "limit": 3},
        )
    )
    escaped = service.execute(
        _read_request(
            repo_root,
            state_root,
            Operation.STATUS,
            {"status_path": "../outside.json"},
        )
    )

    assert denied.status is OperationStatus.DENIED
    assert denied.error and denied.error.code is ErrorCode.FORBIDDEN
    assert bounded.error and bounded.error.code is ErrorCode.BOUNDS_EXCEEDED
    assert escaped.error and escaped.error.code is ErrorCode.PATH_ESCAPE


def test_stale_fence_and_expired_authorization_fail_before_backend(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    calls = 0

    def backend(_request: OperationRequest) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return {"state": "paused"}

    service = _service(
        repo_root,
        state_root,
        handlers={Operation.PAUSE: backend},
        lease_validator=lambda _request: (_ for _ in ()).throw(
            StaleLeaseError("fencing epoch has been superseded")
        ),
    )
    stale = service.execute(_mutation_request(repo_root, state_root))

    assert stale.status is OperationStatus.CONFLICT
    assert stale.error and stale.error.code is ErrorCode.STALE_LEASE
    assert calls == 0
    assert service.mutation_runtime_state().dispatch_count == 0

    expired_service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        handlers={Operation.PAUSE: backend},
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 2_001,
    )
    expired = expired_service.execute(_mutation_request(repo_root, state_root))
    assert expired.status is OperationStatus.DENIED
    assert expired.error and expired.error.code is ErrorCode.UNAUTHORIZED
    assert calls == 0
    assert expired_service.mutation_runtime_state().dispatch_count == 0

    denied_service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        authorization_validator=lambda _request: False,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_500,
    )
    denied = denied_service.execute(
        _read_request(
            repo_root,
            state_root,
            Operation.STATUS,
            {"status_path": "status.json"},
        )
    )
    assert denied.status is OperationStatus.DENIED
    assert denied.error and denied.error.code is ErrorCode.UNAUTHORIZED


@pytest.mark.parametrize(
    ("exception", "code", "status"),
    (
        (FileNotFoundError("missing"), ErrorCode.NOT_FOUND, OperationStatus.NOT_FOUND),
        (TimeoutError("slow"), ErrorCode.TIMED_OUT, OperationStatus.TIMED_OUT),
        (ValueError("bad selector"), ErrorCode.INVALID_REQUEST, OperationStatus.FAILED),
        (RuntimeError("secret backend detail"), ErrorCode.INTERNAL_ERROR, OperationStatus.FAILED),
    ),
)
def test_backend_failures_are_translated_to_stable_typed_errors(
    tmp_path: Path,
    exception: Exception,
    code: ErrorCode,
    status: OperationStatus,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()

    def fail(_request: OperationRequest) -> dict[str, Any]:
        raise exception

    service = _service(
        repo_root,
        state_root,
        handlers={Operation.PLAN: fail},
    )
    result = service.plan(
        _read_request(repo_root, state_root, Operation.PLAN, {"limit": 1})
    )

    assert result.status is status
    assert result.error is not None
    assert result.error.code is code
    if code is ErrorCode.INTERNAL_ERROR:
        assert result.error.message == "control operation failed"


def test_receipt_query_is_bounded_and_does_not_require_raw_state_access(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    store = InMemoryControlStateStore()
    service = _service(repo_root, state_root, state_store=store)

    status_request = _read_request(
        repo_root,
        state_root,
        Operation.STATUS,
        {"status_path": "status.json"},
    )
    # Missing reads are audited too.
    assert service.status(status_request).status is OperationStatus.NOT_FOUND
    receipts = service.receipts(
        _read_request(
            repo_root,
            state_root,
            Operation.RECEIPTS,
            {"limit": 1},
        )
    )

    assert receipts.succeeded
    assert receipts.data["count"] == 1
    assert receipts.data["items"][0]["operation"] == "status"
    assert receipts.data["items"][0]["error_code"] == "not_found"


def test_service_calls_registered_python_handler_without_shell_translation(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    seen: list[OperationRequest] = []

    def replay(request: OperationRequest) -> BackendResponse:
        seen.append(request)
        return BackendResponse(
            data={"validation_receipt": "receipt:new"},
            changed=True,
            applied_effect_ids=("validation_replay:target",),
        )

    backend = RepositorySupervisorBackend(
        {Operation.VALIDATION_REPLAY: replay}
    )
    service = SupervisorControlService(
        repository_allowlist=(repo_root,),
        state_allowlist=(state_root,),
        backend=backend,
        lease_validator=lambda _request: True,
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 1_500,
    )
    request = _mutation_request(
        repo_root,
        state_root,
        operation=Operation.VALIDATION_REPLAY,
    )

    result = service.validation_replay(request)

    assert result.succeeded
    assert seen == [request]
    assert result.data["validation_receipt"] == "receipt:new"


def test_shared_wire_schemas_cover_every_operation_and_mutation_guard() -> None:
    request_schema = operation_request_json_schema()
    result_schema = operation_result_json_schema()

    assert set(request_schema["properties"]["operation"]["enum"]) == {
        item.value for item in Operation
    }
    assert set(result_schema["properties"]["operation"]["enum"]) == {
        item.value for item in Operation
    }
    pause_schema = operation_request_json_schema(Operation.PAUSE)
    assert pause_schema["properties"]["operation"]["const"] == "pause"
    assert {
        "expected_effects",
        "idempotency",
        "authorization",
        "lease_id",
        "fencing_epoch",
    }.issubset(pause_schema["allOf"][0]["then"]["required"])


def test_python_surface_executes_every_closed_operation_with_canonical_results(
    tmp_path: Path,
) -> None:
    """ASI-067: every advertised operation has one typed Python entry point."""

    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    dispatched: list[Operation] = []

    def operation_handler(request: OperationRequest) -> BackendResponse:
        dispatched.append(request.operation)
        return BackendResponse(
            data={"operation": request.operation.value},
            changed=bool(request.expected_effects),
            applied_effect_ids=(
                tuple(effect.effect_id for effect in request.expected_effects)
                if request.operation in MUTATION_OPERATIONS
                else ()
            ),
            checks=("closed_operation", "canonical_result"),
        )

    service = _service(
        repo_root,
        state_root,
        handlers={operation: operation_handler for operation in Operation},
    )
    results: dict[Operation, OperationResult] = {}
    for operation in Operation:
        if operation in MUTATION_OPERATIONS:
            request = _mutation_request(
                repo_root,
                state_root,
                operation,
                key=f"asi-067:{operation.value}",
            )
        elif operation in PROPOSAL_OPERATIONS:
            request = OperationRequest(
                operation=operation,
                **_binding(repo_root, state_root),
                parameters={"target_id": "objective:fixture"},
                expected_effects=(
                    ExpectedEffect(
                        effect_id=f"{operation.value}:proposal",
                        kind=EffectKind.PROPOSE,
                        resource="objective:fixture",
                        paths=("docs/architecture",),
                        description=f"Preview {operation.value}",
                    ),
                ),
            )
        else:
            request = _read_request(repo_root, state_root, operation)

        decoded_request = OperationRequest.from_json(request.to_json())
        assert decoded_request == request
        entry_point = getattr(service, operation.value)
        result = entry_point(decoded_request)
        result.validate_against(request)
        decoded_result = OperationResult.from_json(result.to_json())
        assert decoded_result == result
        assert decoded_result.operation is operation
        assert decoded_result.authority is request.effective_authority
        assert decoded_result.audit_receipt_id
        if operation in PROPOSAL_OPERATIONS:
            assert decoded_result.preview is not None
            assert decoded_result.preview.expected_effects == request.expected_effects
            assert not any(effect.applied for effect in decoded_result.effects)
        elif operation in MUTATION_OPERATIONS:
            assert decoded_result.preview is None
            assert {effect.effect_id for effect in decoded_result.effects} == {
                effect.effect_id for effect in request.expected_effects
            }
            assert all(effect.applied for effect in decoded_result.effects)
        else:
            assert decoded_result.preview is None
            assert decoded_result.effects == ()
        results[operation] = decoded_result

    assert set(results) == set(Operation)
    # Capabilities and receipt queries are intentionally implemented by the
    # service boundary; every other operation reaches the registered Python
    # adapter exactly once.
    assert dispatched == [
        operation
        for operation in Operation
        if operation not in {Operation.CAPABILITIES, Operation.RECEIPTS}
    ]


def test_typed_surface_parity_evidence_proves_exact_requirement(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _parity_service(repo_root, state_root)
    cases = _parity_cases(service, repo_root, state_root)
    request = cases[0].request
    assert isinstance(request, OperationRequest)
    with pytest.raises(ControlContractError, match="complete behavior matrix"):
        ControlSurfaceParityEvidence(
            repository_tree=request.tree_id,
            objective_id=request.objective_id,
            policy_id=request.policy_id,
            policy_revision=request.policy_revision,
            capability_report=service.capability_report(),
            cases=(cases[0],),
        )
    evidence = ControlSurfaceParityEvidence(
        repository_tree=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        policy_revision=request.policy_revision,
        capability_report=service.capability_report().to_record(),
        cases=cases,
    )

    assert evidence.proved_requirement_ids == (
        CONTROL_SURFACE_PARITY_REQUIREMENT_ID,
    )
    assert ControlSurfaceParityEvidence.from_dict(evidence.to_record()) == evidence
    assert evidence.request_schema_id
    assert evidence.result_schema_id


def test_g103_completion_requires_bound_current_tree_validation_health_and_quorum(
    tmp_path: Path,
) -> None:
    """ASI-067: a parity witness cannot self-certify objective completion."""

    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _parity_service(repo_root, state_root)
    cases = _parity_cases(service, repo_root, state_root)
    operational = ControlSurfaceParityEvidence(
        repository_tree="tree:abc",
        objective_id=CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
        policy_id="policy:supervisor",
        policy_revision="policy:1",
        capability_report=service.capability_report(),
        cases=cases,
    )
    now = datetime(2026, 7, 24, 19, 0, tzinfo=timezone.utc)
    command = (
        "python -m pytest test/api/test_agent_supervisor_control_plane.py "
        "test/api/test_agent_supervisor_control_lifecycle.py "
        "test/test_unified_cli_agent_supervisor.py "
        "test/mcp_server/test_agent_supervisor_tools.py -q"
    )
    validation_binding = {
        "status": "passed",
        "tree_id": operational.repository_tree,
        "requirement_id": CONTROL_SURFACE_PARITY_REQUIREMENT_ID,
        "objective_id": CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
        "operational_receipt_id": operational.content_id,
        "validation_policy_id": operational.policy_id,
        "policy_revision": operational.policy_revision,
        "command": command,
    }
    completion_evidence = tuple(
        CompletionEvidence(
            acceptance_criterion=criterion,
            producing_task_or_scan="ASI-067",
            producer_kind="task",
            validation_receipt=validation_binding,
            validation_passed=True,
            repository_tree=operational.repository_tree,
            freshness={"fresh": True},
            observed_at=now,
            provenance_cid=f"validation:asi-067:{index}",
            metadata={
                "evidence_source_policy": {
                    "satisfies": True,
                    "source_tier": "validation_receipt",
                }
            },
        )
        for index, criterion in enumerate(
            CONTROL_SURFACE_PARITY_ACCEPTANCE_CRITERIA,
            start=1,
        )
    )
    coverage = GoalCoverageMap(
        criteria=[
            AcceptanceCoverage(
                criterion_id=f"criterion:g103:{index}",
                goal_id=CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
                criterion=criterion,
                status=CoverageStatus.VERIFIED,
                changed_files=[
                    "ipfs_accelerate_py/agent_supervisor/"
                    "control_contracts.py",
                    "ipfs_accelerate_py/agent_supervisor/control_plane.py",
                ],
                validation_receipt_ids=[
                    completion_evidence[index - 1].provenance_cid
                ],
                explanation="unified control implementation is exactly validated",
            )
            for index, criterion in enumerate(
                CONTROL_SURFACE_PARITY_ACCEPTANCE_CRITERIA,
                start=1,
            )
        ],
        edges=[],
        receipts=[
            ValidationReceiptCoverage(
                receipt_id=item.provenance_cid,
                task_id="ASI-067",
                criterion=item.acceptance_criterion,
                command=command,
                status=CoverageStatus.VERIFIED,
                passed=True,
                repository_tree=operational.repository_tree,
                observed_at=now.isoformat(),
                provenance_cid=item.provenance_cid,
                explanation="fresh passing ASI-067 criterion validation",
                outcome="passed",
                reason_code="validation_verified",
                fresh=True,
            )
            for item in completion_evidence
        ],
        finding_assignments=[],
        registered_goal_ids=[CONTROL_SURFACE_PARITY_OBJECTIVE_ID],
        evaluated_at=now.isoformat(),
        repository_tree=operational.repository_tree,
    )
    health = AnalyzerHealthReport(
        status=AnalyzerHealthStatus.HEALTHY,
        reasons=(),
        thresholds=AnalyzerHealthThresholds(),
        metrics={
            "objective_id": CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
            "repository_tree": operational.repository_tree,
            "analyzer_version": (
                CONTROL_SURFACE_PARITY_COMPLETION_ANALYZER_VERSION
            ),
        },
    )
    binding = ExhaustionBinding(
        repository_id="repository:control",
        tree_id=operational.repository_tree,
        analyzer_version=(
            CONTROL_SURFACE_PARITY_COMPLETION_ANALYZER_VERSION
        ),
        configuration_revision=(
            CONTROL_SURFACE_PARITY_COMPLETION_CONFIGURATION_REVISION
        ),
        objective_revision=CONTROL_SURFACE_PARITY_OBJECTIVE_REVISION,
    )
    generic_quorum = ExhaustionQuorumResult(
        binding=binding,
        required_members=2,
        members=(
            ExhaustionQuorumMember(
                member_id="asi-067-implementation",
                evidence_channel="implementation-validation",
                receipt_cid="scan:asi-067:implementation",
                binding=binding,
                scan_mode="exhaustive",
                finished_at=now.isoformat(),
            ),
            ExhaustionQuorumMember(
                member_id="asi-067-replay",
                evidence_channel="receipt-replay-audit",
                receipt_cid="scan:asi-067:replay",
                binding=binding,
                scan_mode="exhaustive",
                finished_at=now.isoformat(),
            ),
        ),
    )
    member_health = tuple(
        ControlSurfaceParityCompletionMemberHealth(
            member_id=member.member_id,
            receipt_cid=member.receipt_cid,
            healthy=True,
            safe_for_completion_reasoning=True,
        )
        for member in generic_quorum.members
    )
    quorum = ControlSurfaceParityCompletionQuorumEvidence(
        validation_policy_id=operational.policy_id,
        policy_revision=operational.policy_revision,
        operational_receipt_id=operational.content_id,
        quorum=generic_quorum,
        member_health=member_health,
    )
    assert (
        ControlSurfaceParityCompletionQuorumEvidence.from_json(
            quorum.to_json()
        )
        == quorum
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

    incomplete_tasks = operational.evaluate_objective_completion(
        current_state=GoalState.PROVISIONALLY_COMPLETE,
        **{**values, "tasks_complete": False},
    )
    assert incomplete_tasks.state is GoalState.REOPENED
    assert not incomplete_tasks.verified

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
    stale_evidence = list(completion_evidence)
    stale_evidence[0] = CompletionEvidence.from_dict(
        {
            **stale_evidence[0].to_dict(),
            "observed_at": (now - timedelta(seconds=301)).isoformat(),
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
    mapping_coverage = coverage.completion_gate_evidence(
        CONTROL_SURFACE_PARITY_OBJECTIVE_ID
    )
    incomplete_coverage = copy.deepcopy(mapping_coverage)
    incomplete_coverage["criteria"] = incomplete_coverage["criteria"][:-1]
    unsafe_health = {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": False,
        "objective_id": CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
        "repository_tree": operational.repository_tree,
        "analyzer_version": (
            CONTROL_SURFACE_PARITY_COMPLETION_ANALYZER_VERSION
        ),
    }
    wrong_analyzer_health = {
        **unsafe_health,
        "safe_for_completion_reasoning": True,
        "analyzer_version": "asi-g103-objective-validation@stale",
    }
    with pytest.raises(ControlContractError, match="cover every quorum"):
        ControlSurfaceParityCompletionQuorumEvidence(
            validation_policy_id=operational.policy_id,
            policy_revision=operational.policy_revision,
            operational_receipt_id=operational.content_id,
            quorum=generic_quorum,
            member_health=member_health[:-1],
        )
    with pytest.raises(ControlContractError, match="explicitly healthy"):
        ControlSurfaceParityCompletionQuorumEvidence(
            validation_policy_id=operational.policy_id,
            policy_revision=operational.policy_revision,
            operational_receipt_id=operational.content_id,
            quorum=generic_quorum,
            member_health=(
                ControlSurfaceParityCompletionMemberHealth(
                    member_id=member_health[0].member_id,
                    receipt_cid=member_health[0].receipt_cid,
                    healthy=False,
                    safe_for_completion_reasoning=True,
                ),
                member_health[1],
            ),
        )
    rejected_inputs = (
        {"evidence": tuple(detached_evidence)},
        {"evidence": tuple(stale_evidence)},
        {"evidence": tuple(failed_evidence)},
        {"evidence": completion_evidence[:-1]},
        {"coverage": incomplete_coverage},
        {"analyzer_health": unsafe_health},
        {"analyzer_health": wrong_analyzer_health},
        {"exhaustion_quorum": generic_quorum},
        {
            "exhaustion_quorum": ControlSurfaceParityCompletionQuorumEvidence(
                validation_policy_id="policy:foreign",
                policy_revision=operational.policy_revision,
                operational_receipt_id=operational.content_id,
                quorum=generic_quorum,
                member_health=member_health,
            )
        },
        {
            "exhaustion_quorum": ControlSurfaceParityCompletionQuorumEvidence(
                validation_policy_id=operational.policy_id,
                policy_revision=operational.policy_revision,
                operational_receipt_id="sha256:detached",
                quorum=generic_quorum,
                member_health=member_health,
            )
        },
    )
    for replacement in rejected_inputs:
        rejected = operational.evaluate_objective_completion(
            current_state=GoalState.PROVISIONALLY_COMPLETE,
            **{**values, **replacement},
        )
        assert rejected.state is GoalState.PROVISIONALLY_COMPLETE
        assert not rejected.verified
        assert rejected.gate is not None and not rejected.gate.passed


def test_surface_parity_evidence_rejects_behavior_or_schema_drift(
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repo_root.mkdir()
    state_root.mkdir()
    service = _parity_service(repo_root, state_root)
    request = _read_request(
        repo_root,
        state_root,
        Operation.STATUS,
        objective_id=CONTROL_SURFACE_PARITY_OBJECTIVE_ID,
    )
    record = service.execute(request).to_record()
    drifted = dict(record)
    drifted["data"] = {"state": "degraded"}
    drifted.pop("content_id")

    with pytest.raises(ControlContractError, match="not canonically identical"):
        ControlSurfaceParityCase(
            scenario="read_success",
            request=request,
            python_result=record,
            cli_result=drifted,
            mcp_result=record,
        )

    cases = _parity_cases(service, repo_root, state_root)
    with pytest.raises(ControlContractError, match="objective_id"):
        ControlSurfaceParityEvidence(
            repository_tree=request.tree_id,
            objective_id="ASI-G999",
            policy_id=request.policy_id,
            policy_revision=request.policy_revision,
            capability_report=service.capability_report(),
            cases=cases,
        )
    evidence = ControlSurfaceParityEvidence(
        repository_tree=request.tree_id,
        objective_id=request.objective_id,
        policy_id=request.policy_id,
        policy_revision=request.policy_revision,
        capability_report=service.capability_report(),
        cases=cases,
    ).to_record()
    evidence["request_schema_id"] = "sha256:forged"
    evidence.pop("content_id")
    with pytest.raises(ControlContractError, match="request_schema_id"):
        ControlSurfaceParityEvidence.from_dict(evidence)
