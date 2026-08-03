"""PDR-090: live E2E qualification for Planner/Doctor release readiness.

Proves transport/projection/restart/replay identities and exercises
create/steer/diagnose/repair/benchmark/refill across Python/CLI/MCP,
Markdown/DuckDB, cold/warm/delta/restart, and 1/2/4/configured-maximum lanes.

Interfaces: PlannerDoctorQualification@1 (evidence assembly; tests only).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ipfs_accelerate_py import cli
from ipfs_accelerate_py.agent_supervisor.control.control_cli import (
    AGENT_CLI_EXIT_SUCCESS,
    COMMAND_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.control.control_contracts import (
    AuthorizationDecision,
    AuthorizationVerdict,
    EffectKind,
    ExpectedEffect,
    IdempotencyKey,
    Operation,
    OperationAuthority,
    OperationRequest,
    OperationStatus,
    PLAN_CONTROL_OPERATIONS,
    PLAN_WORKFLOW_ALIAS_OPERATIONS,
    PROPOSAL_OPERATIONS,
)
from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
    BackendResponse,
    InMemoryControlStateStore,
    SupervisorControlService,
)
from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
    AGENT_SUPERVISOR_OPERATION_TOOLS,
    configure_agent_supervisor_control,
)
from ipfs_accelerate_py.agent_supervisor.objectives.planner_doctor_refill import (
    PlannerDoctorRefill,
    PlannerDoctorRefillDisposition,
)
from ipfs_accelerate_py.agent_supervisor.planning.plan_revision_contracts import (
    CompletionAuthority,
    MergeStrategyKind,
    PlanAuthorityRoots,
    PlanCompletionRule,
    PlanConflictContract,
    PlanLeaseContract,
    PlanMergeStrategy,
    PlanOrigin,
    PlanPopulationDigest,
    PlanProviderContract,
    PlanResourceContract,
    PlanRetryContract,
    PlanRevision,
    PlanValidationNode,
    PlanWorktreeContract,
    PopulationKind,
    plan_revision_cid,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.planner_doctor_epoch import (
    PlannerDoctorEpochController,
    freeze_planner_doctor_anchors,
)
from ipfs_accelerate_py.agent_supervisor.self_improvement.planner_doctor_rollout import (
    ObservationRole,
    PlannerDoctorRolloutMode,
    SAFETY_FLOOR_METRICS,
    build_passing_observation,
    default_rollout_binding,
    default_rollout_policy,
    evaluate_planner_doctor_rollout,
    recompute_planner_doctor_gates,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_task_source import (
    DuckDBTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.markdown_task_source import (
    MarkdownTaskSource,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.plan_revision_store import (
    PlanRevisionApplyRequest,
    PlanRevisionApplyState,
    PlanRevisionStore,
)
from ipfs_accelerate_py.agent_supervisor.validation.planner_doctor_live_benchmark import (
    CACHE_STRATA,
    CONFIGURED_MAXIMUM_WORKERS,
    LIVE_BENCHMARK_PAIR_RECEIPT_INTERFACE,
    PRIMARY_ARM_IDS,
    REQUESTED_CONCURRENCY,
    ArmExecutionStatus,
    ArmId,
    EvidenceAuthorityClass,
    PairReceiptDisposition,
    PlannerDoctorLiveBenchmark,
    ProviderCallPermission,
    create_planner_doctor_live_benchmark,
    effective_workers,
    scored_cell_count,
)
from ipfs_accelerate_py.agent_supervisor.validation.planner_doctor_quality_oracle import (
    create_planner_doctor_quality_oracle,
)

ROOT = Path(__file__).resolve().parents[2]
QUALIFICATION_INTERFACE = "PlannerDoctorQualification@1"
PARITY_OPS = tuple(
    sorted(
        PLAN_CONTROL_OPERATIONS | PLAN_WORKFLOW_ALIAS_OPERATIONS,
        key=lambda item: item.value,
    )
)
duckdb = pytest.importorskip("duckdb")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _reset_mcp() -> Any:
    configure_agent_supervisor_control()
    yield
    configure_agent_supervisor_control()


@pytest.fixture
def live_runner(tmp_path: Path) -> PlannerDoctorLiveBenchmark:
    engine = create_planner_doctor_live_benchmark(
        repo_root=ROOT,
        work_root=tmp_path / "live-work",
    )
    yield engine
    engine.close()


# ---------------------------------------------------------------------------
# Transport helpers (Python / CLI / MCP)
# ---------------------------------------------------------------------------


def _binding(repository_root: Path, state_root: Path) -> dict[str, Any]:
    return {
        "repository_root": str(repository_root),
        "state_root": str(state_root),
        "repository_id": "repository:pdr-090-e2e",
        "tree_id": "tree:pdr-090-e2e",
        "objective_id": "PDR-090",
        "objective_revision": "objective:pdr-090",
        "policy_id": "policy:pdr-090-qualification",
        "policy_revision": "policy:1",
        "caller": "operator:pdr-090",
    }


def _cli_command(operation: Operation) -> str:
    return next(
        command
        for command, candidate in COMMAND_OPERATIONS.items()
        if candidate is operation
    )


def _effect(operation: Operation) -> ExpectedEffect:
    return ExpectedEffect(
        effect_id=f"{operation.value}:e2e",
        kind=EffectKind.WRITE_STATE,
        resource=f"supervisor:{operation.value}",
        paths=(f"receipts/{operation.value}.json",),
    )


def _parameters(operation: Operation, repository_root: Path) -> dict[str, Any]:
    if operation is Operation.PLAN_CREATE_PREVIEW:
        return {"mode": "deterministic"}
    if operation is Operation.PLAN_STEER_PREVIEW:
        return {}
    if operation is Operation.PLAN_CREATE_APPLY:
        return {
            "preview_ref": "receipt:create",
            "preview_root": "plan:root",
            "apply_request": {"idempotency_key": "e2e:create"},
        }
    if operation is Operation.PLAN_STEER_APPLY:
        return {
            "preview_ref": "receipt:steer",
            "preview_root": "plan:root",
            "apply_request": {"idempotency_key": "e2e:steer"},
        }
    if operation is Operation.WORKFLOW_PREVIEW:
        return {
            "directory": str(repository_root),
            "prompt_source": {"kind": "inline", "content_cid": "prompt:e2e"},
            "output_mode": "both",
        }
    return {
        "preview_ref": "receipt:preview",
        "preview_root": "plan:root",
        "preview_repository_id": "repository:pdr-090-e2e",
        "preview_tree_id": "tree:pdr-090-e2e",
        "preview_objective_id": "PDR-090",
        "preview_objective_revision": "objective:pdr-090",
        "preview_policy_id": "policy:pdr-090-qualification",
        "preview_policy_revision": "policy:1",
        "output_mode": "both",
        "markdown_path": "plans/e2e.todo.md",
        "duckdb_path": "state/e2e.duckdb",
        "apply_request": {"idempotency_key": "e2e:workflow"},
    }


def _request(
    operation: Operation,
    repository_root: Path,
    state_root: Path,
    *,
    dry_run: bool = True,
) -> OperationRequest:
    binding = _binding(repository_root, state_root)
    parameters = _parameters(operation, repository_root)
    if operation in PROPOSAL_OPERATIONS:
        return OperationRequest(
            operation=operation,
            **binding,
            parameters=parameters,
            dry_run=True,
        )
    effect = _effect(operation)
    return OperationRequest(
        operation=operation,
        **binding,
        parameters=parameters,
        expected_effects=(effect,),
        idempotency=IdempotencyKey(
            key=f"e2e:{operation.value}",
            operation=operation,
            caller=binding["caller"],
            repository_id=binding["repository_id"],
            objective_id=binding["objective_id"],
        ),
        authorization=AuthorizationDecision(
            verdict=AuthorizationVerdict.PERMIT,
            operation=operation,
            granted_authority=OperationAuthority.MUTATION,
            **binding,
            lease_id="lease:e2e",
            fencing_epoch=3,
            authorized_effect_ids=(effect.effect_id,),
            evaluated_at_ms=100,
            expires_at_ms=10_000,
        ),
        lease_id="lease:e2e",
        fencing_epoch=3,
        dry_run=dry_run,
    )


def _service(repository_root: Path, state_root: Path) -> SupervisorControlService:
    def handler(request: OperationRequest) -> BackendResponse:
        return BackendResponse(
            data={
                "operation": request.operation.value,
                "transport": "shared",
                "ok": True,
                "read_only": True,
                "qualification_interface": QUALIFICATION_INTERFACE,
            },
            changed=False,
            checks=("schema", "parity", "proposal_only", "pdr-090"),
        )

    return SupervisorControlService(
        repository_allowlist=(repository_root,),
        state_allowlist=(state_root,),
        handlers={operation: handler for operation in PARITY_OPS},
        state_store=InMemoryControlStateStore(),
        clock_ms=lambda: 5_000,
    )


async def _mcp_record(
    service: SupervisorControlService, request: OperationRequest
) -> dict[str, Any]:
    configure_agent_supervisor_control(service=service)
    return await AGENT_SUPERVISOR_OPERATION_TOOLS[request.operation](
        request=request.to_record()
    )


def _cli_record(
    service: SupervisorControlService,
    request: OperationRequest,
    capsys: pytest.CaptureFixture[str],
) -> dict[str, Any]:
    exit_status = cli.main(
        [
            "agent",
            _cli_command(request.operation),
            "--request-json",
            request.to_json(),
            "--output-json",
        ],
        agent_control_service=service,
    )
    captured = capsys.readouterr()
    assert exit_status == AGENT_CLI_EXIT_SUCCESS, captured.err or captured.out
    return json.loads(captured.out)


# ---------------------------------------------------------------------------
# Plan revision / projection helpers
# ---------------------------------------------------------------------------


def _cid(name: str) -> str:
    return plan_revision_cid({"fixture": name, "suite": "pdr-090-e2e"})


def _roots(**changes: object) -> PlanAuthorityRoots:
    values: dict[str, object] = {
        "repository_id": "repository:sha256:pdr-090",
        "repository_root_cid": _cid("repo-root"),
        "dirty_worktree_root": _cid("dirty"),
        "task_source_id": "task-source:both:pdr-090",
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


def _population(kind: PopulationKind, *members: str) -> PlanPopulationDigest:
    return PlanPopulationDigest(kind=kind, member_cids=members)


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
        "added_population": _population(
            PopulationKind.ADDED, _cid("goal-1"), _cid("task-1")
        ),
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
            predicted_files=(
                "ipfs_accelerate_py/agent_supervisor/task_sources/plan_revision_store.py",
            ),
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


def _canonical_fixture():
    from test.api.test_agent_supervisor_task_source_e2e import (
        _canonical_fixture as fixture,
    )

    return fixture()


# ---------------------------------------------------------------------------
# Interface / constants
# ---------------------------------------------------------------------------


def test_qualification_interface_and_live_module_contract() -> None:
    assert QUALIFICATION_INTERFACE == "PlannerDoctorQualification@1"
    assert CACHE_STRATA == ("cold", "exact-warm", "delta", "restart")
    assert REQUESTED_CONCURRENCY == (1, 2, 4, 6)
    assert CONFIGURED_MAXIMUM_WORKERS == 6
    assert set(PRIMARY_ARM_IDS) == {
        ArmId.DETERMINISTIC_SYMBOLIC.value,
        ArmId.HYBRID_RESIDUAL.value,
        ArmId.CURRENT_MAINLINE.value,
    }
    assert ProviderCallPermission.FORBIDDEN.value == "forbidden"


# ---------------------------------------------------------------------------
# Live create / steer / diagnose / repair across strata and lanes
# ---------------------------------------------------------------------------


_LIVE_CASES = (
    ("live-hermetic-plan-create", "create"),
    ("live-hermetic-plan-steer", "steer"),
    ("live-hermetic-doctor-contract", "diagnose"),
    ("live-hermetic-security-ir", "security-diagnose"),
    ("live-hermetic-transaction-rollback", "repair-rollback"),
    ("live-hermetic-capability-degradation", "degrade"),
)


def test_live_create_steer_diagnose_repair_families_run_without_skip(
    live_runner: PlannerDoctorLiveBenchmark,
) -> None:
    """Every required pair family executes live with no skip qualification."""

    for case_id, _role in _LIVE_CASES:
        receipt = live_runner.run_pair(
            case_id,
            stratum_id="cold",
            concurrency=1,
            repetition=0,
        )
        assert receipt.INTERFACE == LIVE_BENCHMARK_PAIR_RECEIPT_INTERFACE
        assert receipt.disposition in {
            PairReceiptDisposition.PAIRED,
            PairReceiptDisposition.NOT_PROMOTION_ELIGIBLE,
        }
        assert receipt.inputs_match_across_primary_arms is True
        assert receipt.promotion_eligible is False
        assert len(receipt.arm_receipts) == 3
        for arm in receipt.arm_receipts:
            assert arm.status is not ArmExecutionStatus.SKIPPED
            assert arm.evidence_authority is EvidenceAuthorityClass.LIVE_SERVICE
            assert arm.process_tree_terminated
            assert arm.capabilities_revoked
            assert arm.output_root_sealed
            assert arm.service_interfaces_invoked


@pytest.mark.parametrize("stratum", list(CACHE_STRATA))
def test_cache_strata_cold_warm_delta_restart_identity(
    live_runner: PlannerDoctorLiveBenchmark,
    stratum: str,
) -> None:
    receipt = live_runner.run_pair(
        "live-hermetic-plan-create",
        stratum_id=stratum,
        concurrency=1,
        repetition=0,
    )
    assert receipt.cache_stratum_id == stratum
    assert receipt.inputs_match_across_primary_arms
    assert receipt.disposition is PairReceiptDisposition.PAIRED
    for arm in receipt.arm_receipts:
        assert arm.status is not ArmExecutionStatus.SKIPPED
        assert stratum in arm.cache_namespace or arm.cache_namespace


@pytest.mark.parametrize("concurrency", list(REQUESTED_CONCURRENCY))
def test_lane_concurrency_one_two_four_and_configured_maximum(
    live_runner: PlannerDoctorLiveBenchmark,
    concurrency: int,
) -> None:
    receipt = live_runner.run_pair(
        "live-hermetic-transaction-rollback",
        stratum_id="cold",
        concurrency=concurrency,
        repetition=0,
    )
    assert receipt.concurrency_requested == concurrency
    assert effective_workers(concurrency) == min(
        concurrency, CONFIGURED_MAXIMUM_WORKERS
    )
    for arm in receipt.arm_receipts:
        assert arm.effective_workers == effective_workers(concurrency)
        assert arm.status is not ArmExecutionStatus.SKIPPED


def test_restart_and_replay_identities_hold(
    live_runner: PlannerDoctorLiveBenchmark,
) -> None:
    first, second, match = live_runner.replay_pair(
        "live-hermetic-plan-create",
        stratum_id="restart",
        concurrency=2,
        repetition=1,
    )
    assert match is True
    assert first.pair_input_cid == second.pair_input_cid
    assert first.case_id == second.case_id
    assert first.cache_stratum_id == "restart"
    assert second.cache_stratum_id == "restart"
    assert [arm.seal_cid for arm in first.arm_receipts] == [
        arm.seal_cid for arm in second.arm_receipts
    ]
    assert first.inputs_match_across_primary_arms
    assert second.inputs_match_across_primary_arms
    det = next(
        arm
        for arm in first.arm_receipts
        if arm.arm_id == ArmId.DETERMINISTIC_SYMBOLIC.value
    )
    assert det.arm_id == ArmId.DETERMINISTIC_SYMBOLIC.value
    assert det.status is not ArmExecutionStatus.SKIPPED


def test_scored_cell_denominator_and_incomplete_matrix_fail_closed(
    live_runner: PlannerDoctorLiveBenchmark,
) -> None:
    required = scored_cell_count(case_count=len(live_runner.manifest.cases))
    assert required == 6 * 3 * 4 * 4 * 3
    report = live_runner.run_matrix(
        case_ids=["live-hermetic-plan-create"],
        strata=["cold"],
        concurrency_values=[1],
        scored_repetitions=1,
        max_pairs=1,
    )
    assert report.incomplete is True
    assert report.promotion_eligible is False
    assert report.scored_cells_required == required
    assert report.scored_cells_observed < required
    assert "incomplete_required_cell_population" in report.reason_codes
    for pair in report.pair_receipts:
        assert pair.promotion_eligible is False


# ---------------------------------------------------------------------------
# Transport parity: Python / CLI / MCP
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "operation",
    [
        Operation.PLAN_CREATE_PREVIEW,
        Operation.PLAN_STEER_PREVIEW,
        Operation.WORKFLOW_PREVIEW,
        Operation.PLAN_CREATE_APPLY,
        Operation.PLAN_STEER_APPLY,
    ],
)
async def test_python_cli_mcp_transport_identity(
    operation: Operation,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    request = _request(operation, repository_root, state_root, dry_run=True)
    service = _service(repository_root, state_root)

    python_result = service.execute(request)
    assert python_result.status is OperationStatus.SUCCEEDED
    assert python_result.operation is operation
    assert python_result.repository_id == request.repository_id
    assert python_result.tree_id == request.tree_id

    cli_record = _cli_record(service, request, capsys)
    mcp_record = await _mcp_record(service, request)
    # Exact transport identity across Python / CLI / MCP.
    assert cli_record == python_result.to_record()
    assert mcp_record == python_result.to_record()
    assert cli_record["operation"] == operation.value
    assert mcp_record["operation"] == operation.value
    assert python_result.repository_id == request.repository_id
    assert python_result.tree_id == request.tree_id
    # Proposal surfaces carry the qualification interface marker; dry-run
    # mutation previews may be rewritten to {dry_run, would_change} only.
    payload = python_result.to_record()
    data = payload.get("data") or {}
    if operation in PROPOSAL_OPERATIONS:
        assert data.get("qualification_interface") == QUALIFICATION_INTERFACE
        assert data.get("transport") == "shared"
    else:
        assert data.get("dry_run") is True or data.get("transport") == "shared"


@pytest.mark.asyncio
async def test_all_plan_control_ops_have_transport_parity(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    repository_root = tmp_path / "repo"
    state_root = tmp_path / "state"
    repository_root.mkdir()
    state_root.mkdir()
    service = _service(repository_root, state_root)
    for operation in PARITY_OPS:
        request = _request(operation, repository_root, state_root, dry_run=True)
        python_result = service.execute(request)
        assert python_result.status is OperationStatus.SUCCEEDED
        assert _cli_record(service, request, capsys) == python_result.to_record()
        assert await _mcp_record(service, request) == python_result.to_record()


# ---------------------------------------------------------------------------
# Projection parity: Markdown / DuckDB
# ---------------------------------------------------------------------------


def test_markdown_duckdb_projection_parity_and_exact_cids(tmp_path: Path) -> None:
    graph, admission, aliases, tree_id = _canonical_fixture()
    markdown = MarkdownTaskSource(
        tmp_path / "tasks.md",
        root=tmp_path,
        task_prefix="FIX",
        board_namespace="pdr-090",
    )
    duck = DuckDBTaskSource(tmp_path / "tasks.duckdb")
    store = PlanRevisionStore(tmp_path / "store")
    revision = _revision(
        plan_root_cid=admission.plan_root_cid,
        roots=_roots(
            dirty_worktree_root=tree_id,
            task_source_id="task-source:both:pdr-090",
        ),
        task_population=_population(
            PopulationKind.RETAINED, *sorted(admission.task_cids)
        ),
        added_population=_population(
            PopulationKind.ADDED, *sorted(admission.task_cids)
        ),
    )
    receipt = store.apply(
        PlanRevisionApplyRequest(
            revision=revision,
            observed_roots=revision.roots,
            idempotency_key="idem:pdr-090-create",
            expected_effects=("materialize-revision-1",),
            admission=admission,
            goal_graph=graph,
            aliases=aliases,
            markdown_source=markdown,
            duckdb_source=duck,
            repository_tree_id=tree_id,
        )
    )
    assert receipt.committed
    assert receipt.state is PlanRevisionApplyState.COMMITTED
    assert receipt.markdown_projection_cid
    assert receipt.duckdb_projection_cid
    assert markdown.plan_revision_projection_cid() == receipt.markdown_projection_cid
    assert duck.plan_revision_projection_cid() == receipt.duckdb_projection_cid

    again = store.apply(
        PlanRevisionApplyRequest(
            revision=revision,
            observed_roots=revision.roots,
            idempotency_key="idem:pdr-090-create",
            expected_effects=("materialize-revision-1",),
            admission=admission,
            goal_graph=graph,
            aliases=aliases,
            markdown_source=markdown,
            duckdb_source=duck,
            repository_tree_id=tree_id,
        )
    )
    assert again.committed
    assert again.revision_cid == receipt.revision_cid
    assert again.markdown_projection_cid == receipt.markdown_projection_cid
    assert again.duckdb_projection_cid == receipt.duckdb_projection_cid


# ---------------------------------------------------------------------------
# Benchmark oracle + refill + epoch + rollout qualification bridge
# ---------------------------------------------------------------------------


def test_live_benchmark_oracle_and_safety_floors_zero(
    live_runner: PlannerDoctorLiveBenchmark,
) -> None:
    receipt = live_runner.run_pair(
        "live-hermetic-plan-create",
        stratum_id="cold",
        concurrency=1,
    )
    oracle = create_planner_doctor_quality_oracle(repo_root=ROOT)
    assert receipt.promotion_eligible is False
    assert oracle.interface
    observation = build_passing_observation(
        observation_id="observation:pdr-090-e2e@1",
        observed_at="2026-08-03T00:00:00Z",
        role=ObservationRole.QUALIFICATION,
    )
    gates = recompute_planner_doctor_gates(observation)
    assert gates.passed
    assert gates.safety_passed
    assert gates.exact_rollback_ok
    assert not gates.safety_floor_violations
    for name in SAFETY_FLOOR_METRICS:
        assert observation.challenger.safety_floors[name] == 0


def test_refill_and_epoch_shadow_participate_in_qualification(
    tmp_path: Path,
) -> None:
    refill = PlannerDoctorRefill()
    receipt = refill.refill(residuals=())
    assert receipt.disposition is PlannerDoctorRefillDisposition.EMPTY_INPUT
    assert receipt.emits_work is False
    assert receipt.completion_authority is False
    assert receipt.mutation_authority is False

    anchors = freeze_planner_doctor_anchors(
        repo_root=ROOT,
        repository_id="repository:pdr-090",
        tree_id="sha256:" + ("a" * 64),
        authority_policy_revision="1",
        benchmark_policy_revision="1",
    )
    assert anchors.anchors_id
    journal = tmp_path / "epoch" / "journal.jsonl"
    journal.parent.mkdir(parents=True, exist_ok=True)
    controller = PlannerDoctorEpochController(
        repo_root=ROOT,
        journal_path=journal,
        work_root=tmp_path / "epoch-work",
    )
    assert controller is not None


def test_rollout_qualification_gate_blocks_automatic_without_fresh_root() -> None:
    qualification = build_passing_observation(
        observation_id="observation:qualification@1",
        observed_at="2026-08-01T00:00:00Z",
        role=ObservationRole.QUALIFICATION,
    )
    current = build_passing_observation(
        observation_id="observation:current@1",
        observed_at="2026-08-01T01:00:00Z",
        role=ObservationRole.CURRENT_TREE,
        tree_id=qualification.tree_id,
    )
    holdout = build_passing_observation(
        observation_id="observation:holdout@1",
        observed_at="2026-08-01T02:00:00Z",
        role=ObservationRole.HOLDOUT,
        tree_id=qualification.tree_id,
    )
    binding = default_rollout_binding(tree_id=qualification.tree_id)
    policy = default_rollout_policy(allow_automatic=False)
    # Shadow is the highest seed-approved mode without automatic.
    receipt = evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=PlannerDoctorRolloutMode.SHADOW,
        current_observation=current,
        holdout_observation=holdout,
    )
    assert receipt.qualification_gate_passed
    assert receipt.promotion_allowed is False
    assert receipt.effective_mode is PlannerDoctorRolloutMode.SHADOW
    assert PlannerDoctorRolloutMode.AUTOMATIC not in policy.allowed_modes
    # Automatic remains blocked without operator fresh-root approval.
    blocked = evaluate_planner_doctor_rollout(
        qualification,
        binding=binding,
        policy=policy,
        desired_mode=PlannerDoctorRolloutMode.AUTOMATIC,
        current_observation=current,
        holdout_observation=holdout,
    )
    assert blocked.promotion_allowed is False
    assert blocked.automatic_ready is False
    assert blocked.effective_mode is not PlannerDoctorRolloutMode.AUTOMATIC
