from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
    DirectoryScanPolicy,
    DirectoryScanReceipt,
    IncidentKind,
    MaterializationReference,
    NonCanonicalPromptWorkflowError,
    OutputMode,
    ProgrammaticRecoveryExhaustionReceipt,
    PromptAcceptanceRecord,
    PromptEvidenceRecord,
    PromptGoalGraph,
    PromptGoalRecord,
    PromptOutputPolicy,
    PromptOutputRecord,
    PromptPlanningPolicy,
    PromptSecretError,
    PromptSource,
    PromptSourceError,
    PromptTaskRecord,
    PromptValidationRecord,
    PromptWorkflowBoundsError,
    PromptWorkflowBudget,
    PromptWorkflowContractError,
    PromptWorkflowIdentityError,
    PromptWorkflowPathError,
    PromptWorkflowPreviewReceipt,
    PromptWorkflowRequest,
    PromptWorkflowResult,
    RecordStatus,
    RecoveryAttempt,
    RecoveryAttemptOutcome,
    RescueAction,
    RescueOperation,
    RescuePlan,
    RescuePlanError,
    SupervisorIncident,
    SupervisorRunReference,
    WorkflowOutcome,
    prompt_workflow_cid,
)


def _cid(name: str) -> str:
    return prompt_workflow_cid({"fixture": name})


def _budget(**changes: int) -> PromptWorkflowBudget:
    values = {
        "max_files": 1_000,
        "max_scan_bytes": 8 * 1024 * 1024,
        "max_file_bytes": 512 * 1024,
        "max_symbols": 10_000,
        "max_prompt_tokens": 4_096,
        "max_provider_tokens": 8_192,
        "max_latency_ms": 60_000,
        "max_goals": 16,
        "max_tasks": 64,
        "max_evidence": 128,
        "max_graph_depth": 8,
        "max_serialized_bytes": 256 * 1024,
        "max_rescue_actions": 4,
    }
    values.update(changes)
    return PromptWorkflowBudget(**values)


def _request(**changes: object) -> PromptWorkflowRequest:
    values: dict[str, object] = {
        "prompt_source": PromptSource.inline(
            "Improve retry recovery",
            redacted_metadata={
                "summary": "Retry recovery improvement request",
                "sensitivity": "redacted",
            },
        ),
        "repository_root": "/workspace/repository",
        "directory": "/workspace/repository/ipfs_accelerate_py",
        "repository_root_cid": _cid("repository"),
        "allowlist_cid": _cid("allowlist"),
        "scan_policy": DirectoryScanPolicy(
            policy_id="scan:default",
            scanner_version="1.0.0",
            include_patterns=("**/*.py",),
            exclude_patterns=(".git/**", "**/__pycache__/**"),
        ),
        "planning_policy": PromptPlanningPolicy(
            policy_id="planning:strict",
            provider_preferences=("local",),
            model_preferences=("deterministic",),
        ),
        "output_policy": PromptOutputPolicy(
            policy_id="output:repo",
            mode=OutputMode.BOTH,
            output_root="/workspace/repository",
            allowed_output_roots=("/workspace/repository",),
            markdown_path="plans/work.todo.md",
            duckdb_path="state/work.duckdb",
            board_namespace="prompt-workflow-test",
            task_prefix="PWT",
        ),
        "budget": _budget(),
        "caller": "principal:test-suite",
        "program_root": _cid("program"),
        "intent_ir_root": _cid("intent"),
        "legal_ir_root": _cid("legal"),
        "security_ir_root": _cid("security"),
        "policy_root": _cid("policy"),
        "dry_run": True,
        "materialize": False,
        "start_after_materialize": False,
    }
    values.update(changes)
    return PromptWorkflowRequest(**values)


def _evidence(**changes: object) -> PromptEvidenceRecord:
    values: dict[str, object] = {
        "evidence_key": "scan:retry-module",
        "source_kind": "directory_scan",
        "artifact_cid": _cid("retry-module-evidence"),
        "summary": "Retry handling is implemented in the supervisor package.",
        "repository_paths": (
            "ipfs_accelerate_py/agent_supervisor/supervisor_recovery.py",
        ),
        "claim_keys": ("claim:retry-location",),
        "provenance": {"scanner": "fixture"},
    }
    values.update(changes)
    return PromptEvidenceRecord(**values)


def _acceptance(evidence_cid: str) -> PromptAcceptanceRecord:
    return PromptAcceptanceRecord(
        criterion_key="criterion:tests",
        criterion="The focused contract tests pass.",
        evidence_cids=(evidence_cid,),
        validation_keys=("validation:pytest",),
    )


def _validation() -> PromptValidationRecord:
    return PromptValidationRecord(
        validation_key="validation:pytest",
        argv=(
            "python",
            "-m",
            "pytest",
            "test/api/test_agent_supervisor_prompt_workflow_contracts.py",
            "-q",
        ),
        cwd=".",
        policy_cid=_cid("validation-policy"),
    )


def _graph(**changes: object) -> PromptGoalGraph:
    evidence = _evidence()
    acceptance = _acceptance(evidence.evidence_cid)
    root = PromptGoalRecord(
        goal_key="goal:root",
        parent_goal_cid="",
        dependency_goal_cids=(),
        title="Improve retry recovery",
        objective="Define a bounded, evidence-backed retry improvement.",
        rationale="The prompt and scan evidence identify retry recovery.",
        scope_paths=("ipfs_accelerate_py/agent_supervisor",),
        acceptance=(acceptance,),
        evidence_cids=(evidence.evidence_cid,),
        provenance={"objective": "model", "scope": "deterministic"},
    )
    task = PromptTaskRecord(
        task_key="task:contracts",
        goal_cid=root.goal_cid,
        dependency_task_cids=(),
        objective="Implement canonical retry workflow contracts.",
        rationale="The root goal requires a durable task contract.",
        scope_paths=("ipfs_accelerate_py/agent_supervisor/prompt_workflow.py",),
        outputs=(
            PromptOutputRecord(
                path="ipfs_accelerate_py/agent_supervisor/prompt_workflow.py",
                effect="modify",
                media_type="text/x-python",
            ),
        ),
        validations=(_validation(),),
        acceptance=(acceptance,),
        evidence_cids=(evidence.evidence_cid,),
        policy_roots=(_cid("policy"), _cid("security")),
        predicted_files=(
            "ipfs_accelerate_py/agent_supervisor/prompt_workflow.py",
        ),
        provenance={"objective": "model", "scope_paths": "deterministic"},
    )
    values: dict[str, object] = {
        "request_cid": _cid("request"),
        "scan_cid": _cid("scan"),
        "program_root": _cid("program"),
        "policy_roots": (_cid("policy"), _cid("security")),
        "goals": (root,),
        "tasks": (task,),
        "evidence": (evidence,),
    }
    values.update(changes)
    return PromptGoalGraph(**values)


def test_request_is_body_free_canonical_and_binds_every_semantic_input() -> None:
    request = _request()
    encoded = request.to_json()

    assert request.prompt_source.transient_body == b"Improve retry recovery"
    assert "Improve retry recovery" not in encoded
    assert request.prompt_cid in encoded
    assert request.repository_root in encoded
    assert request.directory in encoded
    assert request.output_root in encoded
    assert request.program_root in encoded
    assert request.policy_root in encoded
    assert PromptWorkflowRequest.from_json(encoded).content_id == request.content_id

    changed = (
        replace(request, caller="principal:other"),
        replace(request, directory="/workspace/repository/test"),
        replace(request, policy_root=_cid("other-policy")),
        replace(request, budget=_budget(max_tasks=63)),
    )
    assert all(item.content_id != request.content_id for item in changed)


def test_prompt_source_is_unambiguous_and_receipts_reject_secrets() -> None:
    inline = PromptSource.inline("bounded prompt")
    with pytest.raises(PromptSourceError, match="cannot carry"):
        replace(inline, source_path="prompt.md")
    with pytest.raises(PromptSourceError, match="requires only source_path"):
        PromptSource(
            kind="file",
            prompt_cid=inline.prompt_cid,
            byte_count=inline.byte_count,
            redacted_metadata={},
        )
    with pytest.raises(PromptSecretError, match="secret-bearing"):
        replace(inline, redacted_metadata={"api_key": "redacted"})
    with pytest.raises(PromptSecretError, match="secret"):
        _evidence(provenance={"note": "sk-" + "x" * 24})
    with pytest.raises(PromptSourceError, match="raw prompt"):
        PromptSource.inline(
            "verbatim prompt",
            redacted_metadata={"summary": "verbatim prompt"},
        )
    with pytest.raises(PromptWorkflowContractError, match="at least 1"):
        replace(inline, byte_count=0, _transient_body=None)


def test_standalone_module_load_is_provider_free_and_has_no_process_effect() -> None:
    module_path = (
        Path(__file__).resolve().parents[2]
        / "ipfs_accelerate_py"
        / "agent_supervisor"
        / "prompt_workflow.py"
    )
    probe = """
import json
import runpy
import sys
before = set(sys.modules)
runpy.run_path(sys.argv[1], run_name="prompt_workflow_import_probe")
loaded = sorted(set(sys.modules) - before)
forbidden = [
    name for name in loaded
    if any(marker in name.lower() for marker in (
        "duckdb", "llm_router", "provider", "objective_graph",
        "supervisor_recovery", "multiprocessing", "subprocess",
    ))
]
print(json.dumps(forbidden))
raise SystemExit(bool(forbidden))
"""
    result = subprocess.run(
        [sys.executable, "-S", "-c", probe, str(module_path)],
        check=False,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == []


def test_paths_intents_and_mutation_bindings_fail_closed() -> None:
    with pytest.raises(PromptWorkflowPathError, match="within"):
        _request(directory="/other/repository")
    with pytest.raises(PromptWorkflowPathError, match="outside"):
        replace(
            _request().output_policy,
            output_root="/other/output",
        )
    with pytest.raises(PromptWorkflowContractError, match="dry_run"):
        _request(dry_run=True, materialize=True)
    with pytest.raises(PromptWorkflowContractError, match="requires materialize"):
        _request(dry_run=False, start_after_materialize=True)
    with pytest.raises(PromptWorkflowContractError, match="authority"):
        _request(dry_run=False, materialize=True)

    mutation = _request(
        dry_run=False,
        materialize=True,
        authority_cid=_cid("authority"),
        idempotency_key="workflow/attempt-1",
        lease_id="lease:1",
        fencing_epoch=1,
    )
    assert mutation.materialize is True


def test_scan_receipt_binds_worktree_policy_budget_and_redacted_evidence() -> None:
    request = _request()
    evidence = _evidence()
    scan = DirectoryScanReceipt(
        request_cid=request.request_cid,
        repository_root=request.repository_root,
        directory=request.directory,
        repository_root_cid=request.repository_root_cid,
        dirty_worktree_root=_cid("dirty-worktree"),
        scanner_policy_cid=request.scan_policy.content_id,
        program_root=request.program_root,
        ast_root=_cid("ast"),
        index_root=_cid("index"),
        budget=request.budget,
        evidence=(evidence,),
        counts={"files": 42, "symbols": 512},
        exclusions=("credential-like path",),
    )
    assert DirectoryScanReceipt.from_json(scan.to_json()) == scan
    assert scan.scan_cid != replace(scan, dirty_worktree_root=_cid("changed")).scan_cid

    with pytest.raises(PromptWorkflowContractError, match="exactly reflect"):
        replace(scan, truncated=True)
    with pytest.raises(PromptSecretError):
        replace(scan, counts={"api_key": 1})


def test_goal_task_and_plan_ids_ignore_order_status_and_timestamps() -> None:
    graph = _graph()
    task = graph.tasks[0]
    changed_task = replace(
        task,
        status=RecordStatus.COMPLETED,
        created_at_ms=100,
        updated_at_ms=200,
        dependency_task_cids=tuple(reversed(task.dependency_task_cids)),
        evidence_cids=tuple(reversed(task.evidence_cids)),
        policy_roots=tuple(reversed(task.policy_roots)),
    )
    assert changed_task.task_cid == task.task_cid

    changed_goal = replace(
        graph.goals[0],
        status=RecordStatus.ADMITTED,
        created_at_ms=10,
        updated_at_ms=20,
    )
    changed_evidence = replace(
        graph.evidence[0],
        status=RecordStatus.COMPLETED,
        created_at_ms=30,
        updated_at_ms=40,
    )
    changed_graph = replace(
        graph,
        goals=(changed_goal,),
        tasks=(changed_task,),
        evidence=(changed_evidence,),
        status=RecordStatus.ADMITTED,
        created_at_ms=50,
        updated_at_ms=60,
    )
    assert changed_graph.plan_root_cid == graph.plan_root_cid

    changed_scope = replace(
        task,
        scope_paths=("test/api/test_agent_supervisor_prompt_workflow_contracts.py",),
    )
    assert changed_scope.task_cid != task.task_cid


def test_graph_rejects_unknown_edges_cycles_missing_coverage_and_duplicates() -> None:
    graph = _graph()
    task = graph.tasks[0]
    with pytest.raises(PromptWorkflowContractError, match="CIDv1"):
        replace(task, goal_cid=_cid("missing") + "x")
    with pytest.raises(PromptWorkflowContractError, match="unknown task"):
        replace(graph, tasks=(replace(task, dependency_task_cids=(_cid("missing"),)),))
    with pytest.raises(PromptWorkflowContractError, match="must not be empty"):
        replace(graph, goals=())
    with pytest.raises(PromptWorkflowContractError, match="unknown evidence"):
        replace(
            graph,
            evidence=(),
        )
    unknown_acceptance = PromptAcceptanceRecord(
        criterion_key="criterion:unknown-evidence",
        criterion="An unknown artifact proves this criterion.",
        evidence_cids=(_cid("not-in-graph"),),
        validation_keys=("validation:pytest",),
    )
    with pytest.raises(
        PromptWorkflowContractError,
        match="acceptance references unknown evidence",
    ):
        replace(
            graph,
            tasks=(replace(task, acceptance=(unknown_acceptance,)),),
        )


def test_unknown_fields_forged_ids_noncanonical_json_and_bounds_are_rejected() -> None:
    request = _request()
    unknown = request.to_record()
    unknown["ambient_policy"] = "latest"
    with pytest.raises(PromptWorkflowContractError, match="unsupported fields"):
        PromptWorkflowRequest.from_dict(unknown)

    forged = request.to_record()
    forged["caller"] = "principal:attacker"
    with pytest.raises(PromptWorkflowIdentityError, match="identity"):
        PromptWorkflowRequest.from_dict(forged)

    pretty = json.dumps(request.to_record(), indent=2)
    with pytest.raises(NonCanonicalPromptWorkflowError, match="round trip"):
        PromptWorkflowRequest.from_json(pretty)

    duplicate = request.to_json().replace(
        '"caller":"principal:test-suite"',
        '"caller":"principal:test-suite","caller":"principal:attacker"',
    )
    with pytest.raises(NonCanonicalPromptWorkflowError, match="duplicate"):
        PromptWorkflowRequest.from_json(duplicate)

    with pytest.raises(PromptWorkflowBoundsError):
        _budget(max_graph_depth=33)
    with pytest.raises(PromptWorkflowContractError, match="finite integer"):
        _budget(max_files=1.5)  # type: ignore[arg-type]
    oversized = '{"value":"' + ("x" * (1024 * 1024)) + '"}'
    with pytest.raises(PromptWorkflowBoundsError, match="serialized byte"):
        PromptWorkflowRequest.from_json(oversized)


def test_materialization_run_preview_and_result_are_exactly_linked() -> None:
    request = _request()
    graph = _graph(request_cid=request.request_cid)
    preview = PromptWorkflowPreviewReceipt(
        request_cid=request.request_cid,
        scan_cid=graph.scan_cid,
        plan_root_cid=graph.plan_root_cid,
        repository_root_cid=request.repository_root_cid,
        program_root=request.program_root,
        policy_roots=graph.policy_roots,
        admitted_goal_cids=tuple(goal.goal_cid for goal in graph.goals),
        admitted_task_cids=tuple(task.task_cid for task in graph.tasks),
        expected_materialization_effects=("write_duckdb", "write_markdown"),
        budget=request.budget,
    )
    materialization = MaterializationReference(
        request_cid=request.request_cid,
        preview_receipt_cid=preview.receipt_cid,
        plan_root_cid=graph.plan_root_cid,
        repository_root=request.repository_root,
        output_root=request.output_root,
        mode=OutputMode.BOTH,
        projection_cids=(_cid("markdown"), _cid("duckdb")),
        revision=1,
    )
    run = SupervisorRunReference(
        materialization_cid=materialization.materialization_cid,
        plan_root_cid=graph.plan_root_cid,
        repository_root=request.repository_root,
        state_root="/workspace/repository/state/supervisor",
        supervisor_profile="local-parallel",
        lifecycle_request_cid=_cid("lifecycle"),
    )
    result = PromptWorkflowResult(
        request_cid=request.request_cid,
        outcome=WorkflowOutcome.STARTED,
        preview_receipt_cid=preview.receipt_cid,
        materialization=materialization,
        run=run,
        completed_stage_cids=(
            preview.receipt_cid,
            materialization.materialization_cid,
            run.run_cid,
        ),
        status=RecordStatus.RUNNING,
    )
    assert PromptWorkflowResult.from_json(result.to_json()).receipt_cid == result.receipt_cid
    assert replace(run, status=RecordStatus.RUNNING, started_at_ms=42).run_cid == run.run_cid
    assert replace(run, process_identity_cid=_cid("process-tree")).run_cid == run.run_cid
    with pytest.raises(PromptWorkflowIdentityError, match="another materialization"):
        replace(result, run=replace(run, materialization_cid=_cid("other")))
    with pytest.raises(PromptWorkflowIdentityError, match="another workflow request"):
        replace(
            result,
            materialization=replace(
                materialization,
                request_cid=_cid("other-request"),
            ),
        )
    with pytest.raises(PromptWorkflowIdentityError, match="another preview receipt"):
        replace(
            result,
            materialization=replace(
                materialization,
                preview_receipt_cid=_cid("other-preview"),
            ),
        )
    with pytest.raises(PromptWorkflowIdentityError, match="another plan root"):
        replace(result, run=replace(run, plan_root_cid=_cid("other-plan")))
    with pytest.raises(PromptWorkflowIdentityError, match="another repository root"):
        replace(result, run=replace(run, repository_root="/workspace/other"))
    with pytest.raises(PromptWorkflowContractError, match="started outcome requires"):
        replace(result, run=None)
    with pytest.raises(PromptWorkflowContractError, match="previewed outcome cannot"):
        replace(
            result,
            outcome=WorkflowOutcome.PREVIEWED,
            run=None,
        )


def test_preview_receipt_enforces_declared_count_and_byte_budgets() -> None:
    request = _request()
    graph = _graph(request_cid=request.request_cid)
    values = {
        "request_cid": request.request_cid,
        "scan_cid": graph.scan_cid,
        "plan_root_cid": graph.plan_root_cid,
        "repository_root_cid": request.repository_root_cid,
        "program_root": request.program_root,
        "policy_roots": graph.policy_roots,
        "admitted_goal_cids": tuple(goal.goal_cid for goal in graph.goals),
        "admitted_task_cids": tuple(task.task_cid for task in graph.tasks),
    }
    with pytest.raises(PromptWorkflowBoundsError, match="admitted goals"):
        PromptWorkflowPreviewReceipt(
            **{
                **values,
                "admitted_goal_cids": (_cid("goal-a"), _cid("goal-b")),
            },
            budget=_budget(max_goals=1),
        )
    with pytest.raises(PromptWorkflowBoundsError, match="admitted tasks"):
        PromptWorkflowPreviewReceipt(
            **{
                **values,
                "admitted_task_cids": (_cid("task-a"), _cid("task-b")),
            },
            budget=_budget(max_tasks=1),
        )
    with pytest.raises(PromptWorkflowBoundsError, match="serialized"):
        PromptWorkflowPreviewReceipt(
            **values,
            budget=_budget(max_serialized_bytes=1),
        )


def _incident() -> SupervisorIncident:
    return SupervisorIncident(
        repository_root="/workspace/repository",
        state_root="/workspace/repository/state/supervisor",
        repository_root_cid=_cid("repository"),
        policy_root=_cid("policy"),
        run_cid=_cid("run"),
        kind=IncidentKind.STALE_HEARTBEAT,
        failure_fingerprint="sha256:" + "a" * 64,
        target_ids=("lane:implementation",),
        evidence_cids=(_cid("health"),),
        health={"heartbeat_state": "stale", "event_cursor": 42},
        cooldown_key="incident/stale-heartbeat/lane-implementation",
    )


def test_incident_exhaustion_and_rescue_are_bound_and_closed() -> None:
    incident = _incident()
    assert (
        replace(incident, observed_at_ms=999, updated_at_ms=1000).incident_cid
        == incident.incident_cid
    )
    attempt = RecoveryAttempt(
        operation=RescueOperation.RESTART_LANE,
        target_id="lane:implementation",
        attempt=1,
        outcome=RecoveryAttemptOutcome.FAILED,
        receipt_cid=_cid("restart-failed"),
        failure_fingerprint="sha256:" + "b" * 64,
    )
    exhaustion = ProgrammaticRecoveryExhaustionReceipt(
        incident_cid=incident.incident_cid,
        repository_root_cid=incident.repository_root_cid,
        policy_root=incident.policy_root,
        run_cid=incident.run_cid,
        attempts=(attempt,),
        inapplicable_operations=(RescueOperation.REPAIR_ORPHANED_LOCK,),
        exhaustion_reason="Applicable deterministic recovery was exhausted.",
        budget=_budget(),
        circuit_open=False,
    )
    action = RescueAction(
        operation=RescueOperation.QUARANTINE,
        target_id="lane:implementation",
        parameters={"reason_code": "unchanged_stale_heartbeat"},
        precondition_cids=(incident.incident_cid, exhaustion.receipt_cid),
        expected_effects=("lane_quarantined",),
        success_test="health reports lane quarantined",
        stop_condition="stop after the first observed expected effect",
    )
    plan = RescuePlan(
        incident_cid=incident.incident_cid,
        exhaustion_receipt_cid=exhaustion.receipt_cid,
        repository_root_cid=incident.repository_root_cid,
        run_cid=incident.run_cid,
        policy_root=incident.policy_root,
        actions=(action,),
        rationale_reference_cids=(incident.evidence_cids[0],),
        unresolved_risks=("Independent work may require reassignment.",),
        max_actions=2,
    )
    assert RescuePlan.from_json(plan.to_json()).rescue_plan_cid == plan.rescue_plan_cid

    with pytest.raises(PromptWorkflowContractError, match="one of"):
        replace(action, operation="run_shell")
    with pytest.raises(RescuePlanError, match="forbidden"):
        replace(action, parameters={"shell_command": "rm -rf /tmp/example"})
    with pytest.raises(RescuePlanError, match="forbidden"):
        replace(action, parameters={"output_path": "/tmp/new-location"})
    with pytest.raises(PromptSecretError):
        replace(action, parameters={"password": "not-allowed"})
    with pytest.raises(RescuePlanError, match="over budget"):
        replace(plan, max_actions=1, actions=(action, action))


def test_rescue_action_order_is_semantic_but_set_like_order_is_not() -> None:
    incident = _incident()
    first = RescueAction(
        operation=RescueOperation.HEALTH,
        target_id="lane:implementation",
        parameters={},
        precondition_cids=(_cid("precondition-a"),),
        expected_effects=("health_observed",),
        success_test="health is current",
        stop_condition="stop if healthy",
    )
    second = replace(
        first,
        operation=RescueOperation.QUARANTINE,
        precondition_cids=(_cid("precondition-b"),),
        expected_effects=("lane_quarantined",),
        success_test="lane is quarantined",
        stop_condition="stop after quarantine",
    )
    values = {
        "incident_cid": incident.incident_cid,
        "exhaustion_receipt_cid": _cid("exhaustion"),
        "repository_root_cid": incident.repository_root_cid,
        "run_cid": incident.run_cid,
        "policy_root": incident.policy_root,
        "rationale_reference_cids": (_cid("z"), _cid("a")),
        "unresolved_risks": ("risk:z", "risk:a"),
        "max_actions": 2,
    }
    plan = RescuePlan(actions=(first, second), **values)
    same_sets = RescuePlan(
        actions=(first, second),
        **{
            **values,
            "rationale_reference_cids": tuple(
                reversed(values["rationale_reference_cids"])
            ),
            "unresolved_risks": tuple(reversed(values["unresolved_risks"])),
        },
    )
    reordered_actions = RescuePlan(actions=(second, first), **values)
    assert same_sets.rescue_plan_cid == plan.rescue_plan_cid
    assert reordered_actions.rescue_plan_cid != plan.rescue_plan_cid
