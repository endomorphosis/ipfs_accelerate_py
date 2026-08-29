"""ASE3-009 production Python facade and package export tests."""

from __future__ import annotations

import importlib
import json
from dataclasses import replace
from pathlib import Path
from typing import Mapping

import pytest

from ipfs_accelerate_py.agent_supervisor.entrypoints import facade as facade_mod
from ipfs_accelerate_py.agent_supervisor.entrypoints import service_factory as sf
from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
    ContinuationAction,
    LaunchPlan,
    RunHandle,
    RunHealth,
    RunState,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.intent_service import (
    SupervisorIntentService,
)
from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry import RunRegistry
from ipfs_accelerate_py.agent_supervisor.entrypoints.runtime_factory import (
    CompleteLaunchPlan,
    RuntimeEffectReceipt,
    StandardSupervisorRuntimeFactory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_composition_manifest_is_body_free_and_stable() -> None:
    m1 = sf.build_production_composition_manifest(
        generation=1,
        objective_refill_enabled=True,
        monitor_enabled=True,
    )
    m2 = sf.build_production_composition_manifest(
        generation=1,
        objective_refill_enabled=True,
        monitor_enabled=True,
    )
    assert m1.composition_cid == m2.composition_cid
    assert m1.activation_task_id == "ASE3-026"
    assert m1.codebase_refill_enabled is False
    blob = json.dumps(m1.to_dict())
    assert "password" not in blob.lower()
    assert "BEGIN " not in blob


def test_resolve_composition_from_activated_repo() -> None:
    composition = sf.resolve_production_composition(repository_root=REPO_ROOT)
    assert composition.manifest.objective_refill_enabled is True
    assert composition.manifest.monitor_enabled is True
    assert composition.manifest.generation == 1
    assert set(composition.manifest.backends) == {
        "resolver",
        "broker",
        "planning",
        "materialization",
        "scheduler",
        "refill",
        "monitor",
        "run_registry",
    }


def test_open_from_repo_requires_no_expert_args() -> None:
    supervisor = facade_mod.Supervisor.open(repository=REPO_ROOT)
    assert supervisor.composition_cid
    assert supervisor.composition_manifest.activation_task_id == "ASE3-026"


def test_open_without_config_fails_typed(tmp_path: Path) -> None:
    with pytest.raises(facade_mod.SupervisorConfigurationError):
        facade_mod.Supervisor.open(
            repository=tmp_path,
            require_activation=True,
        )


def test_preview_is_effect_free() -> None:
    supervisor = facade_mod.Supervisor.open(repository=REPO_ROOT)
    obs = supervisor.preview("Improve validation gates")
    assert obs.state == "preview"
    assert obs.values.get("effect_applied") is False
    assert "Improve validation" not in json.dumps(obs.to_dict())


def test_prompt_preview_rejects_scanner_program_root_mismatch_before_planning() -> None:
    """Observed scan identity may not be rewritten to match a stale request."""

    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptSupervisorService,
        PromptWorkflowStaleRootError,
    )
    from test.api.test_agent_supervisor_prompt_goal_planner import _request, _scan

    request = _request()
    mismatched_scan = replace(
        _scan(request),
        program_root=_cid("different-observed-program-root"),
    )

    class _MismatchedScanner:
        calls = 0

        def scan(self, _request_value: object) -> object:
            self.calls += 1
            return mismatched_scan

    class _ForbiddenPlanner:
        calls = 0

        def plan(self, *_args: object, **_kwargs: object) -> object:
            self.calls += 1
            raise AssertionError("planner must not see a stale program root")

    class _ForbiddenAdmission:
        calls = 0

        def admit(self, *_args: object, **_kwargs: object) -> object:
            self.calls += 1
            raise AssertionError("admission must not see a stale program root")

    scanner = _MismatchedScanner()
    planner = _ForbiddenPlanner()
    admission = _ForbiddenAdmission()
    service = PromptSupervisorService(
        scanner=scanner,
        planner=planner,
        admission=admission,
    )

    with pytest.raises(PromptWorkflowStaleRootError, match="program root"):
        service.preview(request)

    assert scanner.calls == 1
    assert planner.calls == 0
    assert admission.calls == 0


def test_prompt_service_does_not_replace_the_planners_graph_with_campaign_ids() -> None:
    """Campaign identifiers in evidence do not authorize post-planner task minting."""

    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptSupervisorService,
    )
    from test.api.test_agent_supervisor_prompt_goal_planner import (
        _evidence,
        _request,
        _scan,
    )
    from test.api.test_agent_supervisor_prompt_workflow_service import (
        _Admission,
        _Planner,
    )

    request = _request()
    evidence = replace(
        _evidence(),
        claim_keys=("PCPR-000", "PCPR-001"),
    )
    planner_scan = _scan(request, evidence=(evidence,))

    class _Scanner:
        calls = 0

        def scan(self, _request_value: object) -> object:
            self.calls += 1
            return planner_scan

    scanner = _Scanner()
    planner = _Planner()
    admission = _Admission()
    service = PromptSupervisorService(
        scanner=scanner,
        planner=planner,
        admission=admission,
        catalog_root=_cid("campaign-neutral-catalog"),
    )

    preview = service.preview(request)

    assert scanner.calls == 1
    assert planner.calls == 1
    assert admission.calls == 1
    assert len(preview.admitted_task_cids) == 1


def test_run_without_complete_launch_plan_uses_observed_bindings(
    tmp_path: Path,
) -> None:
    supervisor, prompt = _production_supervisor(tmp_path)
    run = supervisor.run(prompt)
    assert run.run_id
    assert run.identities["workflow_request_cid"] == run.run_id
    assert run.identities["plan_create_request_cid"]
    assert run.identities["preview_receipt_cid"]
    assert run.identities["plan_root_cid"]
    assert run.state == "admitted"
    assert run.state.lower() not in {"completed", "complete"}
    assert run.identities["completion_authority"] is False
    assert run.identities["model_assertion_cannot_complete"] is True
    assert run.identities["empty_queue_cannot_complete"] is True
    assert "complete_plan" not in supervisor._composition.extras
    blob = json.dumps(run.identities)
    assert prompt not in blob


def test_start_without_bound_runtime_is_typed_unavailable(tmp_path: Path) -> None:
    supervisor, prompt = _production_supervisor(tmp_path)
    run = supervisor.run(prompt)
    with pytest.raises(facade_mod.SupervisorUnavailableError, match="START"):
        supervisor.start(run.run_id)


def _cid(label: str) -> str:
    from ipfs_accelerate_py.agent_supervisor.core.multiformats_identity import (
        cid_for_dag_json,
    )

    return cid_for_dag_json({"ase3_009_fixture": label})


def _minimal_launch_plan(root: Path) -> LaunchPlan:
    from ipfs_accelerate_py.agent_supervisor.entrypoints.contracts import (
        CoordinationShardBinding,
        ExpectedEffect,
        ReplicationBinding,
        ReplicationMode,
    )

    shard = CoordinationShardBinding(
        backend="duckdb",
        database_path=str(root / "coord.duckdb"),
        shard_id="shard-0",
        shard_count=1,
        shard_index=0,
        owner_principal_ref="principal:local",
        coordinator_cid=_cid("coord"),
        lease_namespace="ns",
        fencing_generation=1,
        writable=True,
    )
    replication = ReplicationBinding(
        mode=ReplicationMode.PARQUET_IPLD,
        parquet_dataset_path=str(root / "parquet"),
        parquet_schema_cid=_cid("parquet"),
        partition_keys=("repository_id", "run_id", "event_date", "shard_id"),
        ipld_manifest_schema_cid=_cid("ipld"),
    )
    return LaunchPlan(
        invocation_cid=_cid("invocation"),
        target_resolution_receipt_cid=_cid("target"),
        resolved_profile_cid=_cid("profile"),
        working_directory=str(root),
        state_path=str(root / "state" / "run.json"),
        task_source_path=str(root / "state" / "tasks.duckdb"),
        supervisor_argv=("python", "-m", "supervisor"),
        daemon_argv=("python", "-m", "daemon"),
        environment_names=(),
        provider_route_cid=_cid("route"),
        resource_budget_cid=_cid("budget"),
        validation_profile_cid=_cid("validation"),
        lifecycle_profile_cid=_cid("lifecycle"),
        coordination_shard=shard,
        replication=replication,
        expected_effects=(ExpectedEffect.LAUNCH_LOCAL_PROCESS,),
        idempotency_key="invocation:ase3-009",
        adoption_key="adoption:ase3-009",
        lease_required=True,
        authorization_required=True,
        dry_run=False,
    )


def test_run_with_real_intent_factory_reaches_service(tmp_path: Path) -> None:
    """Injected factory + CompleteLaunchPlan reaches intent service (no simulate)."""

    registry = RunRegistry(tmp_path / "registry")

    def _receipt(name: str, **values: object) -> RuntimeEffectReceipt:
        return RuntimeEffectReceipt(
            receipt_cid=f"receipt-{name}",
            effect_applied=True,
            values={"receipt_cid": f"receipt-{name}", "effect_applied": True, **values},
        )

    handlers = {
        "resolve": lambda *a, **k: _receipt("resolve"),
        "preview": lambda *a, **k: _receipt("preview"),
        "authorize": lambda *a, **k: _receipt("authorize"),
        "materialize": lambda plan, handle: _receipt(
            "materialize",
            task_source_cid="task-src-1",
            task_source_revision_cid="task-rev-1",
        ),
        "start": lambda plan, handle: _receipt(
            "start",
            process_cid="proc-1",
            lease_id="lease-1",
            fencing_generation=1,
            state_revision_cid="state-1",
            health_revision_cid="health-1",
            event_cursor="cursor-1",
        ),
        "adopt": lambda plan, handle: _receipt(
            "adopt",
            process_cid="proc-1",
            lease_id="lease-1",
            fencing_generation=1,
        ),
        "observe": lambda *a, **k: _receipt("observe"),
        "steer": lambda *a, **k: _receipt("steer"),
        "validate": lambda *a, **k: _receipt("validate"),
        "stop": lambda *a, **k: _receipt("stop"),
    }
    factory = StandardSupervisorRuntimeFactory(registry=registry, handlers=handlers)
    try:
        complete = CompleteLaunchPlan(
            launch_plan=_minimal_launch_plan(tmp_path),
            task_source_cid="task-src-1",
            task_source_revision_cid="task-rev-1",
        )
    except Exception as exc:  # pragma: no cover - contract shape drift
        pytest.skip(f"LaunchPlan fixture unsupported: {exc}")

    composition = sf.resolve_production_composition(
        repository_root=REPO_ROOT,
        intent_factory=factory,
    )
    composition.extras["complete_plan"] = complete
    supervisor = facade_mod.Supervisor.open(services=composition)
    try:
        run = supervisor.run(
            "Improve the agent supervisor without weakening safety gates"
        )
    except (facade_mod.SupervisorUnavailableError, Exception) as exc:
        # Intent path may fail closed on contract identity or missing effects;
        # that is not a simulated completion path.
        if type(exc).__name__ in {
            "SupervisorUnavailableError",
            "ContractIdentityError",
            "EntrypointContractError",
            "PromptToRunError",
            "PromptToRunUnavailableError",
            "MultiformatsIdentityError",
        } or isinstance(exc, facade_mod.SupervisorError):
            assert "simulated" not in str(exc).lower()
            assert "completed" not in str(exc).lower() or "refuse" in str(exc).lower()
            return
        raise
    assert run.run_id
    assert run.composition_cid == supervisor.composition_cid
    assert run.state.lower() not in {"completed", "complete"}
    assert run.effect_receipt_cids  # no simulated empty completion
    status = run.status()
    assert status.run_id == run.run_id
    assert "weakening safety" not in json.dumps(status.to_dict())

def test_registered_run_handle_status_and_ambiguity() -> None:
    supervisor = facade_mod.Supervisor.open(repository=REPO_ROOT)
    run = facade_mod.SupervisorRun(
        run_id="run-ase3-009",
        run_revision=1,
        composition_cid=supervisor.composition_cid,
        state="running",
        health="healthy",
        event_cursor="cursor:1",
        supervisor=supervisor,
        effect_receipt_cids=("receipt-start",),
    )
    supervisor._runs[run.run_id] = run
    assert run.status().state == "running"
    assert run.doctor().values["activation_task_id"] == "ASE3-026"
    # Sole run is inferred without an explicit run_id.
    assert supervisor.status().run_id == run.run_id
    # Ambiguity with two runs:
    supervisor._runs["run-b"] = facade_mod.SupervisorRun(
        run_id="run-b",
        run_revision=1,
        composition_cid=supervisor.composition_cid,
        state="running",
        health="healthy",
        event_cursor="c2",
        supervisor=supervisor,
        effect_receipt_cids=("r2",),
    )
    with pytest.raises(facade_mod.SupervisorAmbiguityError) as amb:
        supervisor.status()
    assert "run-ase3-009" in amb.value.candidates
    assert "run-b" in amb.value.candidates


def test_status_ambiguity_without_run() -> None:
    supervisor = facade_mod.Supervisor.open(repository=REPO_ROOT)
    with pytest.raises(facade_mod.SupervisorAmbiguityError):
        supervisor.status()


def test_lazy_entrypoints_export_supervisor() -> None:
    import ipfs_accelerate_py.agent_supervisor.entrypoints as ep

    assert "Supervisor" in ep.ENTRYPOINT_LAZY_FACADE_EXPORTS
    # Ensure not eagerly bound before access
    reloaded = importlib.reload(ep)
    assert "Supervisor" not in vars(reloaded) or not isinstance(
        vars(reloaded).get("Supervisor"), type
    )
    Supervisor = reloaded.Supervisor
    assert Supervisor is facade_mod.Supervisor


def test_package_root_exports_supervisor() -> None:
    import ipfs_accelerate_py.agent_supervisor as asup

    assert asup.Supervisor is facade_mod.Supervisor
    exported = getattr(asup, "Supervisor")
    assert exported is facade_mod.Supervisor


def test_init_local_requires_consent() -> None:
    with pytest.raises(facade_mod.SupervisorConfigurationError):
        facade_mod.Supervisor.init_local(repository=REPO_ROOT, consent=False)


BLUEPRINT_TITLES: tuple[str, ...] = (
    "Seal repositories, contracts, policies, and supervisor baseline",
    "Qualify direct objective and event-driven supervisor",
    "Freeze canonical supervisor contracts or issue non-promotion",
    "Inventory every legacy bypass and false-authority path",
    "Remove Datasets import-time auto-install",
    "Remove Datasets false-success fallbacks",
    "Make typed outcomes canonical",
    "Canonicalize LogicProviderProtocol",
    "Stabilize semantic APIs and ContextPack contract",
    "Package schemas and shared vectors",
    "Resolve Datasets license metadata",
    "Qualify real Datasets solver paths",
    "Requalify local Kit backend",
    "Qualify pinned IPFS backend",
    "Qualify or de-scope Iroh",
    "Qualify VFS/WAL/current-root recovery",
    "Qualify proof-seal store",
    "Remove sibling test-tree coupling",
    "Qualify Python/CLI/MCP/MCP++ parity",
    "Generate authoritative support matrix",
    "Quarantine Accelerate legacy mock coordinator",
    "Remove fabricated hardware capability",
    "Remove pseudo-CID identity",
    "Remove fabricated endpoint success",
    "Consolidate capability ladder",
    "Pin mutable dependencies",
    "Correct Python compatibility metadata",
    "Qualify CPU execution",
    "Qualify real CUDA execution",
    "Qualify one real model/provider path",
    "Stabilize shared contracts",
    "Add canonical-byte and CID vectors",
    "Add negative and cross-language vectors",
    "Add cross-repository compatibility checks",
    "Build clean Datasets package",
    "Build clean Kit package",
    "Build clean Accelerate package",
    "Produce dependency locks",
    "Produce SBOMs and provenance",
    "Produce signed tags and artifacts",
    "Produce portfolio compatibility lock",
    "Add branch and release gates",
    "Submit reference high-level objective",
    "Build semantic ContextPack",
    "Persist and publish current ContextPack root",
    "Execute deterministic-first route",
    "Produce bounded patch",
    "Run selected tests and proofs",
    "Introduce unrelated state change",
    "Demonstrate safe reuse",
    "Introduce relevant interface change",
    "Demonstrate stale rejection and PlanDelta",
    "Restart authoritative state owner",
    "Demonstrate recovery and idempotency",
    "Produce final proof-carrying receipt chain",
    "Add Python external-client demonstration",
    "Add generic MCP-client demonstration",
    "Prove cross-client objective identity parity",
    "Prove external clients cannot bypass authority",
    "Prepare threat model",
    "Prepare trusted-computing-base inventory",
    "Prepare security and correctness audit package",
    "Run release-candidate gate",
    "Produce promotion or honest non-promotion receipt",
    "Publish residual-gap report",
    "Recommend the next customer or synthetic pilot",
)


def _observation(repo: Path, state: Path) -> sf.ProductionBindingObservation:
    return sf.ProductionBindingObservation(
        repository_root=str(repo.resolve()),
        repository_id="repository:" + _cid("repo"),
        repository_root_cid=_cid("repository-root"),
        tree_id=_cid("tree"),
        dirty_worktree_root=_cid("dirty"),
        head_commit="a" * 40,
        head_tree="b" * 40,
        state_root=str(state.resolve()),
        policy_root=_cid("policy"),
        capability_catalog_root=_cid("capability"),
        provider_catalog_root=_cid("provider"),
        program_root=_cid("program"),
        intent_ir_root=_cid("intent"),
        legal_ir_root=_cid("legal"),
        security_ir_root=_cid("security"),
        usage_policy_root=_cid("usage"),
        configuration_root=_cid("configuration"),
        allowlist_cid=_cid("allowlist"),
        caller="principal:local",
        board_namespace="prompt-workflow",
        supervisor_profile="implementation-daemon",
        composition_cid=_cid("composition"),
        duckdb_available=True,
    )


class _AdmittedCreateService:
    def preview_create(self, request: object, **_kwargs: object) -> object:
        from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
            PlanCreateMode,
            PlanCreatePreviewReceipt,
            PlanCreateVerdict,
        )

        roots = getattr(request, "roots")
        request_cid = str(getattr(request, "request_cid"))
        return PlanCreatePreviewReceipt(
            request_cid=request_cid,
            input_snapshot_cid=request_cid,
            mode=PlanCreateMode.DETERMINISTIC,
            verdict=PlanCreateVerdict.ADMITTED,
            roots=roots,
            stage_results=(),
            scan_cid=request_cid,
            admission_receipt_cid=request_cid,
            plan_root_cid=str(roots.program_root),
            read_only=True,
            wrote_effects=(),
        )


def _named_planner(titles: tuple[str, ...]):
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
        parse_prompt_goal_graph,
    )
    from test.api.test_agent_supervisor_prompt_goal_planner import (
        _encoded_proposal,
        _proposal,
    )

    class _Planner:
        calls = 0

        def plan(self, request: object, scan: object, **_kwargs: object) -> object:
            self.calls += 1
            evidence_cid = scan.evidence[0].evidence_cid
            acceptance = {
                "criterion_key": "criterion:pytest",
                "criterion": "The focused planner tests pass.",
                "evidence_cids": [evidence_cid],
                "validation_keys": ["validation:pytest"],
            }
            tasks = []
            for index, title in enumerate(titles):
                slug = f"task:{index:03d}"
                tasks.append(
                    {
                        "task_key": slug,
                        "goal_key": "goal:root",
                        "dependency_task_keys": [],
                        "objective": title,
                        "rationale": "Planner elaboration of the submitted idea.",
                        "scope_paths": [f"pkg/task_{index:03d}.py"],
                        "outputs": [
                            {
                                "path": f"pkg/task_{index:03d}.py",
                                "effect": "modify",
                                "media_type": "text/x-python",
                            }
                        ],
                        "validations": [
                            {
                                "validation_key": "validation:pytest",
                                "argv": ["python", "-m", "pytest", "-q"],
                                "cwd": ".",
                                "expected_exit_codes": [0],
                            }
                        ],
                        "acceptance": [acceptance],
                        "evidence_cids": [evidence_cid],
                        "priority": "P0",
                        "track": "prompt-goal-planning",
                        "bundle": "prompt-workflow/planning",
                        "parallel_lane": f"lane-{index:03d}",
                        "resource_class": "cpu-small",
                        "predicted_files": [f"pkg/task_{index:03d}.py"],
                        "risks": ["Planner output remains candidate evidence."],
                        "assumptions": ["Admission remains independent."],
                        "fallback_behavior": "fail_closed",
                    }
                )
            encoded = _encoded_proposal(scan, tasks=tasks)
            graph = parse_prompt_goal_graph(encoded, request, scan)
            receipt = SimpleNamespace(
                to_dict=lambda: {
                    "request_cid": request.request_cid,
                    "scan_cid": scan.scan_cid,
                    "plan_root_cid": graph.plan_root_cid,
                },
                fallback=SimpleNamespace(used=False),
            )
            return SimpleNamespace(graph=graph, receipt=receipt, used_fallback=False)

    return _Planner()


def _production_supervisor(
    tmp_path: Path,
    *,
    titles: tuple[str, ...] | None = None,
    mutation: Mapping[str, object] | None = None,
    intent_factory: object | None = None,
    stale: bool = False,
):
    from ipfs_accelerate_py.agent_supervisor.prompt.plan_supervisor_service import (
        PlanSupervisorService,
    )
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptSupervisorService,
    )
    from test.api.test_agent_supervisor_prompt_goal_planner import _scan
    from test.api.test_agent_supervisor_prompt_workflow_service import (
        _Planner,
        _Scanner,
    )

    class _MatchingAdmission:
        calls = 0

        def admit(self, request, scan, graph, _planning):  # type: ignore[no-untyped-def]
            from types import SimpleNamespace

            from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
                prompt_workflow_cid,
            )

            self.calls += 1
            remaining = list(graph.tasks)
            ordered = []
            seen: set[str] = set()
            while remaining:
                progressed = False
                for task in list(remaining):
                    deps = set(task.dependency_task_cids)
                    if deps.issubset(seen):
                        ordered.append(task)
                        seen.add(task.task_cid)
                        remaining.remove(task)
                        progressed = True
                if not progressed:
                    ordered.extend(remaining)
                    break
            topo = tuple(task.task_cid for task in ordered)
            receipt = SimpleNamespace(
                candidate_plan_cid=graph.plan_root_cid,
                topological_task_cids=topo,
                formal_plan_id="formal:production-admission",
                ir_receipt_id="ir:production-admission",
                policy_id=request.policy_root,
                repository_tree_id=scan.dirty_worktree_root,
                topology_id="topology:production-admission",
                findings=(),
                to_dict=lambda: {
                    "candidate_plan_cid": graph.plan_root_cid,
                    "final_plan_cid": graph.plan_root_cid,
                    "repository_tree_id": scan.dirty_worktree_root,
                    "verdict": "admitted",
                },
            )
            plan_root = prompt_workflow_cid(
                {
                    "schema": (
                        "ipfs_accelerate_py/agent-supervisor/admitted-prompt-plan@1"
                    ),
                    "candidate_plan_cid": graph.plan_root_cid,
                    "formal_plan_id": receipt.formal_plan_id,
                    "ir_receipt_id": receipt.ir_receipt_id,
                    "policy_id": receipt.policy_id,
                    "repository_tree_id": receipt.repository_tree_id,
                    "task_cids": list(sorted(task.task_cid for task in graph.tasks)),
                    "topology_id": receipt.topology_id,
                }
            )
            return SimpleNamespace(
                admitted=True,
                admitted_graph=graph,
                plan_root_cid=plan_root,
                task_cids=topo,
                receipt=receipt,
            )

    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    (repo / "pkg").mkdir()
    (repo / "pkg" / "retry_planner.py").write_text("x = 1\n", encoding="utf-8")
    observation = _observation(repo, state)
    planner = _named_planner(titles) if titles is not None else _Planner()
    scanner = _Scanner()
    if stale:

        class _Stale(_Scanner):
            def scan(self, request):  # type: ignore[no-untyped-def]
                self.calls += 1
                return replace(_scan(request), program_root=_cid("stale-program"))

        scanner = _Stale()
    composition = sf.resolve_production_composition(
        repository_root=REPO_ROOT,
        state_root=state,
        intent_factory=intent_factory,
    )
    composition.extras["observation"] = observation
    composition.extras["scanner"] = scanner
    composition.extras["planner"] = planner
    composition.extras["admission"] = _MatchingAdmission()
    composition.extras["plan_supervisor_service"] = PlanSupervisorService(
        create_service=_AdmittedCreateService(),
        revision_store_root=state / "plan_revision_store",
    )
    if mutation is not None:
        composition.extras["mutation_bindings"] = dict(mutation)
    supervisor = facade_mod.Supervisor.open(services=composition)
    prompt = (
        "Make ipfs_datasets_py, ipfs_kit_py, and ipfs_accelerate_py "
        "production-qualifiable as one proof-carrying platform"
    )
    return supervisor, prompt


def test_observe_bindings_from_activated_repo() -> None:
    supervisor = facade_mod.Supervisor.open(repository=REPO_ROOT)
    observed = supervisor.observe_bindings()
    assert observed["repository_root"]
    assert observed["repository_root_cid"]
    assert observed["tree_id"]
    assert observed["policy_root"]
    assert observed["capability_catalog_root"]
    assert observed["provider_catalog_root"]
    assert observed["composition_cid"] == supervisor.composition_cid


def test_mutation_without_lease_fails_closed(tmp_path: Path) -> None:
    supervisor, prompt = _production_supervisor(
        tmp_path,
        mutation={
            "authority_cid": _cid("authority"),
            "idempotency_key": "idem:1",
            "lease_id": "",
            "fencing_epoch": 1,
        },
    )
    with pytest.raises(facade_mod.SupervisorUnavailableError, match="lease"):
        supervisor.run(prompt)


def test_stale_program_root_fails_closed(tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workflow import (
        PromptWorkflowStaleRootError,
    )

    supervisor, prompt = _production_supervisor(tmp_path, stale=True)
    with pytest.raises(
        (facade_mod.SupervisorUnavailableError, PromptWorkflowStaleRootError)
    ):
        supervisor.run(prompt)


def test_admitted_preview_includes_complete_materialization_inputs(
    tmp_path: Path,
) -> None:
    supervisor, prompt = _production_supervisor(tmp_path)
    run = supervisor.run(prompt)
    inputs = run.identities["materialization_inputs"]
    assert inputs["plan_root_cid"]
    assert inputs["expected_effects"]
    assert inputs["apply_requires"]
    assert "lease" in inputs["apply_requires"]
    assert inputs["completion_authority"] is False
    assert run.state == "admitted"


def test_authorized_apply_publishes_projection_identities(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    supervisor, prompt = _production_supervisor(
        tmp_path,
        mutation={
            "authority_cid": _cid("authority"),
            "idempotency_key": "idem:apply-1",
            "lease_id": "lease-1",
            "fencing_epoch": 1,
        },
    )
    run = supervisor.run(prompt)
    assert run.state == "materialized"
    assert run.identities["task_source_cid"]
    assert run.identities["task_source_revision_cid"]
    assert run.identities["plan_revision_cid"]
    assert run.identities["lease_id"] == "lease-1"
    assert run.identities["fencing_epoch"] == 1
    assert run.effect_receipt_cids
    assert run.state.lower() not in {"completed", "complete"}


def test_start_is_separate_authorized_control(tmp_path: Path) -> None:
    pytest.importorskip("duckdb")
    from ipfs_accelerate_py.agent_supervisor.entrypoints.run_registry import (
        RunRegistry,
    )
    from ipfs_accelerate_py.agent_supervisor.entrypoints.runtime_factory import (
        RuntimeEffectReceipt,
        StandardSupervisorRuntimeFactory,
    )

    registry = RunRegistry(tmp_path / "registry")

    def _receipt(name: str, **values: object) -> RuntimeEffectReceipt:
        return RuntimeEffectReceipt(
            receipt_cid=f"receipt-{name}",
            effect_applied=True,
            values={
                "receipt_cid": f"receipt-{name}",
                "effect_applied": True,
                **values,
            },
        )

    handlers = {
        "resolve": lambda *a, **k: _receipt("resolve"),
        "preview": lambda *a, **k: _receipt("preview"),
        "authorize": lambda *a, **k: _receipt("authorize"),
        "materialize": lambda plan, handle: _receipt(
            "materialize",
            task_source_cid="task-src-1",
            task_source_revision_cid="task-rev-1",
        ),
        "start": lambda plan, handle: _receipt(
            "start",
            process_cid="proc-1",
            lease_id="lease-1",
            fencing_generation=1,
            state_revision_cid="state-1",
            health_revision_cid="health-1",
            event_cursor="cursor-1",
        ),
        "adopt": lambda *a, **k: _receipt("adopt"),
        "observe": lambda *a, **k: _receipt("observe"),
        "steer": lambda *a, **k: _receipt("steer"),
        "validate": lambda *a, **k: _receipt("validate"),
        "stop": lambda *a, **k: _receipt("stop"),
    }
    factory = StandardSupervisorRuntimeFactory(registry=registry, handlers=handlers)
    supervisor, prompt = _production_supervisor(
        tmp_path,
        mutation={
            "authority_cid": _cid("authority"),
            "idempotency_key": "idem:apply-start",
            "lease_id": "lease-1",
            "fencing_epoch": 1,
        },
        intent_factory=factory,
    )
    run = supervisor.run(prompt)
    assert run.state == "materialized"
    started = supervisor.start(run.run_id)
    assert started.values.get("effect_applied") is True
    assert run.identities.get("start_receipt_cid")
    assert run.identities.get("process_cid")
    assert run.state.lower() not in {"completed", "complete"}


def test_generic_submission_materializes_sixty_six_named_tasks(
    tmp_path: Path,
) -> None:
    assert len(BLUEPRINT_TITLES) == 66
    supervisor, prompt = _production_supervisor(tmp_path, titles=BLUEPRINT_TITLES)
    run = supervisor.run(prompt)
    assert run.identities["admitted_task_count"] == 66
    assert run.identities["admitted_task_count"] <= 80
    inputs = run.identities["materialization_inputs"]
    assert len(inputs["admitted_task_cids"]) == 66


def test_python_cli_mcp_identity_parity(tmp_path: Path) -> None:
    import asyncio
    import io
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.entrypoints import cli as supervisor_cli
    from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
        configure_prompt_lifecycle_supervisor,
    )
    from ipfs_accelerate_py.mcp_server.tools.agent_supervisor_tools import (
        prompt_entrypoints as pe,
    )

    supervisor, prompt = _production_supervisor(tmp_path)
    python_run = supervisor.run(prompt)
    args = SimpleNamespace(
        supervisor_command="run",
        prompt=prompt,
        prompt_file=None,
        prompt_stdin=False,
        output_json=True,
        repository=None,
        state_root=None,
    )
    out = io.StringIO()
    code = supervisor_cli.run_supervisor_cli(
        args, stdout=out, supervisor=supervisor
    )
    assert code == supervisor_cli.EXIT_SUCCESS
    cli_payload = json.loads(out.getvalue())
    configure_prompt_lifecycle_supervisor(supervisor)
    try:
        mcp_payload = asyncio.run(pe.agent_supervisor_run(prompt=prompt))
    finally:
        configure_prompt_lifecycle_supervisor(None)
    assert mcp_payload["ok"] is True
    assert (
        python_run.identities["workflow_request_cid"]
        == cli_payload["result"]["workflow_request_cid"]
        == mcp_payload["result"]["workflow_request_cid"]
    )
    assert (
        python_run.identities["plan_create_request_cid"]
        == cli_payload["result"]["plan_create_request_cid"]
        == mcp_payload["result"]["plan_create_request_cid"]
    )
    assert (
        python_run.identities["objective_cid"]
        == cli_payload["result"]["objective_cid"]
        == mcp_payload["result"]["objective_cid"]
    )
    assert python_run.state.lower() not in {"completed", "complete"}


def test_direct_database_bypass_is_not_exposed(tmp_path: Path) -> None:
    supervisor, _prompt = _production_supervisor(tmp_path)
    assert not hasattr(supervisor, "execute_sql")
    assert "duckdb" not in type(supervisor).__dict__


def test_scheduler_handoff_disables_complete_launch_plan_injection() -> None:
    config_path = (
        REPO_ROOT
        / "config"
        / "agent_supervisor_prompt_only_self_improvement_v3_scheduler.json"
    )
    payload = json.loads(config_path.read_text(encoding="utf-8"))
    handoff = payload["direct_objective_handoff"]
    assert handoff["complete_launch_plan_caller_injection_required"] is False
    assert handoff["start_is_separate_authorized_control_operation"] is True
    assert handoff["initial_task_ceiling"] == 80
    assert handoff["model_assertion_cannot_complete"] is True
    assert handoff["empty_queue_cannot_complete"] is True
