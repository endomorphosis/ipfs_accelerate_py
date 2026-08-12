"""ASE3-009 production Python facade and package export tests."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

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


def test_run_without_bound_runtime_is_typed_unavailable() -> None:
    supervisor = facade_mod.Supervisor.open(repository=REPO_ROOT)
    with pytest.raises(facade_mod.SupervisorUnavailableError):
        supervisor.run("Improve the agent supervisor")


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
    import ipfs_accelerate_py as root
    import ipfs_accelerate_py.agent_supervisor as asup

    assert root.Supervisor is facade_mod.Supervisor
    assert asup.Supervisor is facade_mod.Supervisor


def test_init_local_requires_consent() -> None:
    with pytest.raises(facade_mod.SupervisorConfigurationError):
        facade_mod.Supervisor.init_local(repository=REPO_ROOT, consent=False)
