"""Genuine host proof-query custody over native finite execution reservations.

These cases invoke real source/query owners, Python/Lean observations, native
Quack task rows and completion checks. They launch no Docker worker and claim
no coding-child Popen enforcement; the broker has its own qualification.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import uuid

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_execution as execution
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured

from test.api.test_finite_integer_codebase import finite_tools  # noqa: F401
from test.api.test_finite_proof_query_worker_context import (
    _native_proof_query_worker_case as native_proof_query_execution_case,
)


@pytest.fixture(scope="module")
def proof_query_execution_case(tmp_path_factory, finite_tools):
    with native_proof_query_execution_case(
            tmp_path_factory.mktemp("finite-proof-query-native-execution") / "native", finite_tools) as case:
        yield case


@contextmanager
def reserve(case, output):
    with execution.reserve_finite_proof_query_execution(owner=case["owner"],
            admission=case["proof_query_admission"], verification_catalog=case["verification_catalog"],
            server=case["native"].server, source=case["native"].source,
            candidate=case["candidate"], worker_context=case["worker_context"], output=output,
            policy_observer=lambda request: request.roots) as scope:
        yield scope, scope._proof_query_closure


def test_genuine_reservation_signs_complete_query_native_plan_and_public_context(proof_query_execution_case, tmp_path):
    case = proof_query_execution_case
    with reserve(case, tmp_path / "complete") as (scope, closure):
        assert type(closure) is execution.FrozenFiniteProofQueryExecutionClosure
        payload = scope.to_dict()["payload"]
        assert payload["schema"] == execution.EXECUTION_SCHEMA and payload["profile"] == execution.PROFILE
        assert payload["proof_query_closure"] == closure.material_binding
        bound = closure.material_binding
        assert bound["proof_query_admission"] == case["proof_query_admission"]
        assert bound["proof_query_admission_cid"] == cid_for_structured(case["proof_query_admission"])
        indexed = bound["proof_query_admission"]["indexed_plan"]["proof_query_closure"]
        assert len(indexed["canonical_key_membership"]) == 5
        assert indexed["discovery_query"]["page"]["complete"] is True
        assert indexed["exact_query"]["page"]["complete"] is True
        assert indexed["domain_bridge"]["domain_inputs"] == [-2, -1, 0, 1, 2]
        assert bound["worker_context"] == case["worker_context"]
        assert bound["parent_proof_query_fence"] is True and bound["coding_child_proof_query_fence"] is False
        assert all(flag is False for flag in bound["authority"].values())
        assert len(payload["native_population"]["tasks"]) == 2
        assert all("finite_proof_query_admission_ref" in row["body"] for row in bound["native_plan_rows"])
        assert payload["candidate"]["worker_context"] == case["worker_context"]
        assert "--finite-proof-query-context" in payload["candidate"]["argv"]
        closure.require_detached(scope=scope)
        changed = deepcopy(closure.material_binding)
        changed["proof_query_admission"]["indexed_plan"]["proof_query_closure"]["authority"]["proof_authority"] = True
        assert closure.material_binding == bound
    assert scope.parent_lease.released


def test_exact_live_closure_refuses_serialized_or_foreign_scope(proof_query_execution_case, tmp_path):
    case = proof_query_execution_case
    with reserve(case, tmp_path / "live-only") as (scope, closure):
        with pytest.raises(execution.FiniteProofQueryExecutionError, match="exact same native"):
            closure.require_detached(scope=scope.to_dict())
        with pytest.raises(execution.FiniteProofQueryExecutionError, match="exact native finite candidate"):
            changed = json.loads(closure._base_candidate)
            changed["descriptor"]["task_revision"] += 1
            closure.extend_candidate_binding(changed)


@pytest.mark.parametrize("drift", ["proof_reference", "plan_status"])
def test_actual_native_plan_drift_refuses_even_though_old_task_rows_match(proof_query_execution_case, tmp_path, drift):
    case, server = proof_query_execution_case, proof_query_execution_case["native"].server
    with reserve(case, tmp_path / drift) as (scope, closure):
        row = closure.material_binding["native_plan_rows"][0]
        with server._lock:
            before = server._connection.execute(
                "SELECT status,body_json FROM plans WHERE plan_cid=?", [row["plan_cid"]]).fetchone()
            if drift == "proof_reference":
                changed = json.loads(before[1])
                changed["finite_proof_query_admission_ref"]["admission_cid"] = "changed-proof-query"
                server._connection.execute("UPDATE plans SET body_json=? WHERE plan_cid=?",
                    [json.dumps(changed), row["plan_cid"]])
            else:
                server._connection.execute("UPDATE plans SET status='changed-proof-query' WHERE plan_cid=?",
                                           [row["plan_cid"]])
        try:
            with pytest.raises(execution.FiniteProofQueryExecutionError, match="complete native proof-query plan rows"):
                closure.require_detached(scope=scope)
        finally:
            with server._lock:
                server._connection.execute("UPDATE plans SET status=?,body_json=? WHERE plan_cid=?",
                                           [before[0], before[1], row["plan_cid"]])
        closure.require_detached(scope=scope)


def test_physical_proof_cas_replacement_with_same_bytes_is_refused(proof_query_execution_case, tmp_path):
    case = proof_query_execution_case
    with reserve(case, tmp_path / "physical-proof") as (scope, closure):
        proof = closure.material_binding["proof_query_admission"]["indexed_plan"]["proof_query_closure"]
        path = case["index"].artifacts.path_for(proof["projection_cid"])
        backup = path.with_name(path.name + ".original-execution-test")
        raw, mode = path.read_bytes(), path.stat().st_mode & 0o777
        path.rename(backup)
        try:
            path.write_bytes(raw)
            path.chmod(mode)
            with pytest.raises(execution.FiniteProofQueryExecutionError, match="selected native admission, proof"):
                closure.require_detached(scope=scope)
        finally:
            path.unlink(missing_ok=True)
            backup.rename(path)


def test_genuine_catalog_epoch_rebuild_after_preparation_refuses(tmp_path, finite_tools):
    # A successor epoch remains changed; it cannot poison other positive cases.
    with native_proof_query_execution_case(tmp_path / "epoch-native", finite_tools) as case:
        with reserve(case, tmp_path / "real-epoch") as (scope, closure):
            before = closure.material_binding["proof_query_admission"]["indexed_plan"]["proof_query_closure"]
            old_epoch = before["discovery_query"]["page"]["epoch"]
            rebuilt = case["verification_catalog"].rebuild_current(case["repository"],
                expected_head=case["expected_head"], parent_lease=scope.parent_lease,
                cancel_event=scope._owner.cancel_event, timeout_seconds=scope._owner.timeout_seconds,
                memory_mb=scope._owner.memory_mb)
            assert rebuilt.epoch != old_epoch
            with pytest.raises(ValueError, match="proof-query inventory changed"):
                closure.require_detached(scope=scope)


@contextmanager
def native_refusal_runtime(case, scope, output, callback):
    """Host receiving-field harness with a genuine signed scope and run lease.

    The installed profile, proof closure, native lease coordinator, final
    scope fence, Linux snapshots and control journals are real. The lifecycle
    request authorization uses the existing explicit test request fixture;
    this harness does not qualify container launcher preparation or delivery.
    """
    from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation
    from ipfs_accelerate_py.agent_supervisor.control.control_plane import (
        JsonlControlStateStore, SupervisorControlService,
    )
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import (
        LifecycleOrchestrator, LinuxProcessAdapter,
    )
    from ipfs_accelerate_py.agent_supervisor.control.profile_authority import (
        load_local_profile, sign_profile_binding,
    )
    from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
    from ipfs_accelerate_py.agent_supervisor.merge.database_coordination import open_database_coordinator
    from test.api.test_agent_supervisor_lifecycle_orchestrator import _profile

    profile = _profile(output, argv=(sys.executable, "-c", "import time; time.sleep(60)"))
    runtime = AdmittedBenchmarkRuntime()
    runtime.repository, runtime.state = case["repository"], Path(profile.state_root)
    runtime.repository_id = case["manifest"]["payload"]["repository_cid"]
    runtime.admission = case["admission"]["local_admission"]
    runtime.finite_execution_scope, runtime.inventory_execution_scope = scope, None
    runtime.server, runtime.source = case["native"].server, case["native"].source
    runtime.local_profile = load_local_profile(repository_cid=runtime.repository_id,
        profile_dir=case["profile"], lifecycle_dir=case["lifecycle"])
    runtime.run_id = "host-proof-refusal-" + uuid.uuid4().hex
    runtime._children = []
    runtime.manifest = {"schema": "host-proof-query-no-birth-test-harness@1",
        "run_id": runtime.run_id, "finite_execution_scope": scope.material_binding,
        "production_activation": False, "completion_authority": False}
    runtime.manifest_id = cid_for_structured(runtime.manifest)
    runtime.signature = sign_profile_binding(profile_dir=case["profile"],
        lifecycle_dir=case["lifecycle"], payload=runtime.manifest)
    runtime.profile = replace(profile, profile_id="", target_id=runtime.repository_id, run_id=runtime.run_id,
        configuration_root=runtime.manifest_id, repository_root=str(runtime.repository),
        cwd=str(runtime.repository))
    runtime.coordinator = open_database_coordinator(runtime.state / "coordination.duckdb")
    runtime.lease = runtime.coordinator.acquire(lease_kind="resource", scope=runtime.run_id,
        owner_session_id=runtime.local_profile.identity_did, lease_ms=120_000,
        resource_kind="supervisor_run", resource_id=runtime.run_id,
        repository_id=runtime.repository_id, idempotency_key=runtime.manifest_id,
        body={"local_profile_id": runtime.local_profile.profile_id,
              "launch_grant": runtime.manifest_id})
    scope.bind_runtime(runtime)
    runtime.process = LinuxProcessAdapter(popen=lambda *args, **kwargs: callback(runtime, *args, **kwargs))
    runtime.orchestrator = LifecycleOrchestrator(state_root=runtime.state, profiles=(runtime.profile,),
        process_adapter=runtime.process, poll_interval_ms=5, stop_grace_ms=20)
    runtime.service = SupervisorControlService(repository_allowlist=(runtime.repository,),
        state_allowlist=(runtime.state,), handlers={operation: runtime.orchestrator for operation in (
            Operation.START, Operation.STOP)}, lease_validator=runtime._validate_lease,
        state_store=JsonlControlStateStore())
    try:
        yield runtime
    finally:
        assert runtime.process.snapshot(runtime.profile).members == ()
        runtime.coordinator.release(runtime.lease, expected_fencing_token=runtime.lease.fencing_token,
            expected_fence_epoch=runtime.lease.fence_epoch)
        runtime.coordinator.close()


def native_refusal_request(runtime, operation, key):
    from test.api.test_agent_supervisor_lifecycle_orchestrator import _request
    from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation

    request = _request(runtime.profile, operation=operation, key=key, deadline_ms=1_000,
        health_window_ms=0 if operation is Operation.STOP else 20,
        fence=runtime.lease.fence_epoch, lease_id=runtime.lease.lease_id)
    now = time.time_ns() // 1_000_000
    return replace(request, authorization=replace(request.authorization,
        evaluated_at_ms=now - 1, expires_at_ms=min(now + 60_000, runtime.lease.expires_at_ms)))


def test_genuine_proof_fence_refusal_has_native_no_effect_journal_and_stale_source_stop(
        proof_query_execution_case, tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.control import before_popen_refusal as refusal
    from ipfs_accelerate_py.agent_supervisor.control.control_contracts import Operation, OperationStatus
    from ipfs_accelerate_py.agent_supervisor.control.control_plane import MutationTransactionPhase
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import LifecycleSagaPhase

    case, source = proof_query_execution_case, proof_query_execution_case["repository"] / "calc.py"
    before, mode = source.read_bytes(), source.stat().st_mode & 0o777
    observed = {"actual_popen": 0, "native_refusal": None}
    actual_popen = subprocess.Popen
    def count_popen(*args, **kwargs):
        observed["actual_popen"] += 1
        return actual_popen(*args, **kwargs)

    with reserve(case, tmp_path / "real-native-refusal") as (scope, closure):
        def before_popen(runtime, *_args, **_kwargs):
            # This edit occurs after native lifecycle START preparation and
            # immediately before the genuine final detached scope fence.
            source.chmod(0o600)
            source.write_bytes(before + b"\n# native late proof-query source drift\n")
            source.chmod(mode)
            with runtime.server._lock:
                try:
                    scope.require_spawn_fence(runtime)
                except Exception as error:
                    observed["native_refusal"] = refusal.refuse_finite_proof_query_before_popen(runtime, error)
                    raise observed["native_refusal"] from error
            pytest.fail("changed source reached literal Popen")

        with native_refusal_runtime(case, scope, tmp_path / "host-refusal", before_popen) as runtime:
            # Earlier genuine source/prover setup precedes this birth counter.
            monkeypatch.setattr(subprocess, "Popen", count_popen)
            try:
                start = native_refusal_request(runtime, Operation.START, "native-proof:start-refused")
                result = runtime.service.execute(start)
                assert result.status is OperationStatus.CONFLICT and result.effects == ()
                assert observed["actual_popen"] == 0
                assert type(observed["native_refusal"]) is refusal.BeforePopenRefusal
                transaction = runtime.service.mutation_transaction(start)
                assert transaction.phase is MutationTransactionPhase.COMPENSATED
                assert transaction.applied_effect_ids == () and transaction.recovery_action.value == "none"
                boundary = result.data["before_popen_refusal"]["proof"]["boundary"]
                assert boundary["finite_scope_lease_id"] == scope.parent_lease.lease_id
                assert boundary["lease_id"] == runtime.lease.lease_id
                assert boundary["fencing_epoch"] == runtime.lease.fence_epoch
                failed = runtime.orchestrator.store.latest()[runtime.profile.target_id]
                assert failed.phase is LifecycleSagaPhase.FAILED and failed.phase.terminal
                assert failed.failure_code == "refused_before_popen_without_process_effect"
                assert failed.receipt is None and failed.observed_effects == ()
                assert runtime.process.snapshot(runtime.profile).members == () and not scope._spawned
                # STOP retains immutable ownership while the very same source
                # remains stale. It must not retry a proof/query observation.
                closure.require_runtime(scope=scope, runtime=runtime, stopping=True)
                with pytest.raises(ValueError,
                        match="finite source (?:or Git inventory|or captured evidence bytes) changed"):
                    closure.require_detached(scope=scope)
                stopped = runtime.service.execute(native_refusal_request(runtime,
                    Operation.STOP, "native-proof:stop-after-refusal"))
                assert stopped.status is OperationStatus.SUCCEEDED and stopped.data["old_tree_fenced"] is True
                assert runtime.orchestrator.store.latest()[runtime.profile.target_id].phase is LifecycleSagaPhase.COMMITTED
                assert runtime.process.snapshot(runtime.profile).members == () and observed["actual_popen"] == 0
            finally:
                source.chmod(0o600)
                source.write_bytes(before)
                source.chmod(mode)
    assert scope.parent_lease.released


def test_native_proof_refusal_factory_rejects_foreign_lease_and_already_spawned_scope(
        proof_query_execution_case, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.control import before_popen_refusal as refusal

    case = proof_query_execution_case
    with reserve(case, tmp_path / "native-refusal-binding") as (scope, _closure):
        with native_refusal_runtime(case, scope, tmp_path / "host-binding", lambda *_args, **_kwargs: None) as runtime:
            original = runtime.lease
            runtime.lease = replace(original, resource_id="foreign-native-run")
            try:
                with pytest.raises(ValueError, match="signed native profile and live lease binding"):
                    refusal.refuse_finite_proof_query_before_popen(runtime, ValueError("stale proof"))
            finally:
                runtime.lease = original
            scope._spawned = True
            try:
                with pytest.raises(ValueError, match="native unlaunched finite scope"):
                    refusal.refuse_finite_proof_query_before_popen(runtime, ValueError("already born"))
            finally:
                scope._spawned = False
            with pytest.raises(ValueError, match="exact finite proof-query runtime"):
                refusal.refuse_finite_proof_query_before_popen(runtime.manifest, ValueError("inert"))


@pytest.mark.parametrize("callback_stage", ["launch_signature", "completion_binding"])
def test_paired_native_constructor_callback_failure_closes_real_owner_transports(
        proof_query_execution_case, tmp_path, monkeypatch, callback_stage):
    """Exercise actual constructor cleanup with explicit deployment fixtures.

    Native proof/scope ownership, DID launch signing, broker/listener, run
    coordinator and lifecycle STOP are real. Installed container launcher
    custody and sudo UID cleanup are explicitly replaced host fixtures; no
    supervisor or isolated worker is launched or qualified here.
    """
    import hashlib
    from ipfs_accelerate_py.agent_supervisor.entrypoints import admitted_benchmark_runtime as entry
    from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution as candidate_boundary
    from ipfs_accelerate_py.agent_supervisor.runtime import finite_proof_query_worker_dispatch as dispatch
    from ipfs_accelerate_py.agent_supervisor.runtime import local_completion_bridge as completion
    from ipfs_accelerate_py.agent_supervisor.control.lifecycle_orchestrator import LifecycleSagaPhase

    case, source = proof_query_execution_case, proof_query_execution_case["repository"] / "calc.py"
    before, mode = source.read_bytes(), source.stat().st_mode & 0o777
    observed = {"runtime": None, "sudo_cleanup_calls": 0}
    binding = {"schema": candidate_boundary.SCHEMA,
        "argv": ["/opt/ipfs-supervisor/bin/validation-worker"], "files": {},
        "owner_uid": os.geteuid(), "worker_uid": 1001,
        "namespaces": {name: os.readlink("/proc/self/ns/" + name) for name in ("pid", "mnt", "net")},
        "completion_authority": False}
    monkeypatch.setattr(candidate_boundary, "bind_candidate_runner", lambda argv: deepcopy(binding))
    monkeypatch.setattr(candidate_boundary, "verify_candidate_runner", lambda value: value)
    monkeypatch.setattr(candidate_boundary, "_root_file",
        lambda path: hashlib.sha256(str(path).encode("utf-8")).hexdigest())
    start_broker = dispatch.start_owner_finite_proof_query_dispatch_broker
    def capture_broker(*, scope, runtime):
        broker = start_broker(scope=scope, runtime=runtime)
        observed["runtime"] = runtime
        # Preserve real constructor cleanup custody if an assertion fails in
        # the genuine broker controls before the factory's assignment returns.
        runtime.proof_query_dispatch_broker = broker
        from test.api.test_finite_proof_query_worker_dispatch import assert_genuine_broker_private_custody_controls
        observed["cached_custody_controls"] = assert_genuine_broker_private_custody_controls(broker)
        return broker
    monkeypatch.setattr(dispatch, "start_owner_finite_proof_query_dispatch_broker", capture_broker)
    def mutate_source():
        source.chmod(0o600)
        source.write_bytes(before + b"\n# constructor callback source drift\n")
        source.chmod(mode)
    sign_launch = entry.sign_profile_binding
    def late_signer(**arguments):
        signed = sign_launch(**arguments)
        if callback_stage == "launch_signature":
            mutate_source()
            raise ValueError("host launch-signature callback refused")
        return signed
    monkeypatch.setattr(entry, "sign_profile_binding", late_signer)
    def late_completion(**_arguments):
        assert callback_stage == "completion_binding"
        mutate_source()
        raise ValueError("host completion-binding callback refused")
    monkeypatch.setattr(completion, "bind_owner_local_completion_service", late_completion)
    actual_run = subprocess.run
    def host_cleanup(command, *args, **kwargs):
        if list(command[:5]) == ["/usr/bin/sudo", "-n", "-u", "benchmarkworker", "--"]:
            assert command[-1] == "--cleanup"
            observed["sudo_cleanup_calls"] += 1
            return subprocess.CompletedProcess(command, 0, b"", b"")
        return actual_run(command, *args, **kwargs)
    monkeypatch.setattr(subprocess, "run", host_cleanup)

    # The broker's real Unix socket stays within the kernel's pathname bound.
    with tempfile.TemporaryDirectory(prefix="fpq-ctor-") as short_root:
        with reserve(case, tmp_path / ("constructor-" + callback_stage)) as (scope, _closure):
            try:
                with pytest.raises(ValueError, match="host .* callback refused"):
                    entry.AdmittedBenchmarkRuntime.create(Path(short_root) / "launch",
                        admission=case["admission"]["local_admission"],
                        server=case["native"].server, source=case["native"].source,
                        implement=True,
                        implementation_command=scope.to_dict()["payload"]["candidate"]["implementation_command"],
                        candidate_runner_argv=binding["argv"], finite_execution_scope=scope)
                runtime = observed["runtime"]
                assert type(runtime) is entry.AdmittedBenchmarkRuntime
                assert runtime.proof_query_dispatch_broker._closed
                assert not runtime.proof_query_dispatch_broker.socket_path.exists()
                assert not runtime.proof_query_dispatch_broker._thread.is_alive()
                assert runtime._listener.fileno() == -1 and not scope._spawned
                assert not getattr(runtime, "_construction_cleanup_failed", False)
                if callback_stage == "launch_signature":
                    assert not hasattr(runtime, "service") and not hasattr(runtime, "lease")
                    assert observed["sudo_cleanup_calls"] == 0
                else:
                    assert runtime._children == [] and runtime.process.snapshot(runtime.profile).members == ()
                    assert runtime._context_refresh_stopped() and scope._cleaned
                    assert not runtime._bootstrap_thread.is_alive()
                    assert runtime.orchestrator.store.latest()[runtime.profile.target_id].phase is LifecycleSagaPhase.COMMITTED
                    assert observed["sudo_cleanup_calls"] == 1
                # Retain fully closed native transport/STOP/run-lease evidence
                # before this short socket pathname is removed. Original signed
                # /tmp path strings stay opaque historical identities.
                retained = tmp_path / "exactlaunch-state"
                shutil.copytree(runtime.directory, retained)
                native_rows = {}
                database = retained / "state" / "coordination.duckdb"
                if database.exists():
                    import duckdb
                    connection = duckdb.connect(str(database), read_only=True)
                    try:
                        for table in ("fenced_leases", "resource_claims", "lease_events", "token_history"):
                            cursor = connection.execute("SELECT * FROM " + table + " ORDER BY 1 LIMIT 17")
                            columns = [row[0] for row in cursor.description]
                            native_rows[table] = [dict(zip(columns, row)) for row in cursor.fetchall()]
                        leases = native_rows["fenced_leases"]
                        assert len(leases) == 1 and leases[0]["state"] == "released"
                        assert leases[0]["resource_id"] == runtime.run_id
                        assert leases[0]["repository_id"] == runtime.repository_id
                    finally:
                        connection.close()
                evidence = {"schema": "host-proof-query-constructor-closed-native-state@1",
                    "callback_stage": callback_stage,
                    "genuine_broker_cached_custody_controls": observed["cached_custody_controls"],
                    "original_launch_directory": str(runtime.directory),
                    "retained_launch_directory": str(retained), "native_coordinator_rows": native_rows,
                    "native_STOP_executed": callback_stage == "completion_binding",
                    "run_lease_never_created": callback_stage == "launch_signature",
                    "broker_closed": runtime.proof_query_dispatch_broker._closed,
                    "broker_thread_alive": runtime.proof_query_dispatch_broker._thread.is_alive(),
                    "bootstrap_listener_closed": runtime._listener.fileno() == -1,
                    "scope_process_spawned": scope._spawned,
                    "sudo_cleanup_fixture_calls": observed["sudo_cleanup_calls"],
                    "container_launcher_custody": "explicit-host-fixture",
                    "isolated_UID_cleanup_qualified": False, "completion_authority": False}
                (tmp_path / "constructor-closed-native-state.json").write_text(
                    json.dumps(evidence, sort_keys=True, indent=2, allow_nan=False) + "\n")
            finally:
                source.chmod(0o600)
                source.write_bytes(before)
                source.chmod(mode)
    assert scope.parent_lease.released
