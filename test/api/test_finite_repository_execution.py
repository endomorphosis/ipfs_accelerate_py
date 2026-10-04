"""Real finite reservation preserves signed tasks and native completion guards.

These host cases qualify native accounting and prelaunch controls. The actual
isolated worker, kernel limits and publication are exercised in Docker.
"""
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import replace
import threading

import pytest

from benchmarks.agent_supervisor.container_coding.native_quack_qualification import open_existing_native_owner
from ipfs_accelerate_py.agent_supervisor.entrypoints.admitted_benchmark_runtime import AdmittedBenchmarkRuntime
from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_admission as finite
from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_candidate_runner as candidate_runner
from ipfs_accelerate_py.agent_supervisor.runtime import finite_repository_execution as execution
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import run_owner_local_task_validations
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_accelerate_py.agent_supervisor.task_sources.task_execution_route_policy import GROK_CODEX_EXECUTION_MODE
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import DatabaseImplementationDaemon
from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import ResourceLease
from ipfs_datasets_py.logic.software_contracts.content import cid_for_structured
from test.api.test_finite_repository_admission import _build_case, _author, _admit
from test.api.test_finite_integer_codebase import finite_tools, finite_source  # noqa: F401
from test.api.test_local_completion_bridge import complete


def _daemon(case, suffix):
    native = case["native"]
    return DatabaseImplementationDaemon(
        database_path=native.database, coordination_path=case["root"] / (suffix + "-coordination.duckdb"),
        execution_path=case["root"] / (suffix + "-execution.duckdb"), authority_mode="quack",
        task_source_kind="duckdb", owner_session_id="session:finite-execution-" + suffix,
        process_instance_id=native.identity.process_birth_id, quack_uri=native.identity.listen_uri,
        task_source=native.source, close_task_source=False,
        state_owner_bootstrap_credentials=native.credentials,
        strict_task_sharding=True, max_task_attempts=2, lease_ms=120_000,
        require_real_execution=True,
    ).open()


def _population(case):
    page = case["native"].source.list_tasks(limit=17)
    assert not page.next_cursor
    return local._plain([dict(row.to_dict()) for row in page.tasks])


@contextmanager
def _native_finite_case(root, tools, *, complete_type=True):
    case = _build_case(root, tools)
    root.chmod(0o700)
    try:
        case["declaration"] = _author(case)
        case["admission"] = _admit(case, output=root / "signed-evidence")
        with IntentRepository(root / "intent.duckdb") as intent:
            finite.materialize_finite_repository_plan(owner=case["owner"], admission=case["admission"],
                intent=intent, output=root / "materialized-evidence", policy_observer=lambda request: request.roots)
        with open_existing_native_owner(database=root / "intent.duckdb", checkout=case["repository"],
                state_dir=root / "native-owner", repository_id=case["manifest"]["payload"]["repository_cid"],
                execution_routes={"TYPE-TASK": GROK_CODEX_EXECUTION_MODE,
                                  "OFFSET-TASK": GROK_CODEX_EXECUTION_MODE}) as native:
            assert native.identity.extension_fingerprint, "actual native Quack extension identity required"
            case["native"] = native
            by_key = {task.task_key: task.task_cid for task in case["graph"].tasks}
            case["type_cid"], case["offset_cid"] = by_key["TYPE-TASK"], by_key["OFFSET-TASK"]
            if complete_type:
                driver = _daemon(case, "prerequisite")
                try:
                    attempt = driver.claim_next()
                    assert attempt is not None and attempt.task_cid == case["type_cid"]
                    claimed = native.source.get_task(attempt.task_cid)
                    observation = run_owner_local_task_validations(server=native.server,
                        task_cid=attempt.task_cid, attempt_id=attempt.attempt_id,
                        expected_revision=claimed.revision)
                    assert observation["passed"] is True
                    complete(native, attempt, observation["results"][0]["evidence_digest"])
                    assert native.source.get_task(case["type_cid"]).status == "completed"
                finally:
                    driver.close()
                with native.server._lock:
                    with IntentRepository(bound_connection=native.server._connection, install_schema=False) as intent:
                        case["candidate"] = candidate_runner.author_finite_repository_candidate(
                            admission=case["admission"], intent=intent, task_cid=case["offset_cid"],
                            after_bytes=finite_source(2), output=root / "finite-candidate.json")
            yield case
    finally:
        state = case["scheduler"].snapshot()
        assert state["active_lease_count"] == state["waiting_request_count"] == 0
        case["connection"].close()


@pytest.fixture(scope="module")
def native_finite_case(tmp_path_factory, finite_tools):
    with _native_finite_case(tmp_path_factory.mktemp("finite-native-execution") / "case", finite_tools) as case:
        yield case


def _reserve(case, output, **changes):
    options = dict(owner=case["owner"], admission=case["admission"], server=case["native"].server,
        source=case["native"].source, candidate=case["candidate"], output=output,
        policy_observer=lambda request: request.roots)
    options.update(changes)
    return execution.reserve_finite_repository_execution(**options)


def test_real_reservation_preserves_all_native_rows_and_releases_unspawned_envelope(native_finite_case, tmp_path):
    case = native_finite_case
    before = _population(case)
    with _reserve(case, tmp_path / "reservation") as scope:
        assert type(scope) is execution.FrozenFiniteRepositoryExecutionScope
        assert type(scope.parent_lease) is ResourceLease
        assert not scope.parent_lease.released
        assert scope.parent_lease.cpu_slots == 4
        assert scope.parent_lease.memory_mb == 4096
        assert scope.parent_lease.child_process_slots == 8
        assert scope.selected_task_cids == (case["offset_cid"],)
        payload = scope.to_dict()["payload"]
        rows = {row["task_cid"]: row for row in payload["native_population"]["tasks"]}
        assert set(rows) == {case["type_cid"], case["offset_cid"]}
        assert rows[case["type_cid"]]["status"] == "completed"
        assert rows[case["offset_cid"]]["status"] == "ready"
        assert rows[case["offset_cid"]]["revision"] == case["candidate"]["task_revision"]
        assert payload["native_population"]["completed_prerequisites"][case["type_cid"]]
        receipt_rows = payload["native_population"]["completion_rows"][case["type_cid"]]
        assert receipt_rows and all(row[1] == case["type_cid"] for row in receipt_rows)
        assert all(local.CONTRACT_KEY in row["body"] for row in rows.values())
        assert payload["task_population_preserved"] is True
        assert payload["finite_facts_are_context_only"] is True
        assert payload["candidate"]["descriptor"] == case["candidate"]
        assert payload["candidate"]["argv"][0] == "/opt/ipfs-supervisor/bin/owner-worker"
        for flag in ("task_omission_authority", "completion_authority", "proof_authority",
                     "publication_authority", "production_activation"):
            assert payload[flag] is False
        detached = scope.to_dict()
        detached["payload"]["native_population"]["selected_task_cids"].clear()
        assert scope.selected_task_cids == (case["offset_cid"],)
        with pytest.raises(ValueError, match="exact native finite execution scope"):
            AdmittedBenchmarkRuntime.create(tmp_path / "serialized-launch",
                admission=case["admission"]["local_admission"], server=case["native"].server,
                source=case["native"].source, finite_execution_scope=scope.to_dict())
        assert not (tmp_path / "serialized-launch").exists()
        scope.require_prelaunch_current()
        # The host need not install the isolated launcher to reject a foreign
        # command before launcher inspection; the successful launch is Docker.
        with pytest.raises(execution.FiniteRepositoryExecutionError, match="immutable candidate worker command"):
            scope.require_launch(admission=case["admission"]["local_admission"],
                server=case["native"].server, source=case["native"].source,
                implement=True, candidate_runner=object(), context_bundle=None,
                refresh_context_on_completion=False, implementation_command="foreign-command")
        assert case["scheduler"].snapshot()["active_lease_count"] == 1
    assert scope.parent_lease.released
    assert _population(case) == before
    assert case["scheduler"].snapshot()["active_lease_count"] == 0
    with pytest.raises(execution.FiniteRepositoryExecutionError, match="active"):
        scope.require_prelaunch_current()


@pytest.mark.parametrize("changes", [
    {"cpu_slots": True}, {"cpu_slots": 3}, {"memory_mb": True}, {"memory_mb": 4095},
    {"child_process_slots": True}, {"child_process_slots": 7},
    {"admission_timeout_seconds": True}, {"admission_timeout_seconds": 0},
    {"admission_timeout_seconds": 91},
])
def test_invalid_whole_process_budgets_cannot_install_lease_or_change_tasks(native_finite_case, tmp_path, changes):
    case, output = native_finite_case, tmp_path / "refused"
    before = _population(case)
    with pytest.raises(execution.FiniteRepositoryExecutionError, match="envelope"):
        with _reserve(case, output, **changes):
            pytest.fail("invalid resource envelope was admitted")
    assert not output.exists()
    assert case["scheduler"].snapshot()["active_lease_count"] == 0
    assert _population(case) == before


@pytest.mark.parametrize("kind", ["source", "retained-evidence", "candidate"])
def test_late_source_and_pinned_artifact_drift_refuses_prelaunch_and_releases(native_finite_case, tmp_path, kind):
    case = native_finite_case
    from pathlib import Path
    target = {"source": case["repository"] / "calc.py",
        "retained-evidence": Path(case["admission"]["evidence"]["match"]["observation"]["artifacts"]["source"]["path"]),
        "candidate": Path(case["candidate"]["artifact"])}[kind]
    raw, mode = target.read_bytes(), target.stat().st_mode & 0o777
    before = _population(case)
    try:
        with _reserve(case, tmp_path / "reservation") as scope:
            target.chmod(0o600)
            target.write_bytes(raw + b"\n# late mutation\n")
            target.chmod(mode)
            with pytest.raises((ValueError, RuntimeError)):
                scope.require_prelaunch_current()
    finally:
        target.chmod(0o600)
        target.write_bytes(raw)
        target.chmod(mode)
    assert scope.parent_lease.released
    assert _population(case) == before


def test_cancelled_real_envelope_cannot_grant_prelaunch_and_still_releases(native_finite_case, tmp_path):
    case, cancel = native_finite_case, threading.Event()
    with _reserve(case, tmp_path / "reservation", owner=replace(case["owner"], cancel_event=cancel)) as scope:
        cancel.set()
        with pytest.raises(execution.FiniteRepositoryExecutionError, match="cancelled"):
            scope.require_prelaunch_current()
    assert scope.parent_lease.released
    assert case["native"].source.get_task(case["offset_cid"]).status == "ready"


def test_current_policy_drift_invalidates_frozen_context_without_task_transition(native_finite_case, tmp_path):
    case, drifted = native_finite_case, threading.Event()
    before = _population(case)
    def observer(request):
        return replace(request.roots, policy_root=cid_for_structured({"policy": "new"})) \
            if drifted.is_set() else request.roots
    with _reserve(case, tmp_path / "reservation", policy_observer=observer) as scope:
        drifted.set()
        with pytest.raises(ValueError):
            scope.require_prelaunch_current()
    assert scope.parent_lease.released
    assert _population(case) == before


@pytest.mark.parametrize("field", ["task_revision", "task_cid", "semantic_context_cid", "after_sha256"])
def test_genuine_candidate_descriptor_cannot_be_rebound_to_different_native_scope(native_finite_case, tmp_path, field):
    case = native_finite_case
    descriptor = deepcopy(case["candidate"])
    descriptor[field] = descriptor[field] + 1 if field == "task_revision" else "foreign"
    before = _population(case)
    with pytest.raises(execution.FiniteRepositoryExecutionError, match="candidate context"):
        with _reserve(case, tmp_path / "refused", candidate=descriptor):
            pytest.fail("rebound candidate descriptor was admitted")
    assert _population(case) == before
    assert case["scheduler"].snapshot()["active_lease_count"] == 0


def test_source_mutation_after_real_scope_signing_cannot_return_launch_control(native_finite_case, tmp_path, monkeypatch):
    case, signer = native_finite_case, local._signed
    target = case["repository"] / "calc.py"
    raw, signed_scope = target.read_bytes(), []
    def late_edit(payload, manifest):
        result = signer(payload, manifest)
        if payload.get("schema") == execution.SCHEMA:
            signed_scope.append(result)
            target.write_bytes(finite_source(9))
        return result
    monkeypatch.setattr(local, "_signed", late_edit)
    try:
        with pytest.raises((ValueError, RuntimeError)):
            with _reserve(case, tmp_path / "refused"):
                pytest.fail("late source mutation returned a live launch control")
        assert signed_scope, "mutation must follow actual Ed25519 signing of the scope"
    finally:
        target.write_bytes(raw)
    assert case["scheduler"].snapshot()["active_lease_count"] == 0


def test_completed_status_without_real_native_completion_receipts_cannot_reserve(native_finite_case, tmp_path):
    case, server = native_finite_case, native_finite_case["native"].server
    with server._lock:
        receipts = server._connection.execute("SELECT * FROM completion_receipts WHERE task_cid = ?",
            [case["type_cid"]]).fetchall()
        receipts = [tuple(row[position] for position in range(len(row))) for row in receipts]
        assert receipts
        server._connection.execute("DELETE FROM completion_receipts WHERE task_cid = ?", [case["type_cid"]])
    try:
        with pytest.raises(execution.FiniteRepositoryExecutionError, match="public-check evidence"):
            with _reserve(case, tmp_path / "refused"):
                pytest.fail("status-only prerequisite was admitted")
    finally:
        with server._lock:
            marks = ",".join("?" for _ in receipts[0])
            server._connection.executemany("INSERT INTO completion_receipts VALUES (" + marks + ")", receipts)
    assert case["scheduler"].snapshot()["active_lease_count"] == 0


def test_real_external_claim_invalidates_selected_ready_revision(tmp_path, finite_tools):
    with _native_finite_case(tmp_path / "fresh", finite_tools) as case:
        with _reserve(case, tmp_path / "reservation") as scope:
            driver = _daemon(case, "external-offset")
            try:
                attempt = driver.claim_next()
                assert attempt is not None and attempt.task_cid == case["offset_cid"]
                claimed = case["native"].source.get_task(attempt.task_cid)
                assert claimed.status == "in_progress"
                assert claimed.revision > case["candidate"]["task_revision"]
                assert claimed.body["completion_receipt"]["attempt_id"] == attempt.attempt_id
                with pytest.raises(execution.FiniteRepositoryExecutionError):
                    scope.require_prelaunch_current()
            finally:
                driver.close()
        assert scope.parent_lease.released


def test_finite_type_fact_does_not_complete_original_native_prerequisite(tmp_path, finite_tools):
    with _native_finite_case(tmp_path / "not-completed", finite_tools, complete_type=False) as case:
        assert case["native"].source.get_task(case["type_cid"]).status == "ready"
        with case["native"].server._lock:
            with IntentRepository(bound_connection=case["native"].server._connection, install_schema=False) as intent:
                with pytest.raises(ValueError, match="completed type prerequisite"):
                    candidate_runner.author_finite_repository_candidate(admission=case["admission"], intent=intent,
                        task_cid=case["offset_cid"], after_bytes=finite_source(2), output=tmp_path / "refused.json")
        assert case["native"].source.get_task(case["type_cid"]).status == "ready"
        assert not (tmp_path / "refused.json").exists()
