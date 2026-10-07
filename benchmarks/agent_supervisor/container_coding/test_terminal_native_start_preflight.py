"""Native preflight failure evidence survives owned shutdown.

Signed task admission and database materialization are real. The runtime/UID
boundary is explicitly doubled: these tests launch no worker or container and
make no claim that the native lifecycle or task execution was qualified.
"""
from contextlib import contextmanager
import hashlib
import json
from types import SimpleNamespace

import pytest

from benchmarks.agent_supervisor.container_coding import terminal_indexed_preparation as prep
from benchmarks.agent_supervisor.container_coding import terminal_native_preflight as preparation
from benchmarks.agent_supervisor.container_coding import terminal_native_start_preflight as native
from benchmarks.agent_supervisor.container_coding import terminal_resource_diagnostics as diagnostics
from benchmarks.agent_supervisor.container_coding.test_terminal_admission_observation import refused
from benchmarks.agent_supervisor.container_coding.test_terminal_native_preflight import _case, no_provider_calls
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.task_sources.intent_repository import IntentRepository
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import proof_resource_safety as safety
from ipfs_datasets_py.optimizers.logic_theorem_optimizer import resource_scheduler as schedulers


@pytest.fixture
def admitted_case(tmp_path):
    case = _case(tmp_path, "data")
    prepared = prep.prepare(repository=case["repository"], instruction=case["instruction"],
        state=case["state"], task_profile=case["task_profile"], disable_intent_autoencoder=True)
    graph = preparation._authored_graph(prepared)
    admission = local.admit_local_benchmark_plan(graph=graph, manifest=prepared["manifest"])
    prep._write(case["state"] / "admission.json", admission)
    return case, graph


def _run_failure(tmp_path, monkeypatch, admitted_case, error, *, wrapped=False, cleanup_failure=None,
                 fallback_timeout=False):
    case, graph = admitted_case
    root = tmp_path / "isolated-container"
    output = root / "state" / "native-preflight"
    output.parent.mkdir(parents=True)
    events = []
    active = {"worker": False, "sequence": 0}
    monkeypatch.setattr(native, "ROOT", root)
    monkeypatch.setattr(native.os, "getuid", lambda: 1000)
    monkeypatch.setattr(native.os, "geteuid", lambda: 1000)
    monkeypatch.setattr(native, "_workers", lambda: ([{
        "pid": 700, "uid": 1001, "parent_pid": 600, "state": "S", "coding_preflight": True,
    }] if active["worker"] else []))
    monkeypatch.setattr(safety, "collect_proof_host_resources", lambda: safety.ProofHostResources(8, 8192, 8192))
    monkeypatch.setattr(diagnostics, "collect_failure_scheduler", lambda: {
        "schema": diagnostics.SCHEMA, "status": "unavailable", "reason": "test_no_global_owner"})

    @contextmanager
    def open_owner(**kwargs):
        events.append("owner_open")
        # The native qualifier must have materialized the actual signed task.
        with IntentRepository(kwargs["database"], install_schema=False) as intent:
            tasks = intent.list_tasks()
            assert [row["task_cid"] for row in tasks] == [graph.tasks[0].task_cid]
            assert tasks[0]["status"] == "ready"
        try:
            yield SimpleNamespace(server=object(), source=SimpleNamespace(get_task=lambda task_cid:
                SimpleNamespace(status=tasks[0]["status"], revision=tasks[0]["revision"])))
        finally:
            events.append("owner_exit")

    class RuntimeBoundary:
        def __init__(self):
            self.state = output / "launch-state"
            self.state.mkdir()
            self.manifest = {"environment": {}}
            self.bootstrap_receipts = []
            self.bootstrap_errors = []
            self.profile = object()
            self.process = SimpleNamespace(snapshot=lambda profile: SimpleNamespace(members=()))
            workspace = root / "worktrees" / "owned-worker"
            workspace.mkdir(parents=True)
            prompt = b"Public authored preflight control-flow fixture\n"
            (workspace / "worker-preflight-prompt.txt").write_bytes(prompt)
            log = self.state / "run/admitted_database_portal_attempts/owned/implementation-logs/task-attempt-1.log"
            log.parent.mkdir(parents=True)
            log.write_text(json.dumps({"schema": "container-worker-preflight@1", "pid": 700,
                "workspace": str(workspace), "prompt_bytes": len(prompt),
                "prompt_sha256": hashlib.sha256(prompt).hexdigest()}) + "\n")

        def start(self):
            events.append("start")
            active["worker"] = error is None
            return SimpleNamespace(to_dict=lambda: {"status": "succeeded", "fixture_boundary": True})

        def observe(self):
            events.append("observe")
            if error is None:
                active["sequence"] += 1
                return {"healthy": True, "native_heartbeat": {
                    "owner_read_sequence": active["sequence"]}, "process_tree": {"fixture_boundary": True}}
            private_local = "PRIVATE_LOCAL_MUST_NOT_APPEAR_IN_DIAGNOSTICS"
            assert private_local
            if wrapped:
                raise RuntimeError("bounded wrapper") from error
            raise error

        def stop(self):
            events.append("stop")
            active["worker"] = False
            if cleanup_failure == "stop":
                raise OSError("bounded stop failure")
            return SimpleNamespace(to_dict=lambda: {"status": "succeeded", "fixture_boundary": True})

        def close(self):
            events.append("close")
            if fallback_timeout:
                # A final worker observation can reveal a residual after the
                # earlier native STOP sample. This is an authored boundary.
                active["worker"] = True
            if cleanup_failure == "close":
                raise OSError("bounded close failure")

    def create(*args, **kwargs):
        events.append("runtime_create")
        return RuntimeBoundary()

    original_run = native.subprocess.run

    def git_only(argv, **kwargs):
        if argv == ["git", "-C", str(root / "source"), "rev-parse", "--verify", "HEAD"]:
            return SimpleNamespace(returncode=0, stdout="a" * 40 + "\n", stderr="")
        if argv == ["sudo", "-n", "-u", "benchmarkworker", "--", str(root / "bin/worker-entry"), "--cleanup"]:
            assert fallback_timeout
            events.append("fallback_cleanup")
            raise native.subprocess.TimeoutExpired(argv, kwargs["timeout"],
                output=b"PRIVATE_WORKER_STDOUT", stderr=b"PRIVATE_WORKER_STDERR")
        assert argv[0] in ("git", "/usr/bin/git") and str(case["repository"]) in argv
        return original_run(argv, **kwargs)

    monkeypatch.setattr(native, "open_existing_native_owner", open_owner)
    monkeypatch.setattr(native.AdmittedBenchmarkRuntime, "create", create)
    monkeypatch.setattr(native.subprocess, "run", git_only)
    result = native.qualify(prepared_state=case["state"], output=output)
    assert json.loads((output / "result.json").read_bytes()) == result
    assert events == ["owner_open", "runtime_create", "start", "observe",
        *(["observe"] if error is None else []), "stop", "close", "owner_exit",
        *(["fallback_cleanup"] if fallback_timeout else [])]
    assert result["qualified"] is (error is None and cleanup_failure is None and not fallback_timeout)
    assert result["text_generation_calls"] == 0 and result["official_verifier_executed"] is False
    assert result["benchmark_success"] is None
    assert bool(result["workers_after_finally"]) is fallback_timeout
    assert not (case["repository"] / "result.json").exists()
    if error is not None:
        assert result["error_phase"] == "observe"
        frames = result["error_traceback"]["frames"]
        assert any(row["function"] == "observe" for row in frames)
        assert all(set(row) == {"file", "line", "function"} for row in frames)
        assert len(frames) <= 20
    assert "PRIVATE_LOCAL_MUST_NOT_APPEAR_IN_DIAGNOSTICS" not in json.dumps(result)
    return result


@pytest.mark.parametrize("wrapped", [False, True])
def test_actual_lease_admission_survives_observe_failure_and_owned_stop(
        tmp_path, monkeypatch, admitted_case, refused, wrapped):
    result = _run_failure(tmp_path, monkeypatch, admitted_case, refused, wrapped=wrapped)
    assert result["error"]["type"] == ("RuntimeError" if wrapped else "LeaseTimeoutError")
    assert result["failure_admission"] == diagnostics.collect_failure_admission(refused)
    assert result["failure_admission"]["status"] == "observed"
    assert result["failure_admission"]["primary_gate"]["reason"] == "proof_memory_stall"
    assert result["failure_admission"]["last_sample"]["host"]["memory_stall_percent"] == 10
    assert result["failure_resources"]["memory_stall_percent"] == 0
    assert result["failure_admission"]["complete_admission_decision"] is False
    assert result["failure_admission"]["causal_proof"] is False
    assert result["stop"]["status"] == "succeeded"
    assert result["remaining_native_processes"] == 0
    assert result["remaining_worker_processes"] == []
    assert "PRIVATE_REQUEST" not in json.dumps(result)


@pytest.mark.parametrize("cleanup_failure", ["stop", "close"])
def test_cleanup_error_does_not_replace_primary_resource_failure(
        tmp_path, monkeypatch, admitted_case, refused, cleanup_failure):
    result = _run_failure(tmp_path, monkeypatch, admitted_case, refused,
        cleanup_failure=cleanup_failure)
    assert result["error"]["type"] == "LeaseTimeoutError"
    assert result["failure_admission"] == diagnostics.collect_failure_admission(refused)
    assert len(result["cleanup_errors"]) == 1
    secondary = result["cleanup_errors"][0]
    assert secondary["type"] == "OSError"
    assert secondary["error_phase"] == cleanup_failure
    assert secondary["failure_admission"]["status"] == "unavailable"
    assert secondary["failure_admission"]["reason"] == "no_native_admission_error"


@pytest.mark.parametrize("kind,reason", [
    ("unattached", "no_attached_observation"), ("ordinary", "no_native_admission_error"),
])
def test_missing_native_admission_is_explicit_and_still_cleans_up(
        tmp_path, monkeypatch, admitted_case, kind, reason):
    error = (schedulers.LeaseTimeoutError("bounded unattached refusal") if kind == "unattached"
        else ValueError("bounded ordinary failure"))
    result = _run_failure(tmp_path, monkeypatch, admitted_case, error)
    assert result["failure_admission"]["status"] == "unavailable"
    assert result["failure_admission"]["reason"] == reason
    assert result["stop"]["status"] == "succeeded"


def test_two_healthy_observations_and_stop_keep_existing_success_control_flow(
        tmp_path, monkeypatch, admitted_case):
    """Boundary-double success covers control flow, not actual native health."""
    result = _run_failure(tmp_path, monkeypatch, admitted_case, None)
    assert "error" not in result and "failure_admission" not in result
    assert [row["native_heartbeat"]["owner_read_sequence"] for row in result["observations"]] == [1, 2]
    assert all(row["workers"][0]["pid"] == result["worker_preflight_receipt"]["pid"]
        for row in result["observations"])
    assert result["task_state"]["status"] == "ready"
    assert result["remaining_native_processes"] == 0
    assert result["remaining_worker_processes"] == []


@pytest.mark.parametrize("has_primary_failure", [True, False])
def test_fallback_timeout_preserves_primary_and_persists_unqualified_final_report(
        tmp_path, monkeypatch, admitted_case, refused, has_primary_failure):
    result = _run_failure(tmp_path, monkeypatch, admitted_case,
        refused if has_primary_failure else None, fallback_timeout=True)
    assert result["qualified"] is False
    assert result["seconds"] >= 0
    assert result["workers_after_finally"][0]["pid"] == 700
    assert result["stop"]["status"] == "succeeded"
    if has_primary_failure:
        assert result["error"]["type"] == "LeaseTimeoutError"
        assert result["error_phase"] == "observe"
        assert result["failure_admission"] == diagnostics.collect_failure_admission(refused)
        assert len(result["cleanup_errors"]) == 1
        fallback = result["cleanup_errors"][0]
        assert fallback["type"] == "TimeoutExpired"
    else:
        assert result["error"]["type"] == "TimeoutExpired"
        assert len(result["observations"]) == 2
        fallback = result
    assert fallback["error_phase"] == "fallback_cleanup"
    assert fallback["failure_admission"]["status"] == "unavailable"
    assert fallback["failure_admission"]["reason"] == "no_native_admission_error"
    serialized = json.dumps(result)
    assert "PRIVATE_WORKER_STDOUT" not in serialized
    assert "PRIVATE_WORKER_STDERR" not in serialized
