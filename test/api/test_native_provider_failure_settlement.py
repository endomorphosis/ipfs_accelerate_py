"""Native failed-child custody, independent of provider/model completion claims."""
import json
import os
import shlex
import signal
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import (
    DatabasePortalBridgeError, DatabasePortalExecutionBridge,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon, PortalTaskState, DatabaseImplementationAuthorityError,
)
from test.api.test_agent_supervisor_database_implementation_daemon import _open_daemon, _population


def _repository(path):
    path.mkdir()
    for args in (("init", "-q"), ("config", "user.name", "Test"),
                 ("config", "user.email", "test@example.invalid")):
        subprocess.run(["git", *args], cwd=path, check=True)
    (path / "result.py").write_text("VALUE = 0\n")
    subprocess.run(["git", "add", "."], cwd=path, check=True)
    subprocess.run(["git", "commit", "-qm", "fixture"], cwd=path, check=True)
    return subprocess.check_output(["git", "branch", "--show-current"], cwd=path, text=True).strip()


def _native_chain(tmp_path, monkeypatch, *, partial_edit=False, pooled=False):
    repository = tmp_path / "repository"
    branch = _repository(repository)
    script = tmp_path / "router_timeout.py"
    root = Path(__file__).resolve().parents[2]
    script.write_text('''import subprocess, sys
from types import SimpleNamespace
sys.path.insert(0, ROOT)
from ipfs_accelerate_py import llm_router
from ipfs_accelerate_py.cli_runtime.cli_metadata import set_last_cli_observation
from ipfs_accelerate_py.llm_allocation import intelligence_index
from ipfs_accelerate_py.agent_supervisor.runtime import router_implementation_runner as runner
intelligence_index.discover_available_providers = lambda: ["codex_cli"]
intelligence_index.select_efficient_route = lambda **kwargs: SimpleNamespace(provider="codex_cli", model_name="fixture", reasoning_effort="high", catalog_revision="fixture")
llm_router.get_llm_provider = lambda *args, **kwargs: object()
def timeout(*args, **kwargs):
    try:
        subprocess.run([sys.executable, "-c", "import time; time.sleep(5)"], timeout=0.1, check=True)
    except subprocess.TimeoutExpired:
        set_last_cli_observation("codex_cli", {"timed_out": True})
        raise
llm_router.generate_text = timeout
runner.run(prompt="Repair the fixture", provider="codex_cli", model="fixture", timeout=1, max_output_tokens=128)
'''.replace("ROOT", repr(str(root))))
    if partial_edit:
        script.write_text("from pathlib import Path\nPath('result.py').write_text('VALUE = 1\\n')\n" + script.read_text())
    database = _open_daemon(tmp_path, max_task_attempts=1)
    population = _population(1)
    population["tasks"][0].update(body={"completion": "auto", "track": "implementation"},
        outputs=["result.py"], validations=["python -m py_compile result.py"],
        acceptance=["result.py passes validation"])
    database.materialize_population(population)
    portals = []
    def factory(paths, alias):
        portal = PortalImplementationDaemon(
            todo_path=paths.task_projection, state_path=paths.state, strategy_path=paths.strategy,
            events_path=paths.events, repo_root=repository, task_header_prefix="## DQP-",
            board_namespace="fixture", implement=True,
            implementation_command=shlex.join([sys.executable, str(script)]),
            implementation_timeout=20, use_ephemeral_worktree=True,
            worktree_pool_enabled=pooled, worktree_root=tmp_path / "worktrees",
            implementation_log_dir=paths.root / "logs", merge_target_branch=branch,
            max_task_attempts=1, implementation_protected_paths=(),
        )
        portals.append(portal)
        return portal
    bridge = DatabasePortalExecutionBridge(task_source=database.task_source,
        attempt_root=tmp_path / "attempts", portal_factory=factory,
        repo_root=repository, board_namespace="fixture", merge_target_branch=branch,
        task_header_prefix="## DQP-", max_task_attempts=1)
    database._provider_fn = bridge.run_provider
    return database, bridge, portals


@pytest.mark.parametrize("partial_edit", [False, True])
@pytest.mark.parametrize("pooled", [False, True])
def test_actual_router_timeout_settles_native_failure_without_retry(tmp_path, monkeypatch, partial_edit, pooled):
    daemon, bridge, portals = _native_chain(tmp_path, monkeypatch, partial_edit=partial_edit, pooled=pooled)
    try:
        result = daemon.run_once()
        assert result["implementation_result"].get("portal_terminal_failure") is True, result
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt.status == "failed"
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
        receipt = daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}")
        assert receipt["callback_state"] == "failed_outcome_settled"
        assert receipt["native_exit"]["subreaper_children_absent"] is True
        assert receipt["native_exit"]["automatic_retry_admitted"] is False
        assert daemon.claim_next() is None
        assert (tmp_path / "repository/result.py").read_text() == "VALUE = 0\n"
        assert subprocess.check_output(["git", "status", "--porcelain"], cwd=tmp_path / "repository", text=True) == ""
        if partial_edit and not pooled:
            candidates = list((tmp_path / "worktrees").glob("*/result.py"))
            assert len(candidates) == 1 and candidates[0].read_text() == "VALUE = 1\n"
        assert len(portals) == 1
        logs = list(bridge._paths(attempt).root.rglob("*.log"))
        rows = [json.loads(line) for path in logs for line in path.read_text().splitlines()
                if line.startswith('{"') and 'router-implementation-invocation@1' in line]
        assert len(rows) == 1 and rows[0]["error_type"] == "TimeoutExpired"
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import _accepted_source_events
        events, _ = _accepted_source_events(bridge._paths(attempt).events)
        finished = [item for item in events if item.get("type") == "implementation_finished"]
        assert len(finished) == 1 and finished[0]["implementation_commit"] == ""
        assert finished[0]["lifecycle_finalize"]["finalized"] is True
        if pooled:
            cleanup = finished[0]["cleanup_result"]
            original = cleanup["pool_release"]["lifecycle_finalize"]
            assert original["finalized"] is True
            assert cleanup["lifecycle_finalize"] == original
            assert finished[0]["lifecycle_finalize"]["fence"] == original["fence"]
    finally:
        daemon.close()


def _issued_native(process, custody, *, implementation=None):
    native = object.__new__(PortalImplementationDaemon)
    authority = {"fixture": "owned"}
    implementation = implementation or {"returncode": 1}
    native._database_attempt_authority = authority
    native._database_provider_exit = (process, dict(authority), json.dumps(implementation, sort_keys=True, allow_nan=False), custody, "{}")
    return native, authority, implementation


def test_detached_new_session_child_outside_checkout_prevents_settlement(tmp_path):
    custody = PortalImplementationDaemon._begin_database_provider_child_custody()
    assert custody is not None
    marker = tmp_path / "owned-child"
    command = ("import subprocess,sys; "
               "p=subprocess.Popen([sys.executable,'-c','import time; time.sleep(30)'], "
               "start_new_session=True,cwd='/',stdin=subprocess.DEVNULL,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
               f"open({str(marker)!r},'w').write(str(p.pid)); sys.exit(1)")
    process = subprocess.Popen([sys.executable, "-c", command], start_new_session=True)
    detached = None
    try:
        assert process.wait(timeout=5) == 1
        detached = int(marker.read_text())
        with pytest.raises(ProcessLookupError):
            os.killpg(process.pid, 0)
        native, authority, implementation = _issued_native(process, custody)
        assert native._take_database_provider_exit(authority, implementation) is None
        assert native._database_provider_exit is None
        assert os.getpgid(detached) == detached
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if detached is not None:
            os.kill(detached, signal.SIGKILL)
            assert os.waitpid(detached, 0)[0] == detached


def test_preexisting_child_is_preserved_and_disables_custody():
    process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"], start_new_session=True)
    try:
        assert PortalImplementationDaemon._begin_database_provider_child_custody() is None
        assert process.poll() is None
    finally:
        process.kill()
        process.wait(timeout=5)


@pytest.mark.parametrize("invalid", ["foreign_authority", "changed_result", "fake_process", "group_alive", "group_denied", "owner_birth", "children_unknown"])
def test_native_exit_capability_rejects_uncertain_custody(monkeypatch, invalid):
    custody = PortalImplementationDaemon._begin_database_provider_child_custody()
    assert custody is not None
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"], start_new_session=True)
    assert process.wait(timeout=5) == 1
    native, authority, implementation = _issued_native(process, custody)
    if invalid == "foreign_authority":
        authority = {"fixture": "foreign"}
    elif invalid == "changed_result":
        implementation = {"returncode": 2}
    elif invalid == "fake_process":
        native._database_provider_exit = (SimpleNamespace(returncode=1), authority, json.dumps(implementation), custody, "{}")
    elif invalid == "group_alive":
        monkeypatch.setattr(os, "killpg", lambda *args: None)
    elif invalid == "group_denied":
        def denied(*args):
            raise PermissionError("denied")
        monkeypatch.setattr(os, "killpg", denied)
    elif invalid == "owner_birth":
        native._database_provider_exit = (process, authority, json.dumps(implementation), (custody[0], (0, 0, 0, 0)), "{}")
    else:
        def unknown():
            raise PermissionError("child census unavailable")
        monkeypatch.setattr(PortalImplementationDaemon, "_database_provider_children_empty", staticmethod(unknown))
    assert native._take_database_provider_exit(authority, implementation) is None
    assert native._database_provider_exit is None


def test_native_exit_capability_is_one_use():
    custody = PortalImplementationDaemon._begin_database_provider_child_custody()
    assert custody is not None
    process = subprocess.Popen([sys.executable, "-c", "raise SystemExit(1)"], start_new_session=True)
    assert process.wait(timeout=5) == 1
    native, authority, implementation = _issued_native(process, custody)
    assert native._take_database_provider_exit(authority, implementation)["subreaper_children_absent"] is True
    assert native._take_database_provider_exit(authority, implementation) is None


def test_settled_callback_replays_after_response_loss_without_provider_dispatch(tmp_path, monkeypatch):
    daemon, bridge, portals = _native_chain(tmp_path, monkeypatch)
    original = daemon._settle_native_provider_failure
    def lose_response(*args, **kwargs):
        original(*args, **kwargs)
        raise OSError("response lost after exact callback CAS")
    try:
        monkeypatch.setattr(daemon, "_settle_native_provider_failure", lose_response)
        attempt = daemon.claim_next()
        with pytest.raises(OSError, match="response lost"):
            daemon._resume_attempt_without_process_crash(attempt)
        assert daemon.get_attempt(attempt.attempt_id).status == "running"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        receipt = daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}")
        assert receipt["callback_state"] == "failed_outcome_settled"
        monkeypatch.setattr(daemon, "_settle_native_provider_failure", original)
        result = daemon._resume_attempt_without_process_crash(daemon.get_attempt(attempt.attempt_id))
        assert result["portal_terminal_failure"] is True
        assert daemon.get_attempt(attempt.attempt_id).status == "failed"
        assert daemon.task_source.get(attempt.task_cid).status == "blocked"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "released"
        assert len(portals) == 1
        assert daemon.claim_next() is None
    finally:
        daemon.close()


@pytest.mark.parametrize("change", ["foreign_exception", "claim", "fence", "state", "events", "projection", "cas_intent", "stale_source", "later_effect", "altered_native"])
def test_bridge_capability_cannot_cross_changed_authority(tmp_path, monkeypatch, change):
    """Exercise consumption barriers with actual sealed attempt files; no grant is minted."""
    from dataclasses import replace
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import database_portal_bridge as module
    from ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts import content_identity
    daemon = _open_daemon(tmp_path)
    try:
        daemon.materialize_population(_population(1))
        attempt = daemon.claim_next()
        bridge = DatabasePortalExecutionBridge(task_source=daemon.task_source, attempt_root=tmp_path / "attempts", portal_factory=lambda *_: None)
        record = bridge._record_for_attempt(daemon.task_source, attempt)
        paths, binding = bridge._ensure_attempt_projection(attempt, record)
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import save_portal_task_state
        save_portal_task_state(paths.state.stem, {"fixture": "state"}, state_path=paths.state)
        paths.events.write_text('{}\n')
        state_bytes, state_target_digest = bridge._native_provider_state(paths)
        digest = module._accepted_source_events(paths.events)[1]
        receipt = dict(schema="database-native-provider-exit@1",
            **daemon._source384_retry_identity(attempt), task_alias=attempt.task_alias,
            binding_id=binding["binding_id"], portal_attempt=1, returncode=1,
            reaped=True, process_group_absent=True, subreaper_children_absent=True,
            lifecycle_finalized=True, events_digest=digest, state_digest=module._sha256_bytes(state_bytes),
            state_target_digest=state_target_digest,
            completion_authority=False, automatic_retry_admitted=False)
        receipt["receipt_id"] = content_identity(receipt)
        failure = DatabasePortalBridgeError("portal_provider_failed")
        directory = bridge._seal_attempt_directory(paths, attempt_id=attempt.attempt_id, create=False)
        bridge._issued_provider_exit_failure = (failure, receipt, paths, dict(binding), directory, state_bytes)
        if change == "foreign_exception":
            failure = DatabasePortalBridgeError("portal_provider_failed")
        elif change == "claim":
            attempt = replace(attempt, claim_id="foreign")
        elif change == "fence":
            attempt = replace(attempt, fencing_token=attempt.fencing_token + 1)
        elif change == "state":
            save_portal_task_state(paths.state.stem, {"changed": True}, state_path=paths.state)
        elif change == "events":
            paths.events.write_text('{"type":"changed"}\n')
        elif change == "projection":
            paths.task_projection.write_text(paths.task_projection.read_text() + '\n- Acceptance: foreign\n')
        elif change in {"stale_source", "later_effect", "altered_native"}:
            if change == "stale_source":
                def stale(*args):
                    raise DatabaseImplementationAuthorityError("source fence changed")
                monkeypatch.setattr(daemon, "_protect_attempt_write", stale)
                error = "source fence changed"
            elif change == "later_effect":
                monkeypatch.setattr(daemon, "effect_claim_recorded", lambda *a, **kw: {"effect": "already admitted"})
                error = "later effects"
            else:
                receipt["returncode"] = 0
                error = "custody differs"
            with pytest.raises(DatabaseImplementationAuthorityError, match=error):
                daemon._settle_native_provider_failure(attempt, native_exit=receipt, idempotency_key=f"provider:{attempt.attempt_id}")
            assert daemon.provider_invocation_recorded(attempt.attempt_id, idempotency_key=f"provider:{attempt.attempt_id}") is None
            return
        else:
            # Valid local metadata cannot replace an absent/foreign durable
            # callback intent. The authoritative SQL compare-and-swap refuses.
            with pytest.raises(DatabaseImplementationAuthorityError, match="callback changed"):
                daemon._settle_native_provider_failure(attempt, native_exit=receipt, idempotency_key=f"provider:{attempt.attempt_id}")
            return
        assert bridge.take_native_provider_exit_failure(attempt, failure) is None
        assert bridge._issued_provider_exit_failure is None
        assert daemon.coordinator.get_task_claim(daemon.get_attempt(attempt.attempt_id).claim_id).state.value == "accepted"
    finally:
        daemon.close()


def test_forged_native_result_without_process_capability_stays_unknown(tmp_path, monkeypatch):
    daemon, bridge, portals = _native_chain(tmp_path, monkeypatch)
    factory = bridge.portal_factory
    def forged(paths, alias):
        native = factory(paths, alias)
        monkeypatch.setattr(native, "run_once", lambda: {"implementation_result": {
            "returncode": 1, "attempt": 1, "provider_dispatched": True,
            "attempt_consumed": True, "lifecycle_finalize": {"finalized": True},
            "reaped": True, "process_group_absent": True, "subreaper_children_absent": True}})
        return native
    bridge.portal_factory = forged
    try:
        result = daemon.run_once()
        assert result["implementation_result"]["reason"] == "provider_callback_outcome_unknown"
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt.status == "running"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        assert daemon._resume_attempt_without_process_crash(attempt)["reason"] == "provider_callback_outcome_unknown"
        assert len(portals) == 1
    finally:
        daemon.close()


def test_child_census_thread_churn_never_proves_absence(monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon import native_cli_subreaper
    scans = iter([("10",), ("10", "11"), ("10",)])
    original = Path.iterdir
    def listing(path):
        if str(path) == "/proc/self/task":
            return iter(Path(path / item) for item in next(scans))
        return original(path)
    monkeypatch.setattr(Path, "iterdir", listing)
    monkeypatch.setattr(native_cli_subreaper, "_direct_child_pids", lambda: set())
    assert PortalImplementationDaemon._database_provider_children_empty() is False


def test_disabled_subreaper_never_proves_absence(monkeypatch):
    import ctypes
    custody = PortalImplementationDaemon._begin_database_provider_child_custody()
    assert custody is not None
    class Disabled:
        def __call__(self, *args):
            return 0
    monkeypatch.setattr(ctypes, "CDLL", lambda *a, **kw: SimpleNamespace(prctl=Disabled()))
    assert PortalImplementationDaemon._database_provider_children_quiesced(custody) is False


def test_pooled_release_failure_cannot_reload_or_settle_another_owner(tmp_path, monkeypatch):
    daemon, bridge, portals = _native_chain(tmp_path, monkeypatch, pooled=True, partial_edit=True)
    factory = bridge.portal_factory
    released = []
    def guarded_factory(paths, alias):
        native = factory(paths, alias)
        original_release = native._release_pooled_worktree_lease
        original_finalize = native._finalize_worktree_lifecycle
        def release(*args, **kwargs):
            result = original_release(*args, **kwargs)
            assert result["released"] is True
            released.append(True)
            return {**result, "lifecycle_finalize": {
                "finalized": False, "reason": "lifecycle_finalize_race"}}
        def finalize(*args, **kwargs):
            assert not released, "released pooled path must never be looked up as this attempt's owner"
            return original_finalize(*args, **kwargs)
        monkeypatch.setattr(native, "_release_pooled_worktree_lease", release)
        monkeypatch.setattr(native, "_finalize_worktree_lifecycle", finalize)
        return native
    bridge.portal_factory = guarded_factory
    try:
        result = daemon.run_once()
        assert released == [True]
        assert result["implementation_result"]["reason"] == "provider_callback_outcome_unknown"
        attempt = daemon.get_attempt(result["attempt_id"])
        assert attempt.status == "running"
        assert daemon.coordinator.get_task_claim(attempt.claim_id).state.value == "accepted"
        from ipfs_accelerate_py.agent_supervisor.todo_daemon.database_portal_bridge import _accepted_source_events
        events, _ = _accepted_source_events(bridge._paths(attempt).events)
        finished = next(item for item in events if item.get("type") == "implementation_finished")
        assert finished["lifecycle_finalize"] == {"finalized": False, "reason": "lifecycle_finalize_race"}
        assert (tmp_path / "repository/result.py").read_text() == "VALUE = 0\n"
    finally:
        daemon.close()
