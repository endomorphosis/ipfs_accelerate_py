"""Isolated validator rejection and real-container qualification.

The final two tests run only inside the installed root-owned worker deployment;
they do not simulate UIDs, capabilities, signatures, subprocesses or outcomes.
"""
import json
import os
import subprocess

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import candidate_execution as candidate
from ipfs_accelerate_py.agent_supervisor.runtime import local_planning_admission as local
from ipfs_accelerate_py.agent_supervisor.runtime.local_completion_bridge import _git as publication_git
from test.api.test_agent_supervisor_local_planning_admission import scenario, _start, _complete  # noqa: F401


@pytest.mark.parametrize("argv", [None, [], ["/bin/true"], [str(candidate.ROOT / "bin/validation-worker"), "--extra"]])
def test_arbitrary_runner_cannot_select_owner_execution(argv):
    with pytest.raises(ValueError, match="exact installed"):
        candidate.bind_candidate_runner(argv)


def test_explicit_bad_binding_has_no_local_execution_or_signed_result(scenario):
    _start(scenario)
    marker = scenario["repository"] / ".runtime/executed"
    marker.parent.mkdir()
    (scenario["repository"] / "answer.py").write_text(
        "from pathlib import Path\nPath('.runtime/executed').touch()\ndef answer(): return 2\n"
    )
    with pytest.raises(ValueError):
        local.run_local_task_validations(intent=scenario["intent"], task_cid=scenario["task_cid"],
            attempt_id="invalid-runner", candidate_runner={"argv": ["/bin/sh"]})
    assert not marker.exists()
    with scenario["intent"]._connection() as connection:
        assert connection.execute("SELECT count(*) FROM validation_results").fetchone()[0] == 0


@pytest.mark.parametrize("read_git", [local._git, publication_git])
def test_owner_git_does_not_execute_candidate_configured_fsmonitor(tmp_path, read_git):
    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    monitor = tmp_path / "monitor"
    marker = tmp_path / "executed"
    monitor.write_text("#!/bin/sh\ntouch '" + str(marker) + "'\n")
    monitor.chmod(0o755)
    subprocess.run(["git", "-C", str(tmp_path), "config", "core.fsmonitor", str(monitor)], check=True)
    read_git(tmp_path, "status", "--porcelain")
    assert not marker.exists()


@pytest.fixture
def actual_container(scenario, tmp_path):
    if os.environ.get("SUPERVISOR_CANDIDATE_CONTAINER_TEST") != "1":
        pytest.skip("requires the actual separate-UID root-owned container deployment")
    binding = candidate.bind_candidate_runner((str(candidate.ROOT / "bin/validation-worker"),))
    # Pytest creates private parents. Admit read/execute only for fixture source
    # below the deployment's exact worktree root, preserving private profiles.
    worktrees = candidate.ROOT / "worktrees"
    assert tmp_path.is_relative_to(worktrees)
    for path in (tmp_path, *tmp_path.parents):
        if path == worktrees:
            break
        path.chmod(0o750)
    root = scenario["repository"]
    root.chmod(0o750)
    for path in root.iterdir():
        if path.is_file():
            path.chmod(0o640)
    scenario["profile"].chmod(0o700)
    scenario["lifecycle"].chmod(0o700)
    private = candidate.ROOT / "state/candidate-validator-private-probe"
    private.write_text("synthetic private owner state; no credential\n")
    private.chmod(0o600)
    return binding, private


def _candidate_source(private):
    return ("import os\nfrom pathlib import Path\ndef answer():\n"
            "    assert os.getuid() == os.geteuid() == 1001\n"
            f"    try: Path({str(private)!r}).read_bytes()\n"
            "    except PermissionError: return 2\n"
            "    raise AssertionError('owner private state became readable')\n")


def test_actual_container_candidate_import_has_no_owner_authority_but_owner_can_complete(scenario, actual_container):
    binding, private = actual_container
    _start(scenario)
    (scenario["repository"] / "answer.py").write_text(_candidate_source(private))
    observed = local.run_local_task_validations(intent=scenario["intent"], task_cid=scenario["task_cid"],
        attempt_id="actual-isolated-worker-check", candidate_runner=binding)
    assert observed["passed"] is True
    with scenario["intent"]._connection() as connection:
        row = connection.execute("SELECT body_json FROM validation_results WHERE task_cid=?", [scenario["task_cid"]]).fetchone()
    signed = json.loads(row[0])["local_observed_validation"]
    assert signed["payload"]["candidate_runner"] == binding
    assert signed["payload"]["validation"] == scenario["manifest"]["payload"]["tasks"][0]["validations"][0]
    _complete(scenario, evidence_digests=[observed["results"][0]["evidence_digest"]])
    assert scenario["intent"].get_task(scenario["task_cid"])["status"] == "completed"


def test_actual_container_native_portal_scheduler_executes_only_worker(scenario, actual_container, tmp_path):
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import PortalImplementationDaemon, PortalTask
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import bind_task_state_control_plane
    binding, private = actual_container
    root = scenario["repository"]
    (root / "answer.py").write_text(_candidate_source(private))
    state = tmp_path / "portal-state.json"
    bind_task_state_control_plane(state, str(tmp_path / "portal-local.duckdb"))
    portal = PortalImplementationDaemon(todo_path=tmp_path / "todo.md", state_path=state,
        strategy_path=tmp_path / "strategy.json", events_path=tmp_path / "events.jsonl", repo_root=root,
        validation_cache_dir=tmp_path / "validation-cache", implementation_timeout=30,
        worktree_pool_enabled=False)
    try:
        candidate.install_portal_candidate_runner(portal, binding)
        task = PortalTask(task_id="LOCAL-TASK", title="Isolated validator", status="ready",
            completion="manual", priority="P1", track="local", validation=["python3 test_answer.py"])
        observed = portal._run_validation_commands(root, task, tmp_path / "validation.log")
        assert observed["passed"] is True, observed
        assert observed["attempted"] is True
        assert any(item.get("candidate_runner") == binding for item in observed["results"])
        assert not any(item.get("cache_hit") for item in observed["results"])
        rescued = portal._run_auto_rescue_materialize_commands(
            workspace_path=root, log_path=tmp_path / "rescue.log",
            commands=["python3 test_answer.py"], task=task,
        )
        assert len(rescued) == 1 and rescued[0]["ok"] is True
        assert rescued[0]["candidate_runner"] == binding
        scripts = root / "scripts"
        scripts.mkdir()
        (scripts / "materialize_lgswf_099.py").write_text("raise AssertionError('must not execute as owner')\n")
        with pytest.raises(ValueError, match="outside isolated"):
            portal._lgswf_writer_path("LGSWF-099")
    finally:
        portal.close_event_runtime()
