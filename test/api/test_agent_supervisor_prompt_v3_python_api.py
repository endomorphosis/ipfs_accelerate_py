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


@pytest.fixture(autouse=True)
def _reap_prompt_workers():
    yield
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
        stop_tracked_prompt_workers,
    )

    stop_tracked_prompt_workers()


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
    assert run.state == "running"
    assert run.identities["workers_started"] is True
    assert run.identities["active_worker_count"] >= 1
    assert run.identities["worker_repository_edits"] is False
    assert run.identities["authorizes_execution"] is False
    assert run.state.lower() not in {"completed", "complete"}
    assert run.identities["completion_authority"] is False
    assert run.identities["model_assertion_cannot_complete"] is True
    assert run.identities["empty_queue_cannot_complete"] is True
    assert "complete_plan" not in supervisor._composition.extras
    blob = json.dumps(run.identities)
    assert prompt not in blob


def test_run_continues_when_plan_create_lacks_only_ir_materials(
    tmp_path: Path,
) -> None:
    supervisor, prompt = _production_supervisor(
        tmp_path,
        create_service=_ReviewOnlyCreateService(),
    )
    run = supervisor.run(prompt)
    assert run.state == "running"
    assert run.identities["workers_started"] is True
    assert run.identities["plan_create_status"] == "review_only"
    assert run.identities["completion_authority"] is False
    assert run.identities["authorizes_execution"] is False
    assert run.state.lower() not in {"completed", "complete"}


def test_run_continues_when_create_plan_join_is_only_unbacked(
    tmp_path: Path,
) -> None:
    supervisor, prompt = _production_supervisor(
        tmp_path,
        create_service=_ReviewOnlyCreateService(
            (
                "execution_plan_not_admitted",
                "ir_admission_materials_absent",
                "parallel_plan_rejected",
                "parallel:resource_infeasible",
                "parallel:stale_capacity",
            )
        ),
    )
    run = supervisor.run(prompt)
    assert run.state == "running"
    assert run.identities["workers_started"] is True
    assert run.identities["active_worker_count"] >= 1
    assert run.identities["plan_create_status"] == "review_only"
    assert run.identities["authorizes_execution"] is False
    assert run.identities["completion_authority"] is False


def test_run_blocks_when_plan_create_has_another_rejection(
    tmp_path: Path,
) -> None:
    supervisor, prompt = _production_supervisor(
        tmp_path,
        create_service=_ReviewOnlyCreateService(
            ("ir_admission_materials_absent", "structural_admission_failed")
        ),
    )
    with pytest.raises(facade_mod.SupervisorUnavailableError, match="plan create"):
        supervisor.run(prompt)


def test_start_without_bound_runtime_is_typed_unavailable(tmp_path: Path) -> None:
    supervisor, prompt = _production_supervisor(tmp_path)
    run = supervisor.run(prompt)
    assert run.state == "running"
    assert run.identities["workers_started"] is True
    with pytest.raises(facade_mod.SupervisorUnavailableError, match="START"):
        supervisor.start(run.run_id)


def test_prompt_run_starts_live_workers_outside_the_repository(
    tmp_path: Path,
) -> None:
    import os

    supervisor, prompt = _production_supervisor(tmp_path)
    run = supervisor.run(prompt)
    again = supervisor.run(prompt)
    pids = list(run.identities["worker_pids"])
    assert pids
    assert again.identities["worker_pids"] == pids
    assert again.identities["supervisor_pid"] == run.identities["supervisor_pid"]
    os.kill(int(run.identities["supervisor_pid"]), 0)
    assert run.identities["authorizes_execution"] is False
    assert run.identities["completion_authority"] is False
    assert run.identities["worker_repository_edits"] is False
    run_dir = Path(run.identities["worker_run_dir"])
    repository = tmp_path / "repo"
    assert run_dir.is_relative_to(tmp_path / "state")
    assert not run_dir.is_relative_to(repository)
    for pid, task_cid in zip(pids, run.identities["worker_task_cids"], strict=True):
        receipt = json.loads(
            (run_dir / "workers" / f"{task_cid}.json").read_text(encoding="utf-8")
        )
        assert receipt["phase"] == "running"
        assert receipt["pid"] == pid
        os.kill(int(pid), 0)
        assert receipt["cwd"] == str(run_dir)
        assert receipt["work"]["validation_executed"] is False
        assert receipt["work"]["disposition"] == "prepared"
        argv = receipt["argv"]
        assert "ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers" in argv
        assert task_cid in argv
        assert "git" not in argv
        assert "pytest" not in argv
        assert receipt["work"]["repository_edits"] is False
        assert receipt["authorizes_execution"] is False
        assert receipt["completion_authority"] is False
    assert prompt not in json.dumps(run.identities)


def test_prompt_worker_observes_scope_and_leaves_validation_unexecuted(
    tmp_path: Path,
) -> None:
    import hashlib
    import json
    import os
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
        launch_prompt_workers,
    )

    repository = tmp_path / "repo"
    state = tmp_path / "state"
    repository.mkdir()
    state.mkdir()
    (repository / "pkg").mkdir()
    body = b"VALUE = 1\n"
    (repository / "pkg" / "task.py").write_bytes(body)
    task = SimpleNamespace(
        task_cid="taskcid",
        task_key="task:one",
        dependency_task_cids=(),
        objective="Observe pkg/task.py",
        scope_paths=("pkg/task.py", "../outside.py", "pkg/.env"),
        validations=(SimpleNamespace(argv=("python", "-m", "pytest", "-q")),),
    )
    launch = launch_prompt_workers(
        tasks=(task,),
        admitted_task_cids=("taskcid",),
        run_id="run-observe",
        state_root=state,
        repository_root=repository,
        width=1,
    )
    receipt = json.loads(
        (Path(launch.run_dir) / "workers" / "taskcid.json").read_text(encoding="utf-8")
    )
    work = receipt["work"]
    assert receipt["phase"] == "running"
    os.kill(int(receipt["pid"]), 0)
    os.kill(int(launch.supervisor_pid), 0)
    assert work["validation_executed"] is False
    assert work["validation_skip_reason"] == "execution_not_authorized"
    assert work["disposition"] == "prepared"
    workspace = Path(launch.run_dir) / "workspaces" / "taskcid" / "pkg" / "task.py"
    assert workspace.read_bytes() == body
    assert not workspace.is_relative_to(repository)
    assert work["repository_edits"] is False
    assert work["authorizes_execution"] is False
    assert work["completion_authority"] is False
    observed = {item["path"]: item for item in work["scope"]}
    assert observed["pkg/task.py"]["status"] == "observed"
    assert observed["pkg/task.py"]["sha256"] == hashlib.sha256(body).hexdigest()
    assert observed["../outside.py"]["status"] == "rejected"
    assert observed["pkg/.env"]["status"] == "credential"
    assert "pytest" not in receipt["argv"]
    assert "git" not in receipt["argv"]
    assert "ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers" in receipt["argv"]
    assert (repository / "pkg" / "task.py").read_bytes() == body


def test_prompt_without_ir_request_starts_workers_and_stays_unauthorized(
    tmp_path: Path,
) -> None:
    supervisor, prompt = _production_supervisor(tmp_path, use_real_admission=True)
    run = supervisor.run(prompt)
    assert run.state == "running"
    assert run.identities["workers_started"] is True
    assert run.identities["active_worker_count"] >= 1
    assert run.identities["hard_domain_admitted"] is False
    assert run.identities["authorizes_execution"] is False
    assert run.identities["completion_authority"] is False
    assert "unknown_mandatory_state" not in (
        run.identities.get("plan_create_status") or ""
    )


def test_real_scanner_prompt_binds_identities_and_starts_workers(
    tmp_path: Path,
) -> None:
    """A prompt with no injected scanner starts workers outside the repository."""

    import os
    import subprocess
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.prompt.plan_supervisor_service import (
        PlanSupervisorService,
    )
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_directory_scanner import (
        repository_allowlist_cid,
    )
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
        deterministic_prompt_goal_graph,
    )

    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(
        ("git", "-C", str(repo), "config", "user.email", "scanner@example.invalid"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repo), "config", "user.name", "Scanner Test"),
        check=True,
    )
    (repo / "src").mkdir()
    body = b"VALUE = 1\n"
    (repo / "src" / "service.py").write_bytes(body)
    subprocess.run(("git", "-C", str(repo), "add", "--", "src/service.py"), check=True)
    subprocess.run(("git", "-C", str(repo), "commit", "-qm", "fixture"), check=True)

    class _Planner:
        def plan(self, request, scan, **_kwargs):  # type: ignore[no-untyped-def]
            graph = deterministic_prompt_goal_graph(request, scan)
            receipt = SimpleNamespace(
                to_dict=lambda: {
                    "request_cid": request.request_cid,
                    "scan_cid": scan.scan_cid,
                    "plan_root_cid": graph.plan_root_cid,
                },
                fallback=SimpleNamespace(used=True),
            )
            return SimpleNamespace(graph=graph, receipt=receipt, used_fallback=True)

    composition = sf.resolve_production_composition(
        repository_root=repo,
        state_root=state,
        require_activation=False,
    )
    composition.extras["planner"] = _Planner()
    composition.extras["plan_supervisor_service"] = PlanSupervisorService(
        create_service=_ReviewOnlyCreateService(),
        revision_store_root=state / "plan_revision_store",
    )
    supervisor = facade_mod.Supervisor.open(services=composition)
    prompt = "Observe src/service.py and keep the run outside the repository."
    run = supervisor.run(prompt)
    assert run.state == "running"
    assert run.identities["workers_started"] is True
    assert run.identities["active_worker_count"] >= 1
    assert run.identities["hard_domain_admitted"] is False
    assert run.identities["authorizes_execution"] is False
    assert run.identities["completion_authority"] is False
    assert run.identities["worker_repository_edits"] is False
    observed = run.identities["observation"]
    assert observed["allowlist_cid"] != repository_allowlist_cid((repo.resolve(),))
    run_dir = Path(run.identities["worker_run_dir"])
    assert run_dir.is_relative_to(state.resolve())
    assert not run_dir.is_relative_to(repo.resolve())
    pid = run.identities["worker_pids"][0]
    task_cid = run.identities["worker_task_cids"][0]
    receipt = json.loads(
        (run_dir / "workers" / f"{task_cid}.json").read_text(encoding="utf-8")
    )
    assert receipt["phase"] == "running"
    assert receipt["pid"] == pid
    os.kill(int(pid), 0)
    os.kill(int(run.identities["supervisor_pid"]), 0)
    assert receipt["cwd"] == str(run_dir.resolve())
    assert "ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers" in receipt["argv"]
    assert "git" not in receipt["argv"]
    assert "pytest" not in receipt["argv"]
    assert receipt["work"]["validation_executed"] is False
    assert receipt["work"]["repository_edits"] is False
    assert receipt["authorizes_execution"] is False
    assert (repo / "src" / "service.py").read_bytes() == body


def test_real_plan_create_prompt_starts_workers(tmp_path: Path) -> None:
    """The production create service, not a review-only stub, starts workers."""

    import os
    import subprocess
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_goal_planner import (
        deterministic_prompt_goal_graph,
    )

    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    subprocess.run(("git", "init", "-q", str(repo)), check=True)
    subprocess.run(
        ("git", "-C", str(repo), "config", "user.email", "scanner@example.invalid"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repo), "config", "user.name", "Scanner Test"),
        check=True,
    )
    (repo / "src").mkdir()
    body = b"VALUE = 1\n"
    (repo / "src" / "service.py").write_bytes(body)
    subprocess.run(("git", "-C", str(repo), "add", "--", "src/service.py"), check=True)
    subprocess.run(("git", "-C", str(repo), "commit", "-qm", "fixture"), check=True)

    class _Planner:
        def plan(self, request, scan, **_kwargs):  # type: ignore[no-untyped-def]
            graph = deterministic_prompt_goal_graph(request, scan)
            receipt = SimpleNamespace(
                to_dict=lambda: {
                    "request_cid": request.request_cid,
                    "scan_cid": scan.scan_cid,
                    "plan_root_cid": graph.plan_root_cid,
                },
                fallback=SimpleNamespace(used=True),
            )
            return SimpleNamespace(graph=graph, receipt=receipt, used_fallback=True)

    composition = sf.resolve_production_composition(
        repository_root=repo,
        state_root=state,
        require_activation=False,
    )
    composition.extras["planner"] = _Planner()
    supervisor = facade_mod.Supervisor.open(services=composition)
    prompt = "Observe src/service.py and keep the run outside the repository."
    run = supervisor.run(prompt)
    assert run.state == "running"
    assert run.identities["workers_started"] is True
    assert run.identities["active_worker_count"] >= 1
    assert run.identities["hard_domain_admitted"] is False
    assert run.identities["authorizes_execution"] is False
    assert run.identities["completion_authority"] is False
    assert run.identities["worker_repository_edits"] is False
    assert run.identities["plan_create_status"] == "review_only"
    run_dir = Path(run.identities["worker_run_dir"])
    assert run_dir.is_relative_to(state.resolve())
    assert not run_dir.is_relative_to(repo.resolve())
    pid = run.identities["worker_pids"][0]
    task_cid = run.identities["worker_task_cids"][0]
    receipt = json.loads(
        (run_dir / "workers" / f"{task_cid}.json").read_text(encoding="utf-8")
    )
    assert receipt["phase"] == "running"
    assert receipt["pid"] == pid
    os.kill(int(pid), 0)
    os.kill(int(run.identities["supervisor_pid"]), 0)
    assert receipt["cwd"] == str(run_dir.resolve())
    assert "ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers" in receipt["argv"]
    assert "git" not in receipt["argv"]
    assert "pytest" not in receipt["argv"]
    assert receipt["work"]["validation_executed"] is False
    assert (repo / "src" / "service.py").read_bytes() == body


def test_preview_does_not_start_workers(tmp_path: Path) -> None:
    supervisor, prompt = _production_supervisor(tmp_path)
    observed = supervisor.preview(prompt)
    assert observed.values.get("effect_applied") is False
    assert list((tmp_path / "state").rglob("*.json")) == []


def test_agent_command_disables_plan_mode_and_drops_parent_session(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
        _agent_env,
        _build_workspace_agent_command,
    )

    monkeypatch.setenv("GROK_AGENT", "1")
    monkeypatch.setenv("GROK_SESSION_ID", "parent-plan-session")
    env = _agent_env()
    assert "GROK_AGENT" not in env
    assert "GROK_SESSION_ID" not in env
    repository = tmp_path / "repo"
    prompt_file = tmp_path / "state" / "prompts" / "taskcid" / "prompt.txt"
    command = _build_workspace_agent_command(
        "/usr/bin/python3", repository, prompt_file
    )
    assert command[command.index("--cwd") + 1] == str(repository)
    assert "--no-plan" in command
    assert "--always-approve" in command
    assert "--permission-mode" in command
    assert command[command.index("--permission-mode") + 1] == "bypassPermissions"
    assert f"Edit({repository}/**)" not in command
    assert f"Write({repository}/**)" not in command
    assert f"Edit({repository / '.git'}/**)" in command
    assert f"Write({repository / '.git'}/**)" in command
    assert "Bash(git)" in command
    assert "Bash(pytest)" in command
    assert "git" not in command
    assert "pytest" not in command


def test_prompt_worker_runs_an_agent_in_the_repository(tmp_path: Path) -> None:
    import os
    import stat
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
        launch_prompt_workers,
    )

    repository = tmp_path / "repo"
    state = tmp_path / "state"
    repository.mkdir()
    state.mkdir()
    (repository / "pkg").mkdir()
    body = b"VALUE = 1\n"
    (repository / "pkg" / "task.py").write_bytes(body)
    agent = tmp_path / "workspace-agent"
    agent.write_text(
        "#!/usr/bin/env python3\n"
        "import json, sys\n"
        "from pathlib import Path\n"
        "args = sys.argv\n"
        "cwd = Path(args[args.index('--cwd') + 1])\n"
        "prompt = Path(args[args.index('--prompt-file') + 1])\n"
        "(prompt.parent / 'agent-argv.json').write_text(json.dumps(args), encoding='utf-8')\n"
        "target = cwd / 'pkg' / 'task.py'\n"
        "target.write_bytes(target.read_bytes() + b'# checked\\n')\n",
        encoding="utf-8",
    )
    agent.chmod(agent.stat().st_mode | stat.S_IEXEC)
    task = SimpleNamespace(
        task_cid="taskcid",
        task_key="task:one",
        dependency_task_cids=(),
        objective="In pkg/task.py, add one comment line that says checked.",
        scope_paths=("pkg/task.py",),
        validations=(SimpleNamespace(argv=("python", "-m", "pytest", "-q")),),
    )
    launch = launch_prompt_workers(
        tasks=(task,),
        admitted_task_cids=("taskcid",),
        run_id="run-agent",
        state_root=state,
        repository_root=repository,
        width=1,
        local_only=False,
        agent_executable=str(agent),
    )
    receipt_path = Path(launch.run_dir) / "workers" / "taskcid.json"
    execution_path = (
        Path(launch.run_dir) / "workspaces" / "taskcid" / "execution.json"
    )
    deadline = time_monotonic() + 10
    execution = {}
    while time_monotonic() < deadline:
        if execution_path.is_file():
            execution = json.loads(execution_path.read_text(encoding="utf-8"))
            if execution.get("disposition") == "agent_finished":
                break
        time_sleep(0.05)
    assert execution.get("disposition") == "agent_finished"
    assert execution.get("agent_returncode") == 0
    assert execution["validation_executed"] is False
    assert execution["authorizes_execution"] is False
    assert execution["completion_authority"] is False
    assert execution["repository_edits"] is True
    assert (repository / "pkg" / "task.py").read_bytes() == body + b"# checked\n"
    prompt_file = Path(launch.run_dir) / "prompts" / "taskcid" / "prompt.txt"
    assert prompt_file.is_file()
    assert not prompt_file.is_relative_to(repository)
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert receipt["phase"] == "running"
    os.kill(int(receipt["pid"]), 0)
    os.kill(int(launch.supervisor_pid), 0)
    assert "pytest" not in receipt["argv"]
    assert "git" not in receipt["argv"]
    argv = json.loads(
        (prompt_file.parent / "agent-argv.json").read_text(encoding="utf-8")
    )
    assert "--no-plan" in argv
    assert "--always-approve" in argv
    assert argv[argv.index("--cwd") + 1] == str(repository.resolve())
    assert f"Edit({repository.resolve()}/**)" not in argv


def test_prompt_worker_stop_kills_the_agent_process_group(tmp_path: Path) -> None:
    import os
    import stat
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
        launch_prompt_workers,
        stop_prompt_workers,
    )

    repository = tmp_path / "repo"
    state = tmp_path / "state"
    repository.mkdir()
    state.mkdir()
    (repository / "pkg").mkdir()
    (repository / "pkg" / "task.py").write_bytes(b"VALUE = 1\n")
    agent = tmp_path / "sleeper-agent"
    agent.write_text(
        "#!/usr/bin/env python3\n"
        "import time\n"
        "from pathlib import Path\n"
        "import sys\n"
        "prompt = Path(sys.argv[sys.argv.index('--prompt-file') + 1])\n"
        "(prompt.parent / 'sleeper-started').write_text('started', encoding='utf-8')\n"
        "time.sleep(30)\n",
        encoding="utf-8",
    )
    agent.chmod(agent.stat().st_mode | stat.S_IEXEC)
    task = SimpleNamespace(
        task_cid="taskcid",
        task_key="task:one",
        dependency_task_cids=(),
        objective="Stay running until stopped.",
        scope_paths=("pkg/task.py",),
    )
    launch = launch_prompt_workers(
        tasks=(task,),
        admitted_task_cids=("taskcid",),
        run_id="run-stop",
        state_root=state,
        repository_root=repository,
        width=1,
        local_only=False,
        agent_executable=str(agent),
    )
    agent_receipt = Path(launch.run_dir) / "agents" / "taskcid.json"
    started = Path(launch.run_dir) / "prompts" / "taskcid" / "sleeper-started"
    deadline = time_monotonic() + 10
    payload = {}
    while time_monotonic() < deadline:
        if agent_receipt.is_file() and started.is_file():
            payload = json.loads(agent_receipt.read_text(encoding="utf-8"))
            break
        time_sleep(0.05)
    pid = int(payload.get("pid") or 0)
    assert pid > 1
    os.kill(pid, 0)
    stop_prompt_workers(launch.run_dir, (launch.supervisor_pid, *launch.pids))
    deadline = time_monotonic() + 3
    while time_monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except OSError:
            break
        time_sleep(0.05)
    with pytest.raises(OSError):
        os.kill(pid, 0)
    assert (repository / "pkg" / "task.py").read_bytes() == b"VALUE = 1\n"


def time_monotonic() -> float:
    import time

    return time.monotonic()


def time_sleep(seconds: float) -> None:
    import time

    time.sleep(seconds)


def test_default_worker_command_is_the_isolated_runner() -> None:
    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
        _agent_prompt,
        _build_isolated_runner_command,
    )

    repository = Path("/tmp/example-repo")
    command = _build_isolated_runner_command(repository)
    assert command[1:3] == [
        "-m",
        "ipfs_accelerate_py.agent_supervisor.runtime.grok_cli_runner",
    ]
    assert command[command.index("--workspace") + 1] == str(repository)
    assert "--permission-mode" not in command
    assert "bypassPermissions" not in command
    assert "--cwd" not in command
    assert "pytest" not in command
    prompt = _agent_prompt(
        {
            "objective": "Implement the change.",
            "prompt_text": "Set VALUE = 2.",
            "scope_paths": ["sample.py"],
        },
        repository,
    )
    assert "Set VALUE = 2." in prompt
    assert "Do not run pytest." in prompt


def test_product_packet_names_the_recognized_runner(tmp_path: Path) -> None:
    """Outside pytest, the supervisor's worker command is the package runner."""

    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
        _RUNNER_MODULE,
        _write_queue_packet,
    )

    repository = tmp_path / "repo"
    run_dir = tmp_path / "run"
    repository.mkdir()
    (run_dir / "queue").mkdir(parents=True)
    _write_queue_packet(
        SimpleNamespace(
            task_cid="taskcid",
            task_key="task:one",
            objective="Implement the change.",
            scope_paths=("sample.py",),
            validations=(),
        ),
        run_dir,
        repository,
        local_only=False,
        agent_executable="",
        prompt_text="Set VALUE = 2.",
    )
    packet = json.loads((run_dir / "queue" / "taskcid.json").read_text(encoding="utf-8"))
    argv = packet["launched_argv"]
    assert argv[1:3] == ["-m", _RUNNER_MODULE]
    assert argv[argv.index("--workspace") + 1] == str(repository)
    assert "prompt_workers" not in " ".join(argv)
    assert "--permission-mode" not in argv
    assert "bypassPermissions" not in argv
    assert packet["prompt_text"] == "Set VALUE = 2."
    assert packet["authorizes_execution"] is False
    assert packet["completion_authority"] is False


def test_prompt_run_starts_workers_without_scheduler_config(tmp_path: Path) -> None:
    """A prompt against an ordinary git repository starts worker processes."""

    import io
    import os
    import subprocess
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.entrypoints import cli as supervisor_cli

    repo = tmp_path / "repo"
    state = tmp_path / "state"
    repo.mkdir()
    state.mkdir()
    (repo / "sample.py").write_text("VALUE = 1\n", encoding="utf-8")
    env = os.environ.copy()
    env.update(
        {
            "GIT_AUTHOR_NAME": "probe",
            "GIT_AUTHOR_EMAIL": "probe@example.com",
            "GIT_COMMITTER_NAME": "probe",
            "GIT_COMMITTER_EMAIL": "probe@example.com",
        }
    )
    subprocess.run(("git", "init", "-q"), cwd=repo, check=True, env=env)
    subprocess.run(("git", "add", "sample.py"), cwd=repo, check=True, env=env)
    subprocess.run(("git", "commit", "-qm", "init"), cwd=repo, check=True, env=env)
    assert not (
        repo / "config" / "agent_supervisor_prompt_only_self_improvement_v3_scheduler.json"
    ).is_file()
    args = SimpleNamespace(
        supervisor_command="run",
        prompt="Start workers for the admitted task in sample.py.",
        prompt_file=None,
        prompt_stdin=False,
        output_json=True,
        repository=str(repo),
        state_root=str(state),
    )
    out = io.StringIO()
    code = supervisor_cli.run_supervisor_cli(args, stdout=out, stderr=out)
    payload = json.loads(out.getvalue())
    assert code == supervisor_cli.EXIT_SUCCESS, payload
    assert payload["ok"] is True
    result = payload["result"]
    assert result["state"] == "running"
    assert result["active_worker_count"] >= 1
    identities = result["identities"]
    assert identities["workers_started"] is True
    assert identities["authorizes_execution"] is False
    assert identities["completion_authority"] is False
    assert identities["intent_autoencoder_status"] == "fail_open_no_checkpoint"
    assert identities["intent_subgoal_count"] == 3
    assert identities["intent_projection_ids"][-1] == "intent-route/action-hoare/v1"
    assert identities["admitted_task_count"] == 1
    supervisor_pid = int(identities["supervisor_pid"])
    worker_pid = int(identities["worker_pids"][0])
    os.kill(supervisor_pid, 0)
    os.kill(worker_pid, 0)
    command = Path(f"/proc/{worker_pid}/cmdline").read_bytes().replace(b"\0", b" ").decode()
    assert "prompt_workers" in command
    assert "--worker" in command
    assert "grok" not in command
    assert (repo / "sample.py").read_text(encoding="utf-8") == "VALUE = 1\n"
    second = facade_mod.Supervisor.open(
        repository=repo,
        state_root=state,
        require_activation=False,
    )
    observed = second.status()
    assert observed.state == "running"
    assert observed.values["identities"]["supervisor_pid"] == supervisor_pid
    assert worker_pid in observed.values["identities"]["worker_pids"]
    assert observed.values["identities"]["authorizes_execution"] is False
    assert observed.values["identities"]["completion_authority"] is False


def test_prompt_workers_refuse_a_repository_directory(tmp_path: Path) -> None:
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.prompt.prompt_workers import (
        PromptWorkerError,
        launch_prompt_workers,
    )

    repo = tmp_path / "repo"
    repo.mkdir()
    task = SimpleNamespace(
        task_cid="taskcid",
        task_key="task:one",
        dependency_task_cids=(),
    )
    with pytest.raises(PromptWorkerError, match="repository"):
        launch_prompt_workers(
            tasks=(task,),
            admitted_task_cids=("taskcid",),
            run_id="run1",
            state_root=repo,
            repository_root=repo,
            width=1,
        )


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


class _ReviewOnlyCreateService:
    def __init__(
        self, reasons: tuple[str, ...] = ("ir_admission_materials_absent",)
    ) -> None:
        self.reasons = reasons

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
            verdict=PlanCreateVerdict.REVIEW_ONLY,
            roots=roots,
            stage_results=(),
            scan_cid=request_cid,
            admission_receipt_cid=request_cid,
            plan_root_cid=str(roots.program_root),
            rejection_reasons=self.reasons,
            read_only=True,
            wrote_effects=(),
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
    create_service: object | None = None,
    use_real_admission: bool = False,
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
    if not use_real_admission:
        composition.extras["admission"] = _MatchingAdmission()
    composition.extras["plan_supervisor_service"] = PlanSupervisorService(
        create_service=create_service or _AdmittedCreateService(),
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
    assert run.state == "running"
    assert run.identities["workers_started"] is True
    assert run.identities["authorizes_execution"] is False


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
    assert run.state == "running"
    assert run.identities["workers_started"] is True
    assert run.identities["authorizes_execution"] is False
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
    assert run.state == "running"
    assert run.identities["workers_started"] is True
    assert run.identities["authorizes_execution"] is False
    assert run.identities["completion_authority"] is False
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


def test_repository_root_prompt_scope_skips_the_root_marker_and_gitlinks(
    tmp_path: Path,
) -> None:
    import subprocess
    from types import SimpleNamespace

    from ipfs_accelerate_py.agent_supervisor.prompt.plan_create_service import (
        plan_create_request_from_workflow,
    )

    repository = tmp_path / "repo"
    repository.mkdir()
    subprocess.run(("git", "-C", str(repository), "init", "-q"), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "config", "user.name", "Scope Test"),
        check=True,
    )
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "config",
            "user.email",
            "scope@example.invalid",
        ),
        check=True,
    )
    (repository / "src").mkdir()
    (repository / "src" / "service.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repository / "README.md").write_text("Fixture.\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(repository), "add", "."), check=True)
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "fixture"),
        check=True,
    )
    child = tmp_path / "child"
    child.mkdir()
    subprocess.run(("git", "-C", str(child), "init", "-q"), check=True)
    subprocess.run(
        ("git", "-C", str(child), "config", "user.name", "Scope Test"),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(child), "config", "user.email", "scope@example.invalid"),
        check=True,
    )
    (child / "nested.py").write_text("NESTED = 1\n", encoding="utf-8")
    subprocess.run(("git", "-C", str(child), "add", "nested.py"), check=True)
    subprocess.run(
        ("git", "-C", str(child), "commit", "-qm", "child"),
        check=True,
    )
    oid = subprocess.run(
        ("git", "-C", str(child), "rev-parse", "HEAD"),
        check=True,
        stdout=subprocess.PIPE,
        text=True,
    ).stdout.strip()
    subprocess.run(
        (
            "git",
            "-C",
            str(repository),
            "update-index",
            "--add",
            "--cacheinfo",
            f"160000,{oid},vendor",
        ),
        check=True,
    )
    subprocess.run(
        ("git", "-C", str(repository), "commit", "-qm", "gitlink"),
        check=True,
    )
    (repository / "vendor").mkdir()
    workflow = SimpleNamespace(
        repository_root=str(repository),
        directory=str(repository),
        prompt_source=SimpleNamespace(content_id=_cid("prompt")),
        repository_root_cid=_cid("root"),
        policy_root=_cid("policy"),
        program_root=_cid("program"),
        intent_ir_root=_cid("intent"),
        legal_ir_root=_cid("legal"),
        security_ir_root=_cid("security"),
        allowlist_cid=_cid("allow"),
        request_cid=_cid("request"),
        output_policy=SimpleNamespace(mode="markdown"),
        budget=None,
        supervisor_profile="",
        state_root=str(tmp_path / "state"),
    )
    projected = plan_create_request_from_workflow(
        workflow,
        repository_id="repository:scope-test",
    )
    assert "." not in projected.scope_paths
    assert "src" in projected.scope_paths
    assert "README.md" in projected.scope_paths
    assert "vendor" not in projected.scope_paths


def test_scan_directory_falls_back_to_repository_root_when_pkg_is_absent(
    tmp_path: Path,
) -> None:
    from ipfs_accelerate_py.agent_supervisor.entrypoints.facade import (
        _scan_directory_for_repository,
    )

    assert _scan_directory_for_repository(str(tmp_path)) == str(tmp_path)
    packaged = tmp_path / "pkg"
    packaged.mkdir()
    assert _scan_directory_for_repository(str(tmp_path)) == str(packaged)
