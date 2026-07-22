from __future__ import annotations

import subprocess
import threading
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.validation_commands import (
    ValidationStage,
    build_validation_commands,
    select_validation_commands,
)
from ipfs_accelerate_py.agent_supervisor.validation_scheduler import (
    ValidationResultCache,
    ValidationScheduler,
    build_validation_cache_key,
    collect_dependency_state,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


def _result(spec, *, returncode: int = 0) -> dict[str, object]:
    return {
        "command": spec.command,
        "raw_command": spec.raw_command,
        "started_at": "2026-01-01T00:00:00+00:00",
        "finished_at": "2026-01-01T00:00:01+00:00",
        "returncode": returncode,
        "output": f"output:{spec.command}",
    }


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=cwd, text=True, capture_output=True, check=True
    )
    return result.stdout.strip()


def _repo(path: Path) -> str:
    path.mkdir()
    _git(path, "init", "-q")
    _git(path, "config", "user.name", "Validation Test")
    _git(path, "config", "user.email", "validation@example.invalid")
    (path / "pyproject.toml").write_text("[project]\nname='fixture'\nversion='1'\n", encoding="utf-8")
    (path / "src").mkdir()
    (path / "src" / "alpha.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(path, "add", "-A")
    _git(path, "commit", "-qm", "baseline")
    return _git(path, "rev-parse", "HEAD")


def test_cheap_checks_run_before_expensive_tests_and_fail_fast(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(*, spec, **_kwargs):
        calls.append(spec.command)
        return _result(spec, returncode=7 if spec.stage == ValidationStage.CHEAP else 0)

    scheduler = ValidationScheduler(max_workers=2, resource_budget=2, runner=runner)
    report = scheduler.run(
        ["pytest tests/test_alpha.py", "git diff --check"],
        workspace_path=tmp_path,
        changed_files=["src/alpha.py"],
        target_commit="abc",
        dependency_state="deps",
    )

    assert calls == ["git diff --check"]
    assert report["passed"] is False
    assert report["returncode"] == 7
    assert report["failed_command"] == "git diff --check"
    assert [item["stage"] for item in report["stages"]] == ["cheap"]


def test_independent_validations_run_in_parallel_under_weighted_budget(tmp_path: Path) -> None:
    lock = threading.Lock()
    release = threading.Event()
    two_running = threading.Event()
    active = 0
    maximum_active = 0

    def runner(*, spec, **_kwargs):
        nonlocal active, maximum_active
        with lock:
            active += 1
            maximum_active = max(maximum_active, active)
            if active == 2:
                two_running.set()
        assert release.wait(timeout=5)
        with lock:
            active -= 1
        return _result(spec)

    scheduler = ValidationScheduler(max_workers=3, resource_budget=2, runner=runner)
    commands = [
        "pytest tests/test_alpha.py",
        "pytest tests/test_beta.py",
        "pytest tests/test_gamma.py",
    ]
    outcome: dict[str, object] = {}

    def schedule() -> None:
        outcome.update(
            scheduler.run(
                commands,
                workspace_path=tmp_path,
                changed_files=["pyproject.toml"],
                target_commit="abc",
                dependency_state="deps",
            )
        )

    thread = threading.Thread(target=schedule)
    thread.start()
    assert two_running.wait(timeout=5)
    release.set()
    thread.join(timeout=5)

    assert not thread.is_alive()
    assert outcome["passed"] is True
    assert maximum_active == 2
    assert [item["command"] for item in outcome["results"]] == commands


def test_cache_key_includes_commit_command_relevant_environment_and_dependencies() -> None:
    base = build_validation_cache_key(
        target_commit="commit-a",
        command="pytest tests/test_alpha.py",
        environment={"PYTHONPATH": "src", "IGNORED_SECRET": "one"},
        dependency_state={"lock": "one"},
    )
    same = build_validation_cache_key(
        target_commit="commit-a",
        command="pytest tests/test_alpha.py",
        environment={"IGNORED_SECRET": "two", "PYTHONPATH": "src"},
        dependency_state={"lock": "one"},
    )

    assert base.digest == same.digest
    variants = [
        build_validation_cache_key(
            target_commit="commit-b",
            command="pytest tests/test_alpha.py",
            environment={"PYTHONPATH": "src"},
            dependency_state={"lock": "one"},
        ),
        build_validation_cache_key(
            target_commit="commit-a",
            command="pytest tests/test_beta.py",
            environment={"PYTHONPATH": "src"},
            dependency_state={"lock": "one"},
        ),
        build_validation_cache_key(
            target_commit="commit-a",
            command="pytest tests/test_alpha.py",
            environment={"PYTHONPATH": "lib"},
            dependency_state={"lock": "one"},
        ),
        build_validation_cache_key(
            target_commit="commit-a",
            command="pytest tests/test_alpha.py",
            environment={"PYTHONPATH": "src"},
            dependency_state={"lock": "two"},
        ),
    ]
    assert all(item.digest != base.digest for item in variants)
    assert base.to_dict()["target_commit"] == "commit-a"


def test_success_cache_is_durable_and_dirty_or_dependency_content_invalidates(
    tmp_path: Path,
) -> None:
    repo = tmp_path / "repo"
    commit = _repo(repo)
    calls = 0

    def runner(*, spec, **_kwargs):
        nonlocal calls
        calls += 1
        return _result(spec)

    cache_dir = tmp_path / "cache"
    command = "pytest tests/test_alpha.py"
    scheduler = ValidationScheduler(cache_dir=cache_dir, runner=runner)
    first = scheduler.run(
        [command],
        workspace_path=repo,
        changed_files=["src/alpha.py"],
        target_commit=commit,
    )
    second = ValidationScheduler(cache_dir=cache_dir, runner=runner).run(
        [command],
        workspace_path=repo,
        changed_files=["src/alpha.py"],
        target_commit=commit,
    )

    assert calls == 1
    assert first["cache_misses"] == 1
    assert second["cache_hits"] == 1

    (repo / "src" / "alpha.py").write_text("VALUE = 2\n", encoding="utf-8")
    dirty = ValidationScheduler(cache_dir=cache_dir, runner=runner).run(
        [command],
        workspace_path=repo,
        changed_files=["src/alpha.py"],
        target_commit=commit,
    )
    assert calls == 2
    assert dirty["cache_hits"] == 0

    (repo / "pyproject.toml").write_text(
        "[project]\nname='fixture'\nversion='2'\n", encoding="utf-8"
    )
    dependency_changed = ValidationScheduler(cache_dir=cache_dir, runner=runner).run(
        [command],
        workspace_path=repo,
        changed_files=["src/alpha.py"],
        target_commit=commit,
    )
    assert calls == 3
    assert dependency_changed["cache_hits"] == 0


def test_failures_are_not_cached(tmp_path: Path) -> None:
    spec = build_validation_commands(["git diff --check"])[0]
    cache = ValidationResultCache(tmp_path / "cache")
    key = build_validation_cache_key(
        target_commit="abc", command=spec, dependency_state="deps", environment={}
    )

    assert cache.put(key, {"returncode": 1}) is False
    assert cache.get(key) is None


def test_impact_selection_is_explainable_and_dependency_changes_are_conservative() -> None:
    commands = [
        "git diff --check",
        "pytest tests/test_alpha.py",
        "pytest tests/test_beta.py",
        "custom-validation --all",
    ]
    narrow = select_validation_commands(commands, ["src/alpha.py"])
    decisions = {item.spec.command: item for item in narrow.items}

    assert decisions["git diff --check"].selected is True
    assert decisions["pytest tests/test_alpha.py"].selected is True
    assert decisions["pytest tests/test_alpha.py"].matched_paths == ("src/alpha.py",)
    assert decisions["pytest tests/test_beta.py"].selected is False
    assert decisions["pytest tests/test_beta.py"].reason == "no_changed_path_matches_command_target"
    assert decisions["custom-validation --all"].selected is True
    assert decisions["custom-validation --all"].reason == "global_or_unknown_impact"

    broad = select_validation_commands(commands, ["pyproject.toml"])
    assert all(item.selected for item in broad.items)
    assert broad.to_dict()["changed_files"] == ["pyproject.toml"]

    prefixed_test = select_validation_commands(
        ["pytest test/api/test_agent_supervisor_validation_scheduler.py"],
        ["ipfs_accelerate_py/agent_supervisor/validation_scheduler.py"],
    )
    assert prefixed_test.items[0].selected is True
    assert prefixed_test.items[0].reason == "changed_path_matches_command_target"

    ci_change = select_validation_commands(commands, [".github/workflows/test.yml"])
    assert all(item.selected for item in ci_change.items)


def test_pre_merge_escalation_runs_unrelated_targeted_validation(tmp_path: Path) -> None:
    calls: list[str] = []

    def runner(*, spec, **_kwargs):
        calls.append(spec.command)
        return _result(spec, returncode=9 if "beta" in spec.command else 0)

    report = ValidationScheduler(max_workers=2, resource_budget=2, runner=runner).run(
        ["pytest tests/test_alpha.py", "pytest tests/test_beta.py"],
        workspace_path=tmp_path,
        changed_files=["src/alpha.py"],
        target_commit="abc",
        dependency_state="deps",
        require_full_validation=True,
        scope="pre_merge",
    )

    assert set(calls) == {"pytest tests/test_alpha.py", "pytest tests/test_beta.py"}
    assert report["passed"] is False
    assert report["selection"]["escalated"] is True
    beta = next(
        item for item in report["selection"]["decisions"] if "beta" in item["command"]
    )
    assert beta["reason"] == "pre_merge_broad_escalation"
    assert beta["stage"] == "broad"


def test_dependency_state_records_candidate_content_not_only_head(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    _repo(repo)
    before = collect_dependency_state(repo, changed_files=["src/alpha.py"])
    (repo / "src" / "alpha.py").write_text("VALUE = 99\n", encoding="utf-8")
    after = collect_dependency_state(repo, changed_files=["src/alpha.py"])

    assert before["candidate_content_sha256"] != after["candidate_content_sha256"]


def test_daemon_uses_full_pre_merge_scope_and_preserves_result_contract(tmp_path: Path) -> None:
    captured: dict[str, object] = {}

    class Scheduler:
        def run(self, commands, **kwargs):
            captured["commands"] = list(commands)
            captured.update(kwargs)
            return {
                "attempted": True,
                "passed": False,
                "returncode": 6,
                "results": [
                    {
                        "command": commands[0],
                        "returncode": 6,
                        "stage": "cheap",
                        "output": "failed\n",
                    }
                ],
                "failed_command": commands[0],
                "selection": {"scope": "pre_merge", "changed_files": ["src/a.py"]},
            }

    daemon = TodoImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=tmp_path / "state.json",
        strategy_path=tmp_path / "strategy.json",
        events_path=tmp_path / "events.jsonl",
        repo_root=tmp_path,
        validation_scheduler=Scheduler(),  # type: ignore[arg-type]
    )
    task = PortalTask(
        task_id="REF-043",
        title="validation scheduler",
        status="todo",
        completion="manual",
        priority="P1",
        track="g9",
        validation=["git diff --check"],
    )

    report = daemon._run_validation_commands(tmp_path, task, tmp_path / "validation.log")

    assert captured["commands"] == ["git diff --check"]
    assert captured["require_full_validation"] is True
    assert captured["scope"] == "pre_merge"
    assert callable(captured["runner"])
    assert report["attempted"] is True
    assert report["passed"] is False
    assert report["returncode"] == 6
    assert report["failed_command"] == "git diff --check"
