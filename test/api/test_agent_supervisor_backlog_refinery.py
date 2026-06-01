from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
    ConfiguredRetryBudgetRecorder,
    iter_jsonl,
    ensure_task_blocks_present,
    load_strategy,
    release_completed_guardrail_blocks,
    record_codebase_scan_findings,
    record_configured_retry_budget_findings,
    record_dependency_guardrail_findings,
    record_objective_backlog_findings,
    record_retry_budget_findings,
    scan_codebase_findings,
)


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _seed_repo(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    return repo


def _write_todo(path: Path) -> None:
    path.write_text(
        """# Agent Todos

## AUTO-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )


def test_backlog_refinery_appends_missing_task_blocks_in_order(tmp_path):
    todo_path = tmp_path / "tasks.todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Existing task

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: README.md
- Validation: test -f README.md
- Acceptance: Existing task.
""",
        encoding="utf-8",
    )

    changed = ensure_task_blocks_present(
        todo_path,
        (
            ("AUTO-001", "## AUTO-001 Duplicate task\n\n- Status: todo"),
            ("AUTO-002", "## AUTO-002 First appended task\n\n- Status: todo"),
            ("AUTO-003", "## AUTO-003 Second appended task\n\n- Status: todo"),
        ),
    )

    updated = todo_path.read_text(encoding="utf-8")
    assert changed
    assert "Duplicate task" not in updated
    assert updated.index("## AUTO-002 First appended task") < updated.index("## AUTO-003 Second appended task")
    assert not ensure_task_blocks_present(todo_path, (("AUTO-002", "## AUTO-002 First appended task"),))


def test_backlog_refinery_codebase_scan_refills_low_backlog(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text(
        """def route_request(request):
    # TODO: prove routing retry behavior
    return request
""",
        encoding="utf-8",
    )
    _write_todo(todo_path)
    _git(repo, "add", "todo.md", "src/runtime.py")
    _git(repo, "commit", "-m", "seed repo")

    findings = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        repo_root=repo,
        task_prefix="AUTO-",
        max_findings=1,
        force=True,
    )

    assert len(findings) == 1
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve code annotation in src/runtime.py:2" in todo_text
    assert "- Validation: python3 -m py_compile src/runtime.py" in todo_text
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["last_codebase_scan_findings"][0]["follow_up_task_id"] == "AUTO-002"
    assert Path(findings[0]["discovery_path"]).exists()


def test_backlog_refinery_codebase_scan_skips_vanished_git_roots(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor import backlog_refinery

    repo = _seed_repo(tmp_path)
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text(
        """def route_request(request):
    # TODO: prove vanished roots do not crash scanning
    return request
""",
        encoding="utf-8",
    )
    _git(repo, "add", "src/runtime.py")
    _git(repo, "commit", "-m", "seed scan target")
    vanished = tmp_path / "deleted-worktree"
    vanished.mkdir()
    vanished.rmdir()
    monkeypatch.setattr(backlog_refinery, "discover_git_worktrees", lambda *_args, **_kwargs: [repo, vanished])

    findings = scan_codebase_findings(repo, max_findings=5)

    assert [finding.root_relative_path for finding in findings] == ["src/runtime.py"]


def test_backlog_refinery_repairs_invalid_strategy_file(tmp_path):
    repo = _seed_repo(tmp_path)
    strategy_path = repo / "state" / "strategy.json"
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text("{not json", encoding="utf-8")

    strategy = load_strategy(strategy_path)

    assert strategy["blocked_tasks"] == []
    assert strategy["last_strategy_repair_reason"] == "invalid_strategy_json"
    repaired = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert repaired["blocked_tasks"] == []
    assert repaired["last_strategy_repair_reason"] == "invalid_strategy_json"


def test_backlog_refinery_iter_jsonl_quarantines_malformed_events(tmp_path):
    events_path = tmp_path / "events.jsonl"
    valid_event = {"type": "implementation_finished", "task_id": "AUTO-001"}
    events_path.write_text(
        json.dumps(valid_event) + "\nnot json\n[1, 2, 3]\n",
        encoding="utf-8",
    )

    events = iter_jsonl(events_path)

    assert events == [valid_event]
    repaired_lines = events_path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line) for line in repaired_lines] == [valid_event]
    quarantines = list(tmp_path.glob("events.jsonl.invalid-jsonl-*"))
    assert len(quarantines) == 1
    assert "not json" in quarantines[0].read_text(encoding="utf-8")


def test_backlog_refinery_dependency_guardrail_adds_ready_repair_task(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Waiting on missing prerequisite

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: AUTO-999
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: This task cannot become ready until dependencies resolve.
""",
        encoding="utf-8",
    )

    findings = record_dependency_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_prefix="AUTO-",
        max_findings=1,
        repo_root=repo,
    )

    assert len(findings) == 1
    assert findings[0]["source_task_id"] == "AUTO-001"
    assert findings[0]["missing_dependencies"] == ["AUTO-999"]
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve dependency guardrail for AUTO-001" in todo_text
    assert "AUTO-999" in Path(findings[0]["discovery_path"]).read_text(encoding="utf-8")
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]
    assert strategy["dependency_guardrail_findings"][0]["follow_up_task_id"] == "AUTO-002"

    repeated = record_dependency_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_prefix="AUTO-",
        max_findings=1,
        repo_root=repo,
    )
    assert repeated == []
    assert todo_path.read_text(encoding="utf-8").count("Resolve dependency guardrail for AUTO-001") == 1


def test_backlog_refinery_dependency_guardrail_detects_dependency_cycle(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Waiting on cycle entry

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: AUTO-002
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: This task cannot become ready until dependencies resolve.

## AUTO-002 Waiting on cycle return

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: AUTO-001
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: This task cannot become ready until dependencies resolve.
""",
        encoding="utf-8",
    )

    findings = record_dependency_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_prefix="AUTO-",
        max_findings=1,
        repo_root=repo,
    )

    assert len(findings) == 1
    assert findings[0]["source_task_id"] == "AUTO-001"
    assert findings[0]["dependency_cycle"] == ["AUTO-001", "AUTO-002", "AUTO-001"]
    discovery = Path(findings[0]["discovery_path"]).read_text(encoding="utf-8")
    assert "Dependency cycle: AUTO-001 -> AUTO-002 -> AUTO-001" in discovery
    assert "## AUTO-003 Resolve dependency guardrail for AUTO-001" in todo_path.read_text(encoding="utf-8")
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]


def test_backlog_refinery_dependency_guardrail_detects_duplicate_task_ids(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 First duplicate

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: First task.

## AUTO-001 Second duplicate

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Second task.
""",
        encoding="utf-8",
    )

    findings = record_dependency_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_prefix="AUTO-",
        max_findings=1,
        repo_root=repo,
    )

    assert len(findings) == 1
    assert findings[0]["source_task_id"] == "AUTO-001"
    assert findings[0]["duplicate_task_id"] == "AUTO-001"
    assert findings[0]["duplicate_task_lines"] == [3, 14]
    discovery = Path(findings[0]["discovery_path"]).read_text(encoding="utf-8")
    assert "Duplicate task id: AUTO-001" in discovery
    assert "Duplicate source lines: 3, 14" in discovery
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve dependency guardrail for AUTO-001" in todo_text
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]

    repeated = record_dependency_guardrail_findings(
        todo_path=todo_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_prefix="AUTO-",
        max_findings=1,
        repo_root=repo,
    )
    assert repeated == []
    assert todo_path.read_text(encoding="utf-8").count("Resolve dependency guardrail for AUTO-001") == 1


def test_backlog_refinery_releases_completed_guardrail_block(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Original task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Original blocked task.

## AUTO-002 Resolve implementation retry-budget failure for AUTO-001

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Follow-up repair completed.
""",
        encoding="utf-8",
    )
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["AUTO-001", "AUTO-999"],
                "retry_budget_findings": [
                    {
                        "source_task_id": "AUTO-001",
                        "follow_up_task_id": "AUTO-002",
                        "failure_kind": "implementation",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    releases = release_completed_guardrail_blocks(
        todo_path=todo_path,
        strategy_path=strategy_path,
        task_prefix="AUTO-",
    )

    assert releases == [
        {
            "source_task_id": "AUTO-001",
            "follow_up_task_id": "AUTO-002",
            "guardrail_kind": "retry_budget",
        },
        {
            "source_task_id": "AUTO-999",
            "follow_up_task_id": "",
            "guardrail_kind": "stale_strategy_block",
            "reason": "missing_task",
        }
    ]
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == []
    assert strategy["guardrail_unblock_releases"] == releases


def test_backlog_refinery_releases_completed_and_duplicate_stale_strategy_blocks(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Completed source

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Source task already completed.

## AUTO-002 Still blocked intentionally

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/other.py
- Validation: test -f todo.md
- Acceptance: This task remains blocked.
""",
        encoding="utf-8",
    )
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps({"blocked_tasks": ["AUTO-001", "AUTO-002", "AUTO-002"]}),
        encoding="utf-8",
    )

    releases = release_completed_guardrail_blocks(
        todo_path=todo_path,
        strategy_path=strategy_path,
        task_prefix="AUTO-",
    )

    assert releases == [
        {
            "source_task_id": "AUTO-002",
            "follow_up_task_id": "",
            "guardrail_kind": "stale_strategy_block",
            "reason": "duplicate_strategy_block",
        },
        {
            "source_task_id": "AUTO-001",
            "follow_up_task_id": "",
            "guardrail_kind": "stale_strategy_block",
            "reason": "source_completed",
        },
    ]
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-002"]


def test_backlog_refinery_retry_budget_blocks_validation_loop(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    events_path = repo / "state" / "events.jsonl"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Fix validation

- Status: todo
- Completion: manual
- Priority: P1
- Track: runtime
- Depends on:
- Outputs: src/runtime.py
- Validation: pytest tests/test_runtime.py
- Acceptance: Fix the runtime validation failure.
""",
        encoding="utf-8",
    )
    events_path.parent.mkdir(parents=True)
    failure = {
        "type": "implementation_finished",
        "task_id": "AUTO-001",
        "attempt": 1,
        "validation_result": {
            "attempted": True,
            "passed": False,
            "failed_command": "pytest tests/test_runtime.py",
        },
        "log_path": "state/implementation_logs/auto-001-attempt-1.log",
    }
    events_path.write_text(
        json.dumps(failure) + "\n" + json.dumps({**failure, "attempt": 2}) + "\n",
        encoding="utf-8",
    )

    findings = record_retry_budget_findings(
        todo_path=todo_path,
        events_path=events_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_header_prefix_value="## AUTO-",
        task_prefix="AUTO-",
        validation_retry_budget=2,
        merge_retry_budget=0,
    )

    assert len(findings) == 1
    assert findings[0]["failure_kind"] == "validation"
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve validation retry-budget failure for AUTO-001" in todo_text
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]
    assert Path(findings[0]["discovery_path"]).exists()


def test_backlog_refinery_configured_retry_budget_adds_present_dependency(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    events_path = repo / "state" / "events.jsonl"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Fix validation

- Status: todo
- Completion: manual
- Priority: P1
- Track: runtime
- Depends on:
- Outputs: src/runtime.py
- Validation: pytest tests/test_runtime.py
- Acceptance: Fix the runtime validation failure.

## AUTO-002 Guard retry loops

- Status: completed
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: scripts/supervisor.py
- Validation: true
- Acceptance: Guardrails are installed.
""",
        encoding="utf-8",
    )
    events_path.parent.mkdir(parents=True)
    failure = {
        "type": "implementation_finished",
        "task_id": "AUTO-001",
        "attempt": 1,
        "validation_result": {
            "attempted": True,
            "passed": False,
            "failed_command": "pytest tests/test_runtime.py",
        },
        "log_path": "state/implementation_logs/auto-001-attempt-1.log",
    }
    events_path.write_text(json.dumps(failure) + "\n" + json.dumps({**failure, "attempt": 2}) + "\n", encoding="utf-8")

    findings = record_configured_retry_budget_findings(
        todo_path=todo_path,
        events_path=events_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_header_prefix_value="## AUTO-",
        validation_retry_budget=2,
        merge_retry_budget=0,
        implementation_retry_budget=0,
        validation_depends_on_if_present=("AUTO-002", "AUTO-999"),
        validation_task_command_transform=lambda command: f"env TOOL=1 {command}",
        strip_validation_failure_kind=True,
        repo_root=repo,
    )

    assert len(findings) == 1
    assert "failure_kind" not in findings[0]
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-003 Resolve validation retry-budget failure for AUTO-001" in todo_text
    assert "- Depends on: AUTO-002" in todo_text
    assert "- Validation: env TOOL=1 pytest tests/test_runtime.py" in todo_text


def test_backlog_refinery_configured_retry_budget_recorder_uses_aliases(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    events_path = repo / "state" / "events.jsonl"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Fix validation

- Status: todo
- Completion: manual
- Priority: P1
- Track: runtime
- Depends on:
- Outputs: src/runtime.py
- Validation: pytest tests/test_runtime.py
- Acceptance: Fix the runtime validation failure.
""",
        encoding="utf-8",
    )
    events_path.parent.mkdir(parents=True)
    failure = {
        "type": "implementation_finished",
        "task_id": "AUTO-001",
        "attempt": 1,
        "validation_result": {
            "attempted": True,
            "passed": False,
            "failed_command": "pytest tests/test_runtime.py",
        },
        "log_path": "state/implementation_logs/auto-001-attempt-1.log",
    }
    events_path.write_text(json.dumps(failure) + "\n" + json.dumps({**failure, "attempt": 2}) + "\n", encoding="utf-8")
    prepared: list[str] = []

    recorder = ConfiguredRetryBudgetRecorder(
        todo_path=todo_path,
        events_path=events_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_header_prefix_value="## AUTO-",
        validation_retry_budget=5,
        merge_retry_budget=0,
        implementation_retry_budget=0,
        validation_task_command_transform=lambda command: f"env TOOL=1 {command}",
        prepare_environment=lambda: prepared.append("prepared"),
    )

    findings = recorder(retry_budget=2)

    assert prepared == ["prepared"]
    assert len(findings) == 1
    assert findings[0]["follow_up_task_id"] == "AUTO-002"
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "- Validation: env TOOL=1 pytest tests/test_runtime.py" in todo_text


def test_backlog_refinery_retry_budget_blocks_implementation_loop(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    events_path = repo / "state" / "events.jsonl"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Fix setup

- Status: todo
- Completion: manual
- Priority: P1
- Track: runtime
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f src/runtime.py
- Acceptance: Fix the implementation setup failure.
""",
        encoding="utf-8",
    )
    events_path.parent.mkdir(parents=True)
    failure = {
        "type": "implementation_finished",
        "task_id": "AUTO-001",
        "attempt": 1,
        "returncode": 1,
        "validation_result": {"attempted": False, "passed": True},
        "merge_result": {"attempted": False, "merged": False, "reason": "not_attempted"},
        "exception_result": {
            "exception_type": "RuntimeError",
            "phase": "worktree_setup",
            "message": "not a git repository",
        },
        "log_path": "state/implementation_logs/auto-001-attempt-1.log",
        "worktree_path": "worktrees/auto-001-attempt-1",
        "branch": "implementation/auto-001-attempt-1",
    }
    events_path.write_text(json.dumps(failure) + "\n" + json.dumps({**failure, "attempt": 2}) + "\n", encoding="utf-8")

    findings = record_retry_budget_findings(
        todo_path=todo_path,
        events_path=events_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_header_prefix_value="## AUTO-",
        task_prefix="AUTO-",
        implementation_retry_budget=2,
        validation_retry_budget=0,
        merge_retry_budget=0,
    )

    assert len(findings) == 1
    assert findings[0]["failure_kind"] == "implementation"
    assert findings[0]["failed_command"] == "implementation_exception:RuntimeError"
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve implementation retry-budget failure for AUTO-001" in todo_text
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]
    discovery_text = Path(findings[0]["discovery_path"]).read_text(encoding="utf-8")
    assert "Exception type: `RuntimeError`" in discovery_text


def test_backlog_refinery_retry_budget_blocks_merge_loop(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    events_path = repo / "state" / "events.jsonl"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Merge feature

- Status: todo
- Completion: manual
- Priority: P1
- Track: runtime
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f src/runtime.py
- Acceptance: Merge the generated runtime feature.
""",
        encoding="utf-8",
    )
    merge_result = {
        "attempted": True,
        "merged": False,
        "reason": "main_checkout_dirty_conflict",
        "command": ["git", "merge", "implementation/auto-001"],
    }
    event = {
        "type": "implementation_finished",
        "task_id": "AUTO-001",
        "attempt": 1,
        "validation_result": {"attempted": True, "passed": True},
        "merge_result": merge_result,
        "log_path": "state/implementation_logs/auto-001-attempt-1.log",
    }
    events_path.parent.mkdir(parents=True)
    events_path.write_text(json.dumps(event) + "\n" + json.dumps({**event, "attempt": 2}) + "\n", encoding="utf-8")

    findings = record_retry_budget_findings(
        todo_path=todo_path,
        events_path=events_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_header_prefix_value="## AUTO-",
        task_prefix="AUTO-",
        validation_retry_budget=0,
        merge_retry_budget=2,
    )

    assert len(findings) == 1
    assert findings[0]["failure_kind"] == "merge"
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve merge retry-budget failure for AUTO-001" in todo_text
    assert "ipfs-accelerate-agent-merge-resolver" in todo_text
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]


def test_backlog_refinery_retry_budget_blocks_merge_reconcile_skips(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    events_path = repo / "state" / "events.jsonl"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Merge missing branch

- Status: todo
- Completion: manual
- Priority: P1
- Track: runtime
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f src/runtime.py
- Acceptance: Recover the missing implementation branch.
""",
        encoding="utf-8",
    )
    event = {
        "type": "merge_reconcile_skipped",
        "task_id": "AUTO-001",
        "attempt": 1,
        "branch": "implementation/auto-001",
        "implementation_commit": "abc123",
        "resolved": False,
        "reason": "implementation_branch_missing",
    }
    events_path.parent.mkdir(parents=True)
    events_path.write_text(json.dumps(event) + "\n" + json.dumps({**event, "attempt": 2}) + "\n", encoding="utf-8")

    findings = record_retry_budget_findings(
        todo_path=todo_path,
        events_path=events_path,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        task_header_prefix_value="## AUTO-",
        task_prefix="AUTO-",
        implementation_retry_budget=0,
        validation_retry_budget=0,
        merge_retry_budget=2,
    )

    assert len(findings) == 1
    assert findings[0]["failure_kind"] == "merge"
    assert findings[0]["failed_command"] == "git merge (implementation_branch_missing)"
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Resolve merge retry-budget failure for AUTO-001" in todo_text
    discovery_text = Path(findings[0]["discovery_path"]).read_text(encoding="utf-8")
    assert "Merge reason: `implementation_branch_missing`" in discovery_text
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]


def test_backlog_refinery_objective_scan_refills_low_backlog(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    objective_path = repo / "objective-heap.md"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    source = repo / "src" / "capability_router.py"
    source.parent.mkdir()
    source.write_text(
        """class CapabilityRouter:
    def dispatch_task(self, request):
        return request
""",
        encoding="utf-8",
    )
    _write_todo(todo_path)
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G001 Capability routing contract

- Status: active
- Parent:
- Fib priority: 1
- Track: runtime
- Priority: P1
- Goal: Prove capability routing dispatch contracts.
- Evidence: CapabilityRouter.dispatch_task, missing capability route contract
- Outputs: src/capability_router.py, tests
- Validation: test -f objective-heap.md
- AST query: CapabilityRouter.dispatch_task
- Embedding query: capability routing dispatch contract
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md", "objective-heap.md", "src/capability_router.py")
    _git(repo, "commit", "-m", "seed objective repo")

    findings = record_objective_backlog_findings(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        strategy_path=strategy_path,
        task_prefix="AUTO-",
        max_findings=1,
        force=True,
        persist_ast_dataset=False,
    )

    assert len(findings) == 1
    assert findings[0]["goal_id"] == "VAIOS-G001"
    assert findings[0]["bundle_strategy"] == "semantic_ast"
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## AUTO-002 Close objective gap" in todo_text
    assert "- Bundle strategy: semantic_ast" in todo_text
    assert (bundle_dir / "index.json").exists()
