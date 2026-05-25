from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
    record_codebase_scan_findings,
    record_objective_backlog_findings,
    record_retry_budget_findings,
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
    events_path.write_text(json.dumps(failure) + "\n" + json.dumps({**failure, "attempt": 2}) + "\n", encoding="utf-8")

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
