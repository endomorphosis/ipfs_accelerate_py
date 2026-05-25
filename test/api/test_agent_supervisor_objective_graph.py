from __future__ import annotations

import json
import subprocess
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import (
    build_bundle_task_payloads,
    generate_objective_todos,
    resolver_payload,
    scan_objective_gaps,
    submit_bundle_tasks,
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


def _seed_repo(tmp_path: Path) -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")

    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    source = repo / "src" / "runtime_router.py"
    notes = repo / "docs" / "runtime_notes.md"
    source.parent.mkdir()
    notes.parent.mkdir()
    source.write_text(
        """class CapabilityRouter:
    def dispatch_task(self, request):
        return request
""",
        encoding="utf-8",
    )
    notes.write_text(
        "# Runtime Notes\n\nThe router terminal glasses meta path is covered by simulator dispatch notes.\n",
        encoding="utf-8",
    )
    todo_path.write_text(
        """# Objective Todos

## ACCEL-001 Completed seed

- Status: completed
- Completion: manual
- Priority: P2
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: true
- Acceptance: Seed task.
""",
        encoding="utf-8",
    )
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G000 Virtual AI OS outcome

- Status: active
- Parent:
- Fib priority: 1
- Track: ops
- Priority: P0
- Bundle: objective/ops/root
- Goal: Prove the virtual AI OS.
- Evidence: CapabilityRouter.dispatch_task, meta glasses terminal router, missing_meta_glasses_contract
- Outputs: src, tests
- Validation: test -f objective-heap.md
- Gap task: Add the missing runtime proof.
""",
        encoding="utf-8",
    )
    _git(repo, "add", "todo.md", "objective-heap.md", "src/runtime_router.py", "docs/runtime_notes.md")
    _git(repo, "commit", "-m", "seed objective heap")
    return repo, objective_path, todo_path


def test_objective_graph_scanner_uses_ast_and_embedding_evidence(tmp_path):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)

    findings = scan_objective_gaps(repo, objective_path=objective_path, max_findings=1)

    assert len(findings) == 1
    finding = findings[0]
    assert finding.goal_id == "VAIOS-G000"
    assert finding.bundle_key == "objective/ops/root"
    assert finding.missing_evidence == ["missing_meta_glasses_contract"]
    assert finding.present_evidence["CapabilityRouter.dispatch_task"] == ["src/runtime_router.py (ast)"]
    assert finding.present_evidence["meta glasses terminal router"][0].startswith("docs/runtime_notes.md (embedding:")


def test_generate_objective_todos_writes_bundle_shards_and_payloads(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="ACCEL-",
        max_findings=1,
    )

    assert len(records) == 1
    assert records[0].task_id == "ACCEL-002"
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## ACCEL-002 Close virtual AI OS objective gap" in todo_text
    assert "- Bundle: objective/ops/root" in todo_text

    shard = bundle_dir / "objective-ops-root.todo.md"
    assert shard.exists()
    assert "## ACCEL-002 Close virtual AI OS objective gap" in shard.read_text(encoding="utf-8")
    index_path = bundle_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["bundles"]["objective/ops/root"]["tasks"][0]["task_id"] == "ACCEL-002"

    payloads = build_bundle_task_payloads(index_path)
    assert payloads[0]["bundle_key"] == "objective/ops/root"
    assert payloads[0]["todo_path"].endswith("objective-ops-root.todo.md")

    submitted: list[dict[str, object]] = []

    class FakeQueue:
        def submit(self, **kwargs):
            submitted.append(kwargs)
            return "queued-1"

    task_ids = submit_bundle_tasks(index_path, queue=FakeQueue())

    assert task_ids == ["queued-1"]
    assert submitted[0]["task_type"] == "codex.todo_bundle"
    assert submitted[0]["payload"]["bundle_key"] == "objective/ops/root"


def test_merge_resolver_builds_dry_run_payload(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "type": "merge_reconciled",
                "task_id": "ACCEL-009",
                "attempt": 2,
                "resolved": False,
                "merge_result": {
                    "attempted": True,
                    "merged": False,
                    "branch": "implementation/accel-009",
                    "target_branch": "main",
                    "command": ["git", "merge", "--no-ff", "implementation/accel-009"],
                    "reason": "content_conflict",
                    "dirty_paths": ["ipfs_accelerate_py/agent_supervisor"],
                    "stderr": "CONFLICT (content): Merge conflict",
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = resolver_payload(events_path=events_path, repo_root=repo, task_id="ACCEL-009")

    assert payload["found"] is True
    assert payload["task_id"] == "ACCEL-009"
    assert payload["branch"] == "implementation/accel-009"
    assert "Resolve the autonomous-agent supervisor merge conflict" in payload["prompt"]
    assert "ipfs_accelerate_py/agent_supervisor" in payload["prompt"]
