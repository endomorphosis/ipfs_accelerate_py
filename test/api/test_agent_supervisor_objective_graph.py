from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import (
    build_bundle_task_payloads,
    generate_objective_todos,
    persist_objective_ast_dataset,
    resolver_payload,
    scan_objective_gaps,
    submit_bundle_tasks,
)
from ipfs_accelerate_py.agent_supervisor.merge_resolver import main as merge_resolver_main


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
    dataset_manifest = bundle_dir.parent / "objective_datasets" / "accel-objective-ast.manifest.json"
    assert dataset_manifest.exists()
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    assert dataset_payload["row_count"] >= 2
    assert Path(dataset_payload["jsonl_path"]).exists()

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


def test_persist_objective_ast_dataset_uses_ipfs_datasets_bridge(tmp_path, monkeypatch):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    saved: dict[str, object] = {}

    class FakeDataset:
        def __init__(self, rows):
            self.rows = list(rows)

        @classmethod
        def from_list(cls, rows):
            return cls(rows)

        def to_parquet(self, path):
            Path(path).write_text(json.dumps({"rows": len(self.rows)}), encoding="utf-8")

    class FakeManaged:
        def save(self, destination, format=None, **_options):
            return {"location": destination, "format": format, "size": 123}

    class FakeDatasetManager:
        def __init__(self, use_accelerate=True):
            saved["use_accelerate"] = use_accelerate

        def save_dataset(self, dataset_id, dataset):
            saved["dataset_id"] = dataset_id
            saved["row_count"] = len(dataset.rows)

        def get_dataset(self, _dataset_id):
            return FakeManaged()

    package = types.ModuleType("ipfs_datasets_py")
    ipfs_datasets = types.ModuleType("ipfs_datasets_py.ipfs_datasets")
    dataset_manager = types.ModuleType("ipfs_datasets_py.dataset_manager")
    ipfs_datasets.Dataset = FakeDataset
    dataset_manager.DatasetManager = FakeDatasetManager
    monkeypatch.setitem(sys.modules, "ipfs_datasets_py", package)
    monkeypatch.setitem(sys.modules, "ipfs_datasets_py.ipfs_datasets", ipfs_datasets)
    monkeypatch.setitem(sys.modules, "ipfs_datasets_py.dataset_manager", dataset_manager)

    artifact = persist_objective_ast_dataset(
        repo_root=repo,
        objective_path=objective_path,
        dataset_dir=repo / "datasets",
        dataset_id="objective-ast-test",
    )

    assert artifact.backend == "ipfs_datasets_py"
    assert artifact.parquet_path is not None and artifact.parquet_path.exists()
    assert artifact.manager_result == {"location": str(artifact.parquet_path), "format": "parquet", "size": 123}
    assert saved["dataset_id"] == "objective-ast-test"
    assert saved["row_count"] >= 2


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


def test_merge_resolver_cli_prints_payload(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "type": "merge_finished",
                "task_id": "ACCEL-010",
                "attempted": True,
                "merged": False,
                "branch": "implementation/accel-010",
                "target_branch": "main",
                "reason": "content_conflict",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    assert merge_resolver_main(["--events-path", str(events_path), "--repo-root", str(repo)]) == 0
    output = json.loads(capsys.readouterr().out)

    assert output["found"] is True
    assert output["task_id"] == "ACCEL-010"
    assert "Resolve the autonomous-agent supervisor merge conflict" in output["prompt"]
