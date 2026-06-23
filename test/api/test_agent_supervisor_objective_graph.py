from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import (
    build_bundle_task_payloads,
    generate_objective_todos,
    objective_heap_schedule,
    parse_goal_heap,
    persist_objective_ast_dataset,
    resolver_payload,
    scan_objective_gaps,
    submit_bundle_tasks,
)
from ipfs_accelerate_py.agent_supervisor.merge_resolver import (
    MergeResolverCliConfig,
    build_llm_merge_resolver_invoker,
    build_merge_prompt_callback,
    build_namespace_merge_resolver_runner,
    build_resolver_payload_callback,
    main as merge_resolver_main,
    run_configured_merge_resolver_cli,
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


def test_objective_goal_heap_accepts_package_specific_goal_ids():
    goals = parse_goal_heap(
        """# Objective Heap

## APP.GOAL-001 Package-specific proof

- Status: active
- Evidence: package proof
- Goal: Prove a package-specific objective.
"""
    )

    assert len(goals) == 1
    assert goals[0].goal_id == "APP.GOAL-001"
    assert goals[0].title == "Package-specific proof"


def test_objective_heap_schedule_uses_fibonacci_then_work_surface():
    goals = parse_goal_heap(
        """# Objective Heap

## VAIOS-G001 Small earlier band

- Status: active
- Fib priority: 1
- Priority: P1
- Evidence: one
- Outputs: docs

## VAIOS-G002 Large same band

- Status: active
- Fib priority: 2
- Priority: P1
- Evidence: one, two, three
- Outputs: src, tests, docs
- Interoperability pair: hallucinate_app, swissknife

## VAIOS-G003 Small same band

- Status: active
- Fib priority: 2
- Priority: P1
- Evidence: one
- Outputs: docs
"""
    )

    schedule = objective_heap_schedule(goals)

    assert [record.goal_id for record in schedule] == ["VAIOS-G001", "VAIOS-G002", "VAIOS-G003"]
    assert schedule[1].work_surface_score > schedule[2].work_surface_score
    assert schedule[1].sort_key[2] < schedule[2].sort_key[2]


def test_objective_graph_scanner_semantic_ast_bundles_implicit_goals(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    source = repo / "src" / "capability_router.py"
    source.parent.mkdir()
    source.write_text(
        """class CapabilityRouter:
    def dispatch_task(self, request):
        return request

    def schedule_task(self, request):
        return request
""",
        encoding="utf-8",
    )
    objective_path = repo / "objective-heap.md"
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

## VAIOS-G002 Capability scheduling contract

- Status: active
- Parent:
- Fib priority: 2
- Track: runtime
- Priority: P1
- Goal: Prove capability routing scheduling contracts.
- Evidence: CapabilityRouter.schedule_task, missing capability schedule contract
- Outputs: src/capability_router.py, tests
- Validation: test -f objective-heap.md
- AST query: CapabilityRouter.schedule_task
- Embedding query: capability routing schedule contract
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective-heap.md", "src/capability_router.py")
    _git(repo, "commit", "-m", "seed implicit bundle objectives")

    findings = scan_objective_gaps(repo, objective_path=objective_path, max_findings=2)

    assert len(findings) == 2
    assert {finding.bundle_explicit for finding in findings} == {False}
    assert {finding.bundle_strategy for finding in findings} == {"semantic_ast"}
    assert len({finding.bundle_key for finding in findings}) == 1
    assert findings[0].bundle_key.startswith("objective/runtime/src/semantic-")
    assert findings[0].parallel_lane == findings[0].bundle_key


def test_objective_graph_appends_playwright_validation_for_launch_goals(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G697 Production launch readiness gate

- Status: active
- Parent:
- Fib priority: 1
- Track: launch
- Priority: P0
- Bundle: objective/launch/production-readiness-gate
- Goal: Prove phone, desktop, Swissknife, Hallucinate App, and Meta glasses launch readiness.
- Evidence: launch_readiness_receipt_v1
- Outputs: tests
- Validation: test -f objective-heap.md
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective-heap.md")
    _git(repo, "commit", "-m", "seed launch objective")

    findings = scan_objective_gaps(repo, objective_path=objective_path, max_findings=1)

    assert len(findings) == 1
    validation = findings[0].validation
    assert validation.startswith("test -f objective-heap.md && ")
    assert "npm --prefix swissknife run test:e2e:meta-glasses" in validation
    assert "npm --prefix hallucinate_app run test:e2e -- multimodal-control-surface.spec.ts" in validation


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
    assert "## ACCEL-002 Close objective gap" in todo_text
    assert "- Bundle: objective/ops/root" in todo_text

    shard = bundle_dir / "objective-ops-root.todo.md"
    assert shard.exists()
    assert "## ACCEL-002 Close objective gap" in shard.read_text(encoding="utf-8")
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


def test_merge_resolver_payload_accepts_project_prompt_customization(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "type": "merge_finished",
                "task_id": "CUSTOM-001",
                "attempted": True,
                "merged": False,
                "branch": "implementation/custom-001",
                "target_branch": "main",
                "command": ["git", "merge", "implementation/custom-001"],
                "reason": "content_conflict",
                "dirty_paths": ["custom-module"],
            }
        )
        + "\n",
        encoding="utf-8",
    )

    payload = resolver_payload(
        events_path=events_path,
        repo_root=repo,
        task_id="CUSTOM-001",
        prompt_heading="Resolve the project-specific daemon merge conflict.",
        completion_rule="Do not remove the project task from blocked_tasks until validation passes.",
        extra_rules=["Prefer project-local adapters over package-specific defaults."],
    )

    assert payload["found"] is True
    assert "Resolve the project-specific daemon merge conflict." in payload["prompt"]
    assert "Do not remove the project task from blocked_tasks" in payload["prompt"]
    assert "Prefer project-local adapters" in payload["prompt"]


def test_merge_resolver_configured_callbacks_and_cli(tmp_path, capsys):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    events_path = tmp_path / "events.jsonl"
    events_path.write_text(
        json.dumps(
            {
                "type": "merge_finished",
                "task_id": "CUSTOM-002",
                "attempted": True,
                "merged": False,
                "branch": "implementation/custom-002",
                "target_branch": "main",
                "reason": "content_conflict",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    prompt_callback = build_merge_prompt_callback(
        prompt_heading="Resolve the configured merge conflict.",
        completion_rule="Keep configured blocked_tasks intact until validation passes.",
    )
    payload_callback = build_resolver_payload_callback(
        prompt_heading="Resolve the configured merge conflict.",
        completion_rule="Keep configured blocked_tasks intact until validation passes.",
    )
    event = json.loads(events_path.read_text(encoding="utf-8"))
    prompt = prompt_callback(event=event, repo_root=repo)
    payload = payload_callback(events_path=events_path, repo_root=repo, task_id="CUSTOM-002")

    assert "Resolve the configured merge conflict." in prompt
    assert payload["found"] is True
    assert "Keep configured blocked_tasks intact" in payload["prompt"]

    assert run_configured_merge_resolver_cli(
        MergeResolverCliConfig(
            default_events_path=events_path,
            default_repo_root=repo,
            prompt_heading="Resolve the configured merge conflict.",
            completion_rule="Keep configured blocked_tasks intact until validation passes.",
        ),
        ["--task-id", "CUSTOM-002"],
    ) == 0
    output = json.loads(capsys.readouterr().out)
    assert output["task_id"] == "CUSTOM-002"
    assert "Resolve the configured merge conflict." in output["prompt"]


def test_namespace_merge_resolver_runner_uses_namespace_state_and_env(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()

    runner = build_namespace_merge_resolver_runner(
        repo_root=repo,
        namespace="agent_supervisor",
        state_prefix="agent",
        env_prefix="AGENT",
        prompt_heading="Resolve the namespace merge conflict.",
        completion_rule="Keep namespace blocked_tasks intact until validation passes.",
        missing_event_exit_code=7,
        apply_failed_exit_code=8,
    )
    parsed = runner.parse_args([])

    assert parsed.events_path == repo / "data" / "agent_supervisor" / "state" / "agent_events.jsonl"
    assert parsed.repo_root == repo
    assert runner.config.primary_command_env_var == "AGENT_LLM_MERGE_RESOLVER_COMMAND"
    assert runner.config.missing_event_exit_code == 7
    assert runner.config.apply_failed_exit_code == 8
    prompt = runner.build_merge_prompt()(
        event={
            "type": "merge_finished",
            "task_id": "AGENT-001",
            "attempted": True,
            "merged": False,
            "branch": "implementation/agent-001",
            "target_branch": "main",
            "reason": "content_conflict",
        },
        repo_root=repo,
    )
    assert "Resolve the namespace merge conflict." in prompt
    assert "Keep namespace blocked_tasks intact" in prompt


def test_merge_resolver_invoker_reports_configured_env_names(monkeypatch):
    monkeypatch.delenv("PROJECT_MERGE_COMMAND", raising=False)
    monkeypatch.delenv("FALLBACK_MERGE_COMMAND", raising=False)
    invoker = build_llm_merge_resolver_invoker(
        primary_command_env_var="PROJECT_MERGE_COMMAND",
        fallback_command_env_var="FALLBACK_MERGE_COMMAND",
    )

    result = invoker({"found": True, "prompt": "resolve"})

    assert result["applied"] is False
    assert result["apply_error"] == "PROJECT_MERGE_COMMAND or FALLBACK_MERGE_COMMAND is not set"


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
