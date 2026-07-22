from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor import backlog_refinery
from ipfs_accelerate_py.agent_supervisor.analyzer_health import (
    AnalyzerCanaryReport,
    AnalyzerCanaryResult,
    AnalyzerHealthThresholds,
)
from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
    ConfiguredBacklogRecorderBundle,
    ConfiguredCodebaseScanRecorder,
    ConfiguredObjectiveBacklogRecorder,
    ConfiguredRetryBudgetRecorder,
    build_configured_backlog_recorder_bundle,
    build_namespace_codebase_scan_recorder,
    build_namespace_objective_backlog_recorder,
    build_namespace_retry_budget_recorder,
    build_task_blocks_ensurer,
    commit_generated_dirty_outputs,
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
    write_reconciliation_guardrail_discovery_path,
)
from ipfs_accelerate_py.agent_supervisor.dataset_store import ObjectiveDatasetStore
from ipfs_accelerate_py.agent_supervisor.wrapper_utils import agent_supervisor_namespace_paths
from ipfs_accelerate_py.agent_supervisor.scan_receipts import (
    REFILL_SCAN_RESULT_SCHEMA_VERSION,
    SCAN_RECEIPT_PROJECTION_SCHEMA,
    SCAN_RECEIPT_PROJECTION_SCHEMA_VERSION,
    RefillScanResult,
    ScanTerminalReason,
    adapt_legacy_scan_callback,
    adapt_legacy_scan_result,
    build_scan_result,
    persist_scan_receipt,
    scan_identity,
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


def _git_dir(cwd: Path) -> Path:
    git_dir = Path(_git(cwd, "rev-parse", "--git-dir"))
    if not git_dir.is_absolute():
        git_dir = cwd / git_dir
    return git_dir.resolve()


def test_refill_scan_result_has_versioned_terminal_taxonomy_and_explicit_legacy_adapter(tmp_path):
    repo = _seed_repo(tmp_path)
    (repo / "README.md").write_text("seed\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed identity")
    started_at = datetime.now(timezone.utc)
    identity = scan_identity(repo)

    examples = {
        ScanTerminalReason.GENERATED: build_scan_result(
            ScanTerminalReason.GENERATED, "force", "test/v1", repo, started_at, ({"id": "A"},)
        ),
        ScanTerminalReason.EXHAUSTED: build_scan_result(
            ScanTerminalReason.EXHAUSTED,
            "drained_exhaustive",
            "test/v1",
            repo,
            started_at,
            safe_for_completion_reasoning=True,
        ),
        ScanTerminalReason.DUPLICATE_ONLY: build_scan_result(
            ScanTerminalReason.DUPLICATE_ONLY, "force", "test/v1", repo, started_at
        ),
        ScanTerminalReason.THRESHOLD_SATISFIED: build_scan_result(
            ScanTerminalReason.THRESHOLD_SATISFIED, "open_task_threshold", "test/v1", repo, started_at
        ),
        ScanTerminalReason.COOLDOWN: build_scan_result(
            ScanTerminalReason.COOLDOWN, "cooldown", "test/v1", repo, started_at
        ),
        ScanTerminalReason.DISABLED: build_scan_result(
            ScanTerminalReason.DISABLED, "disabled", "test/v1", repo, started_at
        ),
        ScanTerminalReason.PARTIAL: build_scan_result(
            ScanTerminalReason.PARTIAL, "incremental", "test/v1", repo, started_at, ({"id": "partial"},)
        ),
        ScanTerminalReason.FAILED: build_scan_result(
            ScanTerminalReason.FAILED, "force", "test/v1", repo, started_at, error="analyzer failed"
        ),
        ScanTerminalReason.TIMED_OUT: build_scan_result(
            ScanTerminalReason.TIMED_OUT, "force", "test/v1", repo, started_at, error="timed out"
        ),
    }

    assert set(examples) == set(ScanTerminalReason)
    exhausted = examples[ScanTerminalReason.EXHAUSTED]
    assert exhausted.schema_version == REFILL_SCAN_RESULT_SCHEMA_VERSION
    assert exhausted.repository_id == identity.repository_id
    assert exhausted.tree_id == identity.tree_id
    assert exhausted.finished_at >= exhausted.started_at
    assert exhausted.safe_for_completion_reasoning is True
    assert RefillScanResult.from_dict(exhausted.to_dict()) == exhausted
    try:
        bool(exhausted)
    except TypeError:
        pass
    else:
        raise AssertionError("typed refill results must reject implicit truthiness")

    legacy_empty = adapt_legacy_scan_result(
        [],
        empty_reason=ScanTerminalReason.COOLDOWN,
        scan_mode="legacy",
        analyzer_version="legacy/v1",
        repository_id=identity.repository_id,
        tree_id=identity.tree_id,
        started_at=started_at,
    )
    assert legacy_empty.terminal_reason is ScanTerminalReason.COOLDOWN
    assert legacy_empty.generated_count == 0
    assert legacy_empty.safe_for_completion_reasoning is False

    legacy_callback = adapt_legacy_scan_callback(
        lambda: [],
        empty_reason=ScanTerminalReason.PARTIAL,
        scan_mode="legacy",
        analyzer_version="legacy/v1",
        repo_root=repo,
    )
    callback_result = legacy_callback()
    assert callback_result.terminal_reason is ScanTerminalReason.PARTIAL
    assert callback_result.generated_count == 0
    assert callback_result.repository_identity == identity.repository_id
    assert callback_result.tree_identity == identity.tree_id

    (repo / "README.md").write_text("dirty tree\n", encoding="utf-8")
    dirty_identity = scan_identity(repo)
    assert dirty_identity.repository_id == identity.repository_id
    assert dirty_identity.tree_id.startswith("sha256:")
    assert dirty_identity.tree_id != identity.tree_id


def test_scan_receipt_projection_is_versioned_compact_and_content_addressed(tmp_path):
    started_at = datetime.now(timezone.utc)
    result = build_scan_result(
        ScanTerminalReason.GENERATED,
        "exhaustive",
        "test/v1",
        tmp_path,
        started_at,
        ({"task_id": "REF-TEST", "per_file_detail": "artifact-only"},),
        metadata={
            "scan_inventory": {
                "git_roots": 2,
                "tracked_files": 20,
                "eligible_files": 12,
                "parsed_files": 10,
                "excluded_files": 8,
                "parser_failures": 2,
                "raw_candidates": 6,
                "seen_candidates": 3,
                "deduplicated_candidates": 2,
                "appended_tasks": 1,
            },
            "per_file_details": [{"path": "src/example.py", "reason": "large"}],
        },
    )

    projection = persist_scan_receipt(
        result,
        tmp_path / "receipts",
        scan_kind="codebase",
        relative_to=tmp_path,
    )
    repeated = persist_scan_receipt(
        result,
        tmp_path / "receipts",
        scan_kind="codebase",
        relative_to=tmp_path,
    )

    assert repeated["receipt_cid"] == projection["receipt_cid"]
    assert repeated["artifact_path"] == projection["artifact_path"]
    assert projection["schema"] == SCAN_RECEIPT_PROJECTION_SCHEMA
    assert projection["schema_version"] == SCAN_RECEIPT_PROJECTION_SCHEMA_VERSION
    assert projection["receipt_cid"] == result.receipt_cid
    assert projection["freshness"]["status"] == "fresh"
    assert projection["freshness"]["fresh"] is True
    stale_projection = result.compact_projection(
        now=result.finished_at + timedelta(seconds=2),
        fresh_for_seconds=1,
    )
    assert stale_projection["freshness"]["status"] == "stale"
    assert stale_projection["freshness"]["fresh"] is False
    assert projection["candidate_funnel"] == {
        "git_root_count": 2,
        "tracked_file_count": 20,
        "eligible_file_count": 12,
        "parsed_file_count": 10,
        "excluded_file_count": 8,
        "parser_failure_count": 2,
        "raw_candidate_count": 6,
        "seen_candidate_count": 3,
        "deduplicated_candidate_count": 2,
        "appended_task_count": 1,
    }
    assert "items" not in projection
    assert "metadata" not in projection
    artifact = tmp_path / projection["artifact_path"]
    assert artifact.name == f"{result.receipt_cid}.json"
    assert "artifact-only" in artifact.read_text(encoding="utf-8")
    assert len(list((tmp_path / "receipts").glob("*.json"))) == 1


def test_commit_generated_dirty_outputs_commits_nested_repo_and_parent_gitlink(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "checkout", "-b", "main")
    _git(source, "config", "user.name", "Test User")
    _git(source, "config", "user.email", "test@example.invalid")
    (source / "docs").mkdir()
    (source / "docs" / "todo.md").write_text("# Todos\n", encoding="utf-8")
    _git(source, "add", "docs/todo.md")
    _git(source, "commit", "-m", "seed submodule")

    repo = _seed_repo(tmp_path)
    (repo / "README.md").write_text("root\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed root")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(source), "hallucinate_app")
    _git(repo, "commit", "-am", "add submodule")

    nested = repo / "hallucinate_app"
    (nested / "docs" / "todo.md").write_text("# Todos\n\n## Generated\n", encoding="utf-8")
    discovery = repo / "data" / "discovery" / "generated.md"
    discovery.parent.mkdir(parents=True)
    discovery.write_text("# Generated\n", encoding="utf-8")
    (repo / "unknown.txt").write_text("preserve me\n", encoding="utf-8")

    result = commit_generated_dirty_outputs(
        repo_root=repo,
        generated_paths=("hallucinate_app/docs/todo.md",),
        generated_prefixes=("data/discovery",),
        candidate_git_roots=(nested,),
        subject="Agent: commit generated outputs",
    )

    assert result["committed_count"] == 2
    assert result["selected_path_count"] == 3
    assert _git(nested, "log", "-1", "--pretty=%s") == "Agent: commit generated outputs"
    assert _git(repo, "log", "-1", "--pretty=%s") == "Agent: commit generated outputs"
    root_status = _git(repo, "status", "--short")
    assert "unknown.txt" in root_status
    assert "hallucinate_app" not in root_status
    assert "data/discovery/generated.md" not in root_status


def test_commit_generated_dirty_outputs_repairs_recursive_clean_gitlinks(tmp_path):
    leaf_source = tmp_path / "leaf-source"
    leaf_source.mkdir()
    _git(leaf_source, "init")
    _git(leaf_source, "checkout", "-b", "main")
    _git(leaf_source, "config", "user.name", "Test User")
    _git(leaf_source, "config", "user.email", "test@example.invalid")
    (leaf_source / "leaf.txt").write_text("base\n", encoding="utf-8")
    _git(leaf_source, "add", "leaf.txt")
    _git(leaf_source, "commit", "-m", "seed leaf")

    parent_source = tmp_path / "parent-source"
    parent_source.mkdir()
    _git(parent_source, "init")
    _git(parent_source, "checkout", "-b", "main")
    _git(parent_source, "config", "user.name", "Test User")
    _git(parent_source, "config", "user.email", "test@example.invalid")
    (parent_source / "README.md").write_text("parent\n", encoding="utf-8")
    _git(parent_source, "add", "README.md")
    _git(parent_source, "commit", "-m", "seed parent")
    _git(
        parent_source,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(leaf_source),
        "vendor/leaf",
    )
    _git(parent_source, "commit", "-am", "add leaf submodule")

    repo = _seed_repo(tmp_path)
    (repo / "README.md").write_text("root\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed root")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(parent_source), "modules/parent")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "update", "--init", "--recursive")
    _git(repo, "commit", "-am", "add parent submodule")

    nested_leaf = repo / "modules" / "parent" / "vendor" / "leaf"
    _git(nested_leaf, "config", "user.name", "Test User")
    _git(nested_leaf, "config", "user.email", "test@example.invalid")
    (nested_leaf / "leaf.txt").write_text("updated\n", encoding="utf-8")
    _git(nested_leaf, "commit", "-am", "update leaf")

    result = commit_generated_dirty_outputs(
        repo_root=repo,
        subject="Agent: repair nested gitlinks",
    )

    assert result["committed_count"] == 2
    assert result["selected_path_count"] == 2
    parent = repo / "modules" / "parent"
    assert _git(parent, "log", "-1", "--pretty=%s") == "Agent: repair nested gitlinks"
    assert _git(repo, "log", "-1", "--pretty=%s") == "Agent: repair nested gitlinks"
    assert _git(parent, "status", "--porcelain") == ""
    assert _git(repo, "status", "--porcelain") == ""


def test_commit_generated_dirty_outputs_repairs_stale_nested_index_lock(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    _git(source, "init")
    _git(source, "checkout", "-b", "main")
    _git(source, "config", "user.name", "Test User")
    _git(source, "config", "user.email", "test@example.invalid")
    (source / "docs").mkdir()
    (source / "docs" / "todo.md").write_text("# Todos\n", encoding="utf-8")
    _git(source, "add", "docs/todo.md")
    _git(source, "commit", "-m", "seed submodule")

    repo = _seed_repo(tmp_path)
    (repo / "README.md").write_text("root\n", encoding="utf-8")
    _git(repo, "add", "README.md")
    _git(repo, "commit", "-m", "seed root")
    _git(repo, "-c", "protocol.file.allow=always", "submodule", "add", str(source), "hallucinate_app")
    _git(repo, "commit", "-am", "add submodule")

    nested = repo / "hallucinate_app"
    lock_path = _git_dir(nested) / "index.lock"
    lock_path.write_text("stale lock\n", encoding="utf-8")
    os.utime(lock_path, (0, 0))
    (nested / "docs" / "todo.md").write_text("# Todos\n\n## Generated\n", encoding="utf-8")

    result = commit_generated_dirty_outputs(
        repo_root=repo,
        generated_paths=("hallucinate_app/docs/todo.md",),
        candidate_git_roots=(nested,),
        subject="Agent: commit generated outputs",
        stale_git_lock_seconds=1.0,
    )

    assert not lock_path.exists()
    assert any(item.get("removed") for item in result["lock_repairs"])
    assert result["committed_count"] == 2
    assert result["selected_path_count"] == 2
    assert "hallucinate_app" not in _git(repo, "status", "--short")


def test_commit_generated_dirty_outputs_defers_during_merge(tmp_path):
    repo = _seed_repo(tmp_path)
    generated = repo / "data" / "discovery" / "generated.md"
    generated.parent.mkdir(parents=True)
    generated.write_text("# Generated\n", encoding="utf-8")
    _git(repo, "add", "data/discovery/generated.md")
    _git(repo, "commit", "-m", "seed generated")

    generated.write_text("# Generated\n\nupdated\n", encoding="utf-8")
    (_git_dir(repo) / "MERGE_HEAD").write_text(f"{_git(repo, 'rev-parse', 'HEAD')}\n", encoding="utf-8")

    result = commit_generated_dirty_outputs(
        repo_root=repo,
        generated_prefixes=("data/discovery",),
        subject="Agent: commit generated outputs",
    )

    assert result["committed_count"] == 0
    assert result["selected_path_count"] == 0
    assert any(item.get("reason") == "repo_merge_in_progress" for item in result["skipped"])
    assert "data/discovery/generated.md" in _git(repo, "status", "--short")
    assert _git(repo, "log", "-1", "--pretty=%s") == "seed generated"


def test_namespace_recorder_factories_bind_standard_paths(tmp_path):
    namespace_paths = agent_supervisor_namespace_paths(tmp_path, "agent_supervisor")
    prepared: list[str] = []

    objective_recorder = build_namespace_objective_backlog_recorder(
        repo_root=tmp_path,
        namespace_paths=namespace_paths,
        objective_path=tmp_path / "objective.md",
        todo_path=tmp_path / "todo.md",
        strategy_path=tmp_path / "state" / "strategy.json",
        state_path=tmp_path / "state" / "state.json",
        task_header_prefix_value="## EX-",
        depends_on_if_present=("EX-001",),
        min_open_tasks=2,
        max_findings=3,
        cooldown_seconds=60,
        surplus_findings_per_goal=4,
        surplus_min_terms_per_todo=2,
        discovery_output_path="data/agent_supervisor/discovery",
        commit_outputs=True,
        commit_subject="EX: record objective findings",
        prepare_environment=lambda: prepared.append("objective"),
    )
    codebase_recorder = build_namespace_codebase_scan_recorder(
        repo_root=tmp_path,
        namespace_paths=namespace_paths,
        todo_path=tmp_path / "todo.md",
        strategy_path=tmp_path / "state" / "strategy.json",
        state_path=tmp_path / "state" / "state.json",
        task_header_prefix_value="## EX-",
        depends_on_if_present=("EX-002",),
        min_open_tasks=1,
        max_findings=5,
        cooldown_seconds=120,
        discovery_output_path="data/agent_supervisor/discovery",
        skip_prefixes=("data/agent_supervisor/state/",),
        commit_outputs=True,
        commit_subject="EX: record codebase findings",
        prepare_environment=lambda: prepared.append("codebase"),
    )
    retry_recorder = build_namespace_retry_budget_recorder(
        namespace_paths=namespace_paths,
        todo_path=tmp_path / "todo.md",
        events_path=tmp_path / "state" / "events.jsonl",
        strategy_path=tmp_path / "state" / "strategy.json",
        task_header_prefix_value="## EX-",
        validation_retry_budget=2,
        merge_retry_budget=1,
        implementation_retry_budget=0,
        validation_depends_on_if_present=("EX-003",),
        discovery_output_path="data/agent_supervisor/discovery",
        strip_validation_failure_kind=True,
        commit_outputs=True,
        repo_root=tmp_path,
        commit_subject="EX: record retry findings",
        prepare_environment=lambda: prepared.append("retry"),
    )

    assert isinstance(objective_recorder, ConfiguredObjectiveBacklogRecorder)
    assert objective_recorder.discovery_dir == namespace_paths.discovery_dir
    assert objective_recorder.default_bundle_dir == namespace_paths.objective_bundle_dir
    assert objective_recorder.default_dataset_dir == namespace_paths.objective_dataset_dir
    assert objective_recorder.todo_vector_index_path == namespace_paths.objective_todo_vector_index_path
    assert objective_recorder.depends_on_if_present == ("EX-001",)
    assert objective_recorder.min_open_tasks == 2
    assert objective_recorder.commit_subject == "EX: record objective findings"

    assert isinstance(codebase_recorder, ConfiguredCodebaseScanRecorder)
    assert codebase_recorder.discovery_dir == namespace_paths.discovery_dir
    assert codebase_recorder.default_bundle_dir == namespace_paths.objective_bundle_dir
    assert codebase_recorder.skip_prefixes == ("data/agent_supervisor/state/",)
    assert codebase_recorder.depends_on_if_present == ("EX-002",)
    assert codebase_recorder.max_findings == 5

    assert isinstance(retry_recorder, ConfiguredRetryBudgetRecorder)
    assert retry_recorder.discovery_dir == namespace_paths.discovery_dir
    assert retry_recorder.validation_depends_on_if_present == ("EX-003",)
    assert retry_recorder.validation_retry_budget == 2
    assert retry_recorder.repo_root == tmp_path
    assert retry_recorder.strip_validation_failure_kind is True

    objective_recorder.prepare_environment()
    codebase_recorder.prepare_environment()
    retry_recorder.prepare_environment()
    assert prepared == ["objective", "codebase", "retry"]


def test_configured_backlog_recorder_bundle_delegates_to_runtime_factories(monkeypatch, tmp_path):
    captured: dict[str, dict[str, object]] = {}

    def objective_recorder(**_: object) -> list[dict[str, object]]:
        return [{"kind": "objective"}]

    def codebase_recorder(**_: object) -> list[dict[str, object]]:
        return [{"kind": "codebase"}]

    def retry_recorder(**_: object) -> list[dict[str, object]]:
        return [{"kind": "retry"}]

    def daemon_factory(**kwargs: object) -> str:
        captured["daemon"] = kwargs
        return "daemon-hooks"

    def supervisor_factory(**kwargs: object) -> str:
        captured["supervisor"] = kwargs
        return "supervisor-hooks"

    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.implementation_daemon_runner."
        "build_daemon_refill_hooks_factory_from_recorders",
        daemon_factory,
    )
    monkeypatch.setattr(
        "ipfs_accelerate_py.agent_supervisor.implementation_supervisor_runner."
        "build_supervisor_refill_hooks_factory_from_recorders",
        supervisor_factory,
    )

    bundle = build_configured_backlog_recorder_bundle(
        objective_recorder=objective_recorder,
        codebase_scan_recorder=codebase_recorder,
        retry_budget_recorder=retry_recorder,
    )

    assert isinstance(bundle, ConfiguredBacklogRecorderBundle)
    assert bundle.daemon_refill_hooks_factory(
        discovery_dir=tmp_path / "discovery",
        objective_path_key="objective_path",
        repo_root=tmp_path,
        retry_budget_extra_kwargs={"discovery_output_path": "data/discovery"},
        scope_label="Example",
        after_order=("retry-budget", "objective-goal"),
    ) == "daemon-hooks"
    assert bundle.supervisor_refill_hooks_factory(
        discovery_dir_key="discovery_dir",
        objective_path=tmp_path / "objective.md",
        codebase_scan_extra_kwargs={"force": True},
        scope_label="Example",
        after_once_order=("retry-budget", "objective-goal"),
    ) == "supervisor-hooks"

    assert captured["daemon"]["objective_recorder"] is objective_recorder
    assert captured["daemon"]["codebase_scan_recorder"] is codebase_recorder
    assert captured["daemon"]["retry_budget_recorder"] is retry_recorder
    assert captured["daemon"]["discovery_dir"] == tmp_path / "discovery"
    assert captured["daemon"]["objective_path_key"] == "objective_path"
    assert captured["daemon"]["repo_root"] == tmp_path
    assert captured["daemon"]["retry_budget_extra_kwargs"] == {"discovery_output_path": "data/discovery"}
    assert captured["daemon"]["after_order"] == ("retry-budget", "objective-goal")

    assert captured["supervisor"]["objective_recorder"] is objective_recorder
    assert captured["supervisor"]["codebase_scan_recorder"] is codebase_recorder
    assert captured["supervisor"]["retry_budget_recorder"] is retry_recorder
    assert captured["supervisor"]["discovery_dir_key"] == "discovery_dir"
    assert captured["supervisor"]["objective_path"] == tmp_path / "objective.md"
    assert captured["supervisor"]["codebase_scan_extra_kwargs"] == {"force": True}
    assert captured["supervisor"]["after_once_order"] == ("retry-budget", "objective-goal")


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

    callback_path = tmp_path / "callback-tasks.todo.md"
    callback_path.write_text("# Agent Todos\n", encoding="utf-8")
    ensure_callback_blocks = build_task_blocks_ensurer(
        (("AUTO-004", "## AUTO-004 Callback task\n\n- Status: todo"),),
        default_todo_path=callback_path,
    )
    assert ensure_callback_blocks()
    assert "## AUTO-004 Callback task" in callback_path.read_text(encoding="utf-8")
    assert not ensure_callback_blocks()


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


def test_codebase_scan_receipt_accounts_inventory_candidates_and_durable_details(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "discovery"
    dataset_dir = repo / "datasets"
    (repo / "first.py").write_text("# TODO: repair first path\n", encoding="utf-8")
    (repo / "second.py").write_text("# TODO: repair second path\n", encoding="utf-8")
    (repo / "asset.bin").write_bytes(b"not an eligible source file\n")
    _write_todo(todo_path)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed instrumented scan")

    limited = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        dataset_dir=dataset_dir,
        repo_root=repo,
        max_findings=1,
        force=True,
    )

    assert limited.terminal_reason is ScanTerminalReason.PARTIAL
    assert limited.coverage.to_dict() == {
        "git_roots": 1,
        "tracked_files": 4,
        "eligible_files": 2,
        "parsed_files": 2,
        "cache_hits": 0,
        "excluded_files": 2,
        "parser_failures": 0,
    }
    assert limited.candidate_accounting.to_dict() == {
        "raw_candidates": 2,
        "seen_candidates": 0,
        "deduplicated_candidates": 0,
        "rejected_candidates": 1,
        "appended_tasks": 1,
    }
    assert limited.candidate_accounting.is_balanced
    assert limited.details_artifact is not None
    summaries = {
        summary.reason_code.value: summary
        for summary in limited.reason_summaries["exclusions"]
    }
    assert set(summaries) == {"todo_board", "unsupported_suffix"}
    assert summaries["todo_board"].representative_paths == ("todo.md",)
    assert summaries["unsupported_suffix"].representative_paths == ("asset.bin",)
    details = ObjectiveDatasetStore(dataset_dir).load_scan_details(
        limited.details_artifact.to_dict()
    )
    assert {(row["path"], row["reason_code"]) for row in details} == {
        ("asset.bin", "unsupported_suffix"),
        ("todo.md", "todo_board"),
    }
    assert RefillScanResult.from_dict(limited.to_dict()) == limited

    # The next pass observes the first task as durable history and accounts
    # for it separately from the second candidate that becomes a new task.
    complete = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        dataset_dir=dataset_dir,
        repo_root=repo,
        max_findings=1,
        force=True,
    )
    assert complete.terminal_reason is ScanTerminalReason.GENERATED
    assert complete.raw_candidates == 2
    assert complete.seen_candidates == 1
    assert complete.deduplicated_candidates == 0
    assert complete.rejected_candidates == 0
    assert complete.appended_tasks == 1
    assert complete.candidate_accounting.is_balanced


def test_codebase_scan_parser_failure_is_partial_reason_coded_and_durable(
    tmp_path,
    monkeypatch,
):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "discovery"
    dataset_dir = repo / "datasets"
    (repo / "healthy.py").write_text("VALUE = 1\n", encoding="utf-8")
    (repo / "unreadable.py").write_text("VALUE = 2\n", encoding="utf-8")
    _write_todo(todo_path)
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed parser failure scan")

    real_scan = backlog_refinery._scan_findings_in_file

    def fail_one_file(path, *, repo_root):
        if path.name == "unreadable.py":
            return [], {
                "path": "unreadable.py",
                "reason_code": "read_failed",
                "error": "PermissionError: fixture denied access",
            }
        return real_scan(path, repo_root=repo_root)

    monkeypatch.setattr(backlog_refinery, "_scan_findings_in_file", fail_one_file)
    receipt = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=strategy_path,
        discovery_dir=discovery_dir,
        dataset_dir=dataset_dir,
        repo_root=repo,
        max_findings=5,
        force=True,
    )

    assert receipt.terminal_reason is ScanTerminalReason.PARTIAL
    assert receipt.safe_for_completion_reasoning is False
    assert receipt.eligible_files == 2
    assert receipt.parsed_files == 1
    assert receipt.parser_failures == 1
    assert receipt.coverage.unparsed_eligible_files == 0
    failures = receipt.reason_summaries["parser_failures"]
    assert len(failures) == 1
    assert failures[0].reason_code.value == "read_failed"
    assert failures[0].representative_paths == ("unreadable.py",)
    assert receipt.details_artifact is not None
    details = ObjectiveDatasetStore(dataset_dir).load_scan_details(
        receipt.details_artifact.to_dict()
    )
    assert any(
        row.get("path") == "unreadable.py"
        and row.get("reason_code") == "read_failed"
        and "PermissionError" in row.get("error", "")
        for row in details
    )


def test_codebase_scan_writes_file_local_ast_bundle(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
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
        bundle_dir=bundle_dir,
        task_prefix="AUTO-",
        max_findings=1,
        force=True,
    )

    assert findings[0]["bundle_key"] == "codebase/runtime/src-runtime"
    shard_path = bundle_dir / "codebase-runtime-src-runtime.todo.md"
    assert "## AUTO-002 Resolve code annotation" in shard_path.read_text(encoding="utf-8")
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "- Bundle: codebase/runtime/src-runtime" in todo_text
    assert "- AST symbols:" in todo_text
    assert "- AST symbol scope: file" in todo_text
    assert "route_request" in todo_text
    index = json.loads((bundle_dir / "index.json").read_text(encoding="utf-8"))
    member = index["bundles"]["codebase/runtime/src-runtime"]["tasks"][0]
    assert member["candidate_kind"] == "codebase_scan"
    assert member["goal_registration"] == "dynamic"
    assert member["paths"] == ["src/runtime.py"]
    assert "route_request" in member["ast_symbols"]
    assert member["ast_symbol_scope"] == "file"


def test_codebase_scan_synchronizes_fingerprints_across_strategy_files(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text("# TODO: prove shared scan identity\n", encoding="utf-8")
    _write_todo(todo_path)
    _git(repo, "add", "todo.md", "src/runtime.py")
    _git(repo, "commit", "-m", "seed repo")

    first = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=repo / "state" / "first.json",
        discovery_dir=discovery_dir,
        repo_root=repo,
        task_prefix="AUTO-",
        max_findings=1,
        force=True,
    )
    second_strategy = repo / "state" / "second.json"
    second = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=second_strategy,
        discovery_dir=discovery_dir,
        repo_root=repo,
        task_prefix="AUTO-",
        max_findings=1,
        force=True,
    )

    assert len(first) == 1
    assert second.generated_count == 0
    assert second.terminal_reason is ScanTerminalReason.DUPLICATE_ONLY
    assert second.safe_for_completion_reasoning is False
    todo_text = todo_path.read_text(encoding="utf-8")
    assert todo_text.count("## AUTO-002") == 1
    assert "## AUTO-003" not in todo_text
    synchronized = json.loads(second_strategy.read_text(encoding="utf-8"))
    assert any(
        first[0]["fingerprint"].startswith(item)
        for item in synchronized["codebase_scan_seen_fingerprints"]
    )


def test_codebase_scan_retires_later_duplicate_vector_tasks(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Original finding

- Status: todo
- Completion: manual
- Candidate kind: codebase_scan
- Todo vector key: abcdef1234567890

## AUTO-002 Duplicate finding

- Status: todo
- Completion: manual
- Candidate kind: codebase_scan
- Todo vector key: abcdef1234567890
""",
        encoding="utf-8",
    )
    strategy_path = repo / "state" / "strategy.json"

    findings = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=strategy_path,
        discovery_dir=repo / "discovery",
        repo_root=repo,
        task_prefix="AUTO-",
        max_findings=0,
    )

    assert findings.generated_count == 0
    assert findings.terminal_reason is ScanTerminalReason.DISABLED
    assert findings.safe_for_completion_reasoning is False
    todo_text = todo_path.read_text(encoding="utf-8")
    duplicate_block = todo_text.split("## AUTO-002", 1)[1]
    assert "- Status: completed" in duplicate_block
    assert "- Completion: deduplicated:AUTO-001" in duplicate_block
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["codebase_scan_duplicate_tasks_retired"] == {"AUTO-002": "AUTO-001"}


def test_codebase_scan_reserves_ids_from_discovery_artifacts(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text("# TODO: prove durable task id allocation\n", encoding="utf-8")
    _write_todo(todo_path)
    discovery_dir.mkdir(parents=True)
    (discovery_dir / "2026-07-22-auto-009-objective-gap-deadbeef.md").write_text(
        "# Prior durable finding\n",
        encoding="utf-8",
    )
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

    assert findings[0]["follow_up_task_id"] == "AUTO-010"
    assert "## AUTO-010" in todo_path.read_text(encoding="utf-8")


def test_backlog_refinery_annotation_scan_ignores_literal_status_strings(tmp_path):
    repo = _seed_repo(tmp_path)
    source = repo / "src" / "runtime.py"
    source.parent.mkdir()
    source.write_text(
        """def status_values():
    literal = "todo"
    fixture = _task_status_line("todo")
    path = "docs/example.todo.md"
    markdown_status = "- todo ✅"
    # TODO: real source follow-up
    return literal, fixture, path
""",
        encoding="utf-8",
    )
    _git(repo, "add", "src/runtime.py")
    _git(repo, "commit", "-m", "seed scan target")

    findings = scan_codebase_findings(repo, max_findings=5)

    assert [(finding.root_relative_path, finding.line_number) for finding in findings] == [
        ("src/runtime.py", 6)
    ]


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


def test_backlog_refinery_releases_historical_completed_retry_repairs(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Original blocked task

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
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in AUTO-001. Use evidence in data/discovery/auto-002.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release AUTO-001 from strategy blocked_tasks.

## AUTO-003 Resolve implementation retry-budget failure for AUTO-999

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in AUTO-999. Use evidence in data/discovery/auto-003.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release AUTO-999 from strategy blocked_tasks.
""",
        encoding="utf-8",
    )
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["AUTO-001", "AUTO-999"],
                "retry_budget_findings": [],
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
            "failure_kind": "implementation",
            "reason": "historical_retry_repair_completed",
        },
        {
            "source_task_id": "AUTO-999",
            "follow_up_task_id": "",
            "guardrail_kind": "stale_strategy_block",
            "reason": "missing_task",
        },
    ]
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == []


def test_backlog_refinery_releases_orphaned_block_without_repair_path(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Resolve merge retry-budget failure for AUTO-099

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Merge retry-budget guardrail filed this from repeated merge failures in AUTO-099. Use evidence in data/discovery/auto-001.md to fix the merge blocker, verify the intended implementation changes are committed in their owning repository or submodule, run `ipfs-accelerate-agent-merge-resolver --events-path ... --apply` when the conflict is semantic, then remove AUTO-099 from the strategy blocked_tasks list so the original backlog item can continue without an indefinite retry loop.

## AUTO-002 Source with pending repair

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: This task must remain blocked until AUTO-003 completes.

## AUTO-003 Resolve implementation retry-budget failure for AUTO-002

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in AUTO-002. Use evidence in data/discovery/auto-003.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release AUTO-002 from strategy blocked_tasks.
""",
        encoding="utf-8",
    )
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps({"blocked_tasks": ["AUTO-001", "AUTO-002"], "retry_budget_findings": []}),
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
            "follow_up_task_id": "",
            "guardrail_kind": "stale_strategy_block",
            "reason": "no_guardrail_repair_path",
        }
    ]
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-002"]


def test_backlog_refinery_releases_recursive_retry_repair_block(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Original source task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Original task stays blocked until its first repair task completes.

## AUTO-002 Resolve implementation retry-budget failure for AUTO-001

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in AUTO-001. Use evidence in data/discovery/auto-002.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release AUTO-001 from strategy blocked_tasks.

## AUTO-003 Resolve implementation retry-budget failure for AUTO-002

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in AUTO-002. Use evidence in data/discovery/auto-003.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release AUTO-002 from strategy blocked_tasks.
""",
        encoding="utf-8",
    )
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["AUTO-001", "AUTO-002"],
                "retry_budget_findings": [
                    {
                        "source_task_id": "AUTO-002",
                        "follow_up_task_id": "AUTO-003",
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
            "source_task_id": "AUTO-002",
            "follow_up_task_id": "",
            "guardrail_kind": "stale_strategy_block",
            "failure_kind": "implementation",
            "reason": "recursive_retry_repair_block",
            "original_source_task_id": "AUTO-001",
        },
        {
            "source_task_id": "AUTO-003",
            "follow_up_task_id": "",
            "guardrail_kind": "retry_budget",
            "failure_kind": "implementation",
            "reason": "recursive_retry_repair_task_retired",
            "parent_repair_task_id": "AUTO-002",
            "original_source_task_id": "AUTO-001",
        }
    ]
    assert "- Status: completed" in todo_path.read_text(encoding="utf-8").split("## AUTO-003", 1)[1]
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]
    assert strategy["last_recursive_retry_repair_retired_task_ids"] == ["AUTO-003"]


def test_backlog_refinery_retires_ready_recursive_retry_repair_task(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Original source task

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Original task stays blocked until its first repair task completes.

## AUTO-002 Resolve implementation retry-budget failure for AUTO-001

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in AUTO-001. Use evidence in data/discovery/auto-002.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release AUTO-001 from strategy blocked_tasks.

## AUTO-003 Resolve implementation retry-budget failure for AUTO-002

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in AUTO-002. Use evidence in data/discovery/auto-003.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release AUTO-002 from strategy blocked_tasks.
""",
        encoding="utf-8",
    )
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps({"blocked_tasks": ["AUTO-001"], "retry_budget_findings": []}),
        encoding="utf-8",
    )

    releases = release_completed_guardrail_blocks(
        todo_path=todo_path,
        strategy_path=strategy_path,
        task_prefix="AUTO-",
    )

    assert releases == [
        {
            "source_task_id": "AUTO-003",
            "follow_up_task_id": "",
            "guardrail_kind": "retry_budget",
            "failure_kind": "implementation",
            "reason": "recursive_retry_repair_task_retired",
            "parent_repair_task_id": "AUTO-002",
            "original_source_task_id": "AUTO-001",
        }
    ]
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "- Status: todo" in todo_text.split("## AUTO-002", 1)[1].split("## AUTO-003", 1)[0]
    assert "- Status: completed" in todo_text.split("## AUTO-003", 1)[1]
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]
    assert strategy["last_recursive_retry_repair_retired_task_ids"] == ["AUTO-003"]


def test_backlog_refinery_releases_stale_dependency_guardrail_after_metadata_repaired(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Metadata repaired

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Dependency metadata no longer blocks readiness.

## AUTO-002 Resolve dependency guardrail for AUTO-001

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Stale repair task from a previous dependency guardrail.
""",
        encoding="utf-8",
    )
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["AUTO-001"],
                "dependency_guardrail_findings": [
                    {
                        "source_task_id": "AUTO-001",
                        "follow_up_task_id": "AUTO-002",
                        "fingerprint": "stale",
                        "missing_dependencies": ["AUTO-999"],
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
            "guardrail_kind": "dependency_guardrail",
            "reason": "dependency_metadata_resolved",
        }
    ]
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == []
    assert strategy["dependency_guardrail_findings"] == []


def test_backlog_refinery_keeps_block_when_dependency_guardrail_still_active(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    strategy_path = repo / "state" / "strategy.json"
    todo_path.write_text(
        """# Agent Todos

## AUTO-001 Still missing dependency

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: AUTO-999
- Outputs: src/runtime.py
- Validation: test -f todo.md
- Acceptance: Dependency metadata is still impossible.

## AUTO-002 Resolve dependency guardrail for AUTO-001

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f todo.md
- Acceptance: Existing repair task.
""",
        encoding="utf-8",
    )
    strategy_path.parent.mkdir(parents=True, exist_ok=True)
    strategy_path.write_text(
        json.dumps(
            {
                "blocked_tasks": ["AUTO-001"],
                "dependency_guardrail_findings": [
                    {
                        "source_task_id": "AUTO-001",
                        "follow_up_task_id": "AUTO-002",
                        "fingerprint": "stale",
                        "missing_dependencies": ["AUTO-404"],
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

    assert releases == []
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == ["AUTO-001"]
    assert strategy["dependency_guardrail_findings"][0]["source_task_id"] == "AUTO-001"


def test_backlog_refinery_preserves_reconciliation_resolution_sections(tmp_path):
    path = tmp_path / "discovery" / "reconciliation.md"
    path.parent.mkdir(parents=True)
    path.write_text(
        """# AUTO-010 Reconciliation Guardrail

## Main Checkout Status

- old

## Resolution Evidence

Manual cleanup reduced the stale worktree count from 81 to 20.

## Reconciliation Plan

old generated plan
""",
        encoding="utf-8",
    )
    record = {
        "fingerprint": "abc123",
        "kind": "main_checkout_dirty",
        "reason": "main_checkout_dirty",
        "candidate_count": 2,
        "priority": "P1",
        "track": "ops",
        "status_short": [" M generated.md"],
        "summary": "Resolve dirty main checkout blocking 2 worktree merges",
        "dedupe_key": "reconciliation_guardrail:main_checkout_dirty",
        "samples": [],
    }

    write_reconciliation_guardrail_discovery_path(
        path=path,
        task_id="AUTO-010",
        record=record,
        date="2026-06-07",
    )
    write_reconciliation_guardrail_discovery_path(
        path=path,
        task_id="AUTO-010",
        record={**record, "candidate_count": 3},
        date="2026-06-07",
    )

    text = path.read_text(encoding="utf-8")
    assert "Candidate count: 3" in text
    assert "Manual cleanup reduced the stale worktree count from 81 to 20." in text
    assert text.count("## Resolution Evidence") == 1


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


def test_backlog_refinery_retry_budget_skips_recursive_repair_tasks(tmp_path):
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
- Acceptance: Fix the original implementation setup failure.

## AUTO-002 Resolve implementation retry-budget failure for AUTO-001

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on:
- Outputs: discovery
- Validation: test -f data/discovery/auto-002.md
- Acceptance: Implementation retry-budget guardrail filed this from repeated implementation failures in AUTO-001. Use evidence in data/discovery/auto-002.md to fix the setup, runtime, or timeout blocker, then mark this repair task completed so the supervisor can release AUTO-001 from strategy blocked_tasks.
""",
        encoding="utf-8",
    )
    events_path.parent.mkdir(parents=True)
    failure = {
        "type": "implementation_finished",
        "task_id": "AUTO-002",
        "attempt": 1,
        "returncode": 1,
        "validation_result": {"attempted": False, "passed": True},
        "merge_result": {"attempted": False, "merged": False, "reason": "not_attempted"},
        "exception_result": {
            "exception_type": "RuntimeError",
            "phase": "worktree_setup",
            "message": "not a git repository",
        },
        "log_path": "state/implementation_logs/auto-002-attempt-1.log",
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
        validation_retry_budget=2,
        merge_retry_budget=2,
    )

    assert findings == []
    assert "AUTO-003" not in todo_path.read_text(encoding="utf-8")
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert strategy["blocked_tasks"] == []
    assert "retry_budget_findings" not in strategy


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


def test_codebase_scan_healthy_exhaustion_records_policy_and_is_completion_safe(tmp_path):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    _write_todo(todo_path)
    (repo / "healthy.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed healthy exhaustive scan")
    thresholds = AnalyzerHealthThresholds(max_excluded_file_ratio=0.75)

    receipt = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=repo / "state" / "strategy.json",
        discovery_dir=repo / "discovery",
        repo_root=repo,
        max_findings=5,
        health_thresholds=thresholds,
    )

    assert receipt.terminal_reason is ScanTerminalReason.EXHAUSTED
    assert receipt.safe_for_completion_reasoning is True
    assert receipt.metadata["health_thresholds"] == thresholds.to_dict()
    assert receipt.metadata["analyzer_health"]["status"] == "healthy"
    assert receipt.metadata["analyzer_canaries"]["passed"] is True
    assert receipt.metadata["expected_git_root_count"] == 1


def test_codebase_scan_missing_root_fails_closed_without_suppressing_retry(tmp_path, monkeypatch):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    _write_todo(todo_path)
    (repo / "healthy.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed root discovery failure")
    monkeypatch.setattr(backlog_refinery, "discover_git_worktrees", lambda *_args, **_kwargs: [])
    strategy_path = repo / "state" / "strategy.json"

    receipt = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=strategy_path,
        discovery_dir=repo / "discovery",
        repo_root=repo,
        max_findings=5,
    )

    assert receipt.terminal_reason is ScanTerminalReason.FAILED
    assert receipt.safe_for_completion_reasoning is False
    assert "no_git_roots_discovered" in receipt.metadata["analyzer_health"]["reasons"]
    strategy = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert "last_drained_codebase_scan_task_count" not in strategy


def test_codebase_scan_canary_failure_keeps_generated_work_but_returns_partial(
    tmp_path, monkeypatch
):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    _write_todo(todo_path)
    (repo / "finding.py").write_text("# TODO: retain safe work\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed canary failure")
    failed_canaries = AnalyzerCanaryReport(
        backlog_refinery.CODEBASE_SCAN_ANALYZER_VERSION,
        (
            AnalyzerCanaryResult(
                "broken", "line_source", ("annotated_followup",), ()
            ),
        ),
    )
    monkeypatch.setattr(
        backlog_refinery, "run_codebase_analyzer_canaries", lambda: failed_canaries
    )

    receipt = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=repo / "state" / "strategy.json",
        discovery_dir=repo / "discovery",
        repo_root=repo,
        max_findings=5,
        force=True,
    )

    assert receipt.terminal_reason is ScanTerminalReason.PARTIAL
    assert receipt.generated_count == 1
    assert receipt.safe_for_completion_reasoning is False
    assert "## AUTO-002" in todo_path.read_text(encoding="utf-8")
    assert receipt.metadata["analyzer_health"]["status"] == "unhealthy"


def test_codebase_scan_strict_parser_budget_fails_closed(tmp_path, monkeypatch):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    _write_todo(todo_path)
    (repo / "unreadable.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed strict parser budget")
    real_scan = backlog_refinery._scan_findings_in_file

    def fail_one(path, *, repo_root):
        if path.name == "unreadable.py":
            return [], {
                "path": "unreadable.py",
                "reason_code": "read_failed",
                "error": "PermissionError: fixture",
            }
        return real_scan(path, repo_root=repo_root)

    monkeypatch.setattr(backlog_refinery, "_scan_findings_in_file", fail_one)
    receipt = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=repo / "state" / "strategy.json",
        discovery_dir=repo / "discovery",
        repo_root=repo,
        max_findings=5,
        force=True,
        health_thresholds={
            "max_parser_failures": 0,
            "max_parser_failure_ratio": 0.0,
        },
    )

    assert receipt.terminal_reason is ScanTerminalReason.FAILED
    assert receipt.parser_failures == 1
    assert "parser_failure_budget_exceeded" in receipt.error
    assert receipt.metadata["health_thresholds"]["max_parser_failures"] == 0


def test_codebase_scan_impossible_funnel_returns_typed_failure(tmp_path, monkeypatch):
    repo = _seed_repo(tmp_path)
    todo_path = repo / "todo.md"
    _write_todo(todo_path)
    (repo / "healthy.py").write_text("VALUE = 1\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed impossible funnel")
    invalid = backlog_refinery.CodebaseScanInventory(
        git_roots=["."],
        expected_git_roots=["."],
        tracked_file_count=2,
        eligible_file_count=1,
        parsed_file_count=1,
        excluded_files=[{"path": "todo.md", "reason_code": "todo_board"}],
        raw_candidate_count=1,
    )
    monkeypatch.setattr(backlog_refinery, "scan_codebase_findings", lambda *_a, **_k: invalid)

    receipt = record_codebase_scan_findings(
        todo_path=todo_path,
        state_path=None,
        strategy_path=repo / "state" / "strategy.json",
        discovery_dir=repo / "discovery",
        repo_root=repo,
        max_findings=5,
        force=True,
    )

    assert receipt.terminal_reason is ScanTerminalReason.FAILED
    assert receipt.safe_for_completion_reasoning is False
    assert "impossible_funnel:raw_candidates_do_not_balance" in receipt.error
    assert receipt.accounting is None
    assert receipt.metadata["invalid_scan_accounting"]["raw_candidates"] == 1
