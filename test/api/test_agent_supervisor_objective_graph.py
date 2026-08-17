from __future__ import annotations

import json
import subprocess
import sys
import types
from dataclasses import replace
from pathlib import Path

import pytest

import ipfs_accelerate_py.agent_supervisor.objectives.objective_graph as objective_graph_module
import ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker as objective_tracker_module
from ipfs_accelerate_py.agent_supervisor import (
    build_bundle_task_payloads,
    generate_objective_todos,
    goal_graph,
    objective_heap_schedule,
    parse_goal_heap,
    persist_objective_ast_dataset,
    resolver_payload,
    scan_objective_gaps,
    submit_bundle_tasks,
)
from ipfs_accelerate_py.agent_supervisor.merge.merge_resolver import (
    MergeResolverCliConfig,
    build_llm_merge_resolver_invoker,
    build_merge_prompt_callback,
    build_namespace_merge_resolver_runner,
    build_resolver_payload_callback,
    main as merge_resolver_main,
    run_configured_merge_resolver_cli,
)
from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import plan_bundle_lanes
from ipfs_accelerate_py.agent_supervisor.objectives.objective_daemon import (
    build_arg_parser as build_objective_daemon_arg_parser,
    completion_gate_work_terms,
    objective_generation_proposals,
    objective_generation_task_findings,
    run_objective_daemon,
)
from ipfs_accelerate_py.agent_supervisor.objectives.goal_completion import CompletionEvidence
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor_runner import (
    build_goal_completion_projection,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_graph import (
    EXTERNAL_AUTHORITY_BENCHMARK_GOAL_IDS,
    EvidenceSourcePolicy,
    ObjectiveFinding,
    add_goal_packet_aggregate_findings,
    assign_goal_subgoal_packets,
    collect_ast_dataset_records,
    evidence_index,
    materialize_task_dependency_dag,
    materialize_task_planning_graph,
    objective_fingerprint,
    objective_finding_conflict_record,
    objective_finding_evidence_output_paths,
    objective_finding_task_identity,
    taskboard_namespace_from_todo,
    tracked_files,
    write_bundle_shards,
)
from ipfs_accelerate_py.agent_supervisor.objectives.objective_tracker import (
    append_launch_readiness_goals,
    append_refinement_goals,
    completion_tree_identity,
    migrate_legacy_objective_goals,
    reconcile_objective_goal_completion,
    rewrite_goal_fields,
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


def test_goal_heap_parser_preserves_wrapped_fields_across_rewrite():
    objective_text = """# Objective Heap

## HSSL-G240 Bind runtime namespaces

- Status: active
- Parent: HSSL-G211,
  HSSL-G230
- Goal: Implement the fail-closed source executor and detached-replay boundary
  that proves the actual environment, cache, process, and state namespaces
  came from the exact preregistered policy.
- Outputs: benchmarks/logic_pipeline/source_orchestration.py,
  benchmarks/logic_pipeline/runtime_confinement.py,
  tests/integration/benchmarks/logic_pipeline/test_source_orchestration.py
- Validation: python -m pytest
  tests/integration/benchmarks/logic_pipeline/test_source_orchestration.py -q
- Acceptance: Source-safe tests prove disjoint namespace CIDs;
  production children enforce the pinned runtime and confinement policy;
  no protected input enters public evidence.
- Gap task: Finish the pinned runtime and cache-truth implementation;
  pass the complete source-safe regression suite.
"""

    goal = parse_goal_heap(objective_text)[0]

    assert goal.parent_goal_ids == ["HSSL-G211", "HSSL-G230"]
    assert goal.fields["goal"] == (
        "Implement the fail-closed source executor and detached-replay boundary "
        "that proves the actual environment, cache, process, and state namespaces "
        "came from the exact preregistered policy."
    )
    assert goal.fields["outputs"].endswith(
        "tests/integration/benchmarks/logic_pipeline/test_source_orchestration.py"
    )
    assert goal.fields["validation"] == (
        "python -m pytest "
        "tests/integration/benchmarks/logic_pipeline/test_source_orchestration.py -q"
    )
    assert "production children enforce" in goal.fields["acceptance"]
    assert goal.fields["gap_task"].endswith(
        "pass the complete source-safe regression suite."
    )

    rewritten = rewrite_goal_fields(
        objective_text,
        {"HSSL-G240": {"Status": "reopened"}},
    )
    reparsed = parse_goal_heap(rewritten)[0]
    assert reparsed.status == "reopened"
    assert reparsed.fields["goal"] == goal.fields["goal"]
    assert reparsed.fields["outputs"] == goal.fields["outputs"]
    assert reparsed.fields["acceptance"] == goal.fields["acceptance"]
    assert reparsed.fields["gap_task"] == goal.fields["gap_task"]

    replaced = rewrite_goal_fields(
        objective_text,
        {
            "HSSL-G240": {
                "Goal": "Use the reviewed replacement boundary.",
            }
        },
    )
    replaced_goal = parse_goal_heap(replaced)[0]
    assert replaced_goal.fields["goal"] == (
        "Use the reviewed replacement boundary."
    )
    assert "actual environment" not in replaced_goal.fields["goal"]


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


def test_evidence_index_streams_each_direct_source_before_reading_the_next(
    tmp_path,
    monkeypatch,
):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    first = repo / "src" / "runtime_router.py"
    second = repo / "docs" / "runtime_notes.md"
    first_was_scored = False

    original_evaluate = EvidenceSourcePolicy.evaluate

    def observed_evaluate(self, requirement, **kwargs):
        nonlocal first_was_scored
        if kwargs.get("source_path") == "src/runtime_router.py":
            first_was_scored = True
        return original_evaluate(self, requirement, **kwargs)

    def observed_candidates(*_args, **_kwargs):
        yield first
        assert first_was_scored
        yield second

    monkeypatch.setattr(EvidenceSourcePolicy, "evaluate", observed_evaluate)
    monkeypatch.setattr(
        objective_graph_module,
        "objective_candidate_files",
        observed_candidates,
    )

    evidence = evidence_index(
        repo,
        objective_path=objective_path,
        terms=["CapabilityRouter"],
        embedding_min_score=2.0,
    )

    assert first_was_scored is True
    assert evidence == {
        "CapabilityRouter": ["src/runtime_router.py (exact)"],
    }


def test_objective_scanner_excludes_sensitive_root_without_reading_it(
    tmp_path,
    monkeypatch,
):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    excluded_root = repo / "private_inputs"
    excluded_source = excluded_root / "answer.py"
    excluded_root.mkdir()
    excluded_source.write_text(
        "def HSSLEV_PRIVATE_INPUT():\n    return 'hidden evidence'\n",
        encoding="utf-8",
    )
    outside_source = tmp_path / "outside_answer.py"
    outside_source.write_text(
        "def HSSLEV_PRIVATE_INPUT():\n    return 'linked hidden evidence'\n",
        encoding="utf-8",
    )
    excluded_link = excluded_root / "linked_answer.py"
    excluded_link.symlink_to(outside_source)
    objective_path.write_text(
        """# Objective Heap

## TEST-G001 Require independently visible evidence

- Status: active
- Parent:
- Fib priority: 1
- Track: test
- Priority: P0
- Bundle: objective/test/exclusions
- Goal: Keep excluded evidence unavailable to the objective scanner.
- Evidence: HSSLEV_PRIVATE_INPUT
- Outputs: src
- Validation: true
- Gap task: Produce public evidence.
""",
        encoding="utf-8",
    )
    _git(
        repo,
        "add",
        "objective-heap.md",
        "private_inputs/answer.py",
        "private_inputs/linked_answer.py",
    )
    _git(repo, "commit", "-m", "seed private objective evidence")

    assert scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        embedding_min_score=2.0,
    ) == []

    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    excluded_resolved = {
        excluded_source.resolve(),
        excluded_link.resolve(),
    }

    def guarded_read_text(path, *args, **kwargs):
        assert path.resolve() not in excluded_resolved
        return original_read_text(path, *args, **kwargs)

    def guarded_read_bytes(path, *args, **kwargs):
        assert path.resolve() not in excluded_resolved
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    direct_stats = {}
    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        embedding_min_score=2.0,
        scan_exclude_paths=[excluded_root],
        scan_stats=direct_stats,
    )

    assert [finding.goal_id for finding in findings] == ["TEST-G001"]
    assert direct_stats["scan_exclude_paths"] == ["private_inputs"]
    assert direct_stats["scan_exclude_path_count"] == 1

    dataset_dir = tmp_path / "objective-dataset"
    dataset_stats = {}
    dataset_findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        embedding_min_score=2.0,
        dataset_dir=dataset_dir,
        dataset_id="excluded-root-test",
        scan_exclude_paths=["private_inputs"],
        scan_stats=dataset_stats,
    )

    assert [finding.goal_id for finding in dataset_findings] == ["TEST-G001"]
    assert dataset_stats["scan_exclude_paths"] == ["private_inputs"]
    dataset_rows = (
        dataset_dir / "excluded-root-test.jsonl"
    ).read_text(encoding="utf-8")
    assert "private_inputs/answer.py" not in dataset_rows

    stale_cached_evidence = evidence_index(
        repo,
        objective_path=objective_path,
        terms=["HSSLEV_PRIVATE_INPUT"],
        embedding_min_score=2.0,
        records=[
            {
                "root_relative_path": "private_inputs/answer.py",
                "evidence_text": "def HSSLEV_PRIVATE_INPUT(): pass",
                "symbols_json": json.dumps(["HSSLEV_PRIVATE_INPUT"]),
                "document_tokens_json": "[]",
                "document_embedding_json": "[]",
            }
        ],
        scan_exclude_paths=["private_inputs"],
    )
    assert stale_cached_evidence == {"HSSLEV_PRIVATE_INPUT": []}


def test_completed_objective_card_refills_only_a_genuine_path_residual(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective.md"
    todo_path = repo / "todo.md"
    discovery_dir = repo / "generated" / "discovery"
    bundle_dir = repo / "generated" / "bundles"
    analyzer = "data/policy/analyzer-profile-v1.json"
    resource_bounds = "data/policy/resource-bounds-v1.json"
    objective_path.write_text(
        f"""# Objective Heap

## TEST-G050 Pin execution policy

- Status: active
- Parent:
- Track: trust
- Priority: P0
- Bundle: objective/trust
- Goal: Pin both execution policy artifacts.
- Evidence: {analyzer}, {resource_bounds}
- Outputs: {analyzer}, {resource_bounds}
- Validation: true
- Gap task: Add each missing policy artifact.
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Objective Todo\n", encoding="utf-8")
    _git(repo, "add", "objective.md", "todo.md")
    _git(repo, "commit", "-m", "seed objective")

    original = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="TEST-",
        max_findings=1,
        scan_exclude_paths=["data"],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )
    assert [record.finding.missing_evidence for record in original] == [
        [analyzer, resource_bounds]
    ]
    todo_path.write_text(
        todo_path.read_text(encoding="utf-8").replace(
            "- Status: todo",
            "- Status: completed",
            1,
        ),
        encoding="utf-8",
    )

    analyzer_path = repo / analyzer
    analyzer_path.parent.mkdir(parents=True)
    analyzer_path.write_text('{"schema":"analyzer@1"}\n', encoding="utf-8")
    validation_path = repo / "tests" / "test_execution_profile.py"
    validation_path.parent.mkdir()
    validation_path.write_text(
        f'RESOURCE_BOUNDS = "{resource_bounds}"\n',
        encoding="utf-8",
    )
    _git(repo, "add", analyzer, "tests/test_execution_profile.py")
    _git(repo, "commit", "-m", "complete only analyzer profile")

    evidence = evidence_index(
        repo,
        objective_path=objective_path,
        terms=[analyzer, resource_bounds],
        scan_exclude_paths=["data"],
    )
    assert evidence == {
        analyzer: [f"{analyzer} (path)"],
        resource_bounds: [],
    }

    residual = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="TEST-",
        max_findings=1,
        scan_exclude_paths=["data"],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )
    assert [record.task_id for record in residual] == ["TEST-002"]
    assert [record.finding.missing_evidence for record in residual] == [
        [resource_bounds]
    ]
    assert todo_path.read_text(encoding="utf-8").count("## TEST-") == 2

    replay = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="TEST-",
        max_findings=1,
        scan_exclude_paths=["data"],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )
    assert replay == []
    assert todo_path.read_text(encoding="utf-8").count("## TEST-") == 2


def test_declared_path_evidence_requires_a_tracked_regular_file(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective.md"
    objective_path.write_text("# Objective Heap\n", encoding="utf-8")
    policy_dir = repo / "data" / "policy"
    policy_dir.mkdir(parents=True)
    tracked = policy_dir / "tracked.json"
    tracked.write_text('{"tracked":true}\n', encoding="utf-8")
    untracked = policy_dir / "untracked.json"
    untracked.write_text('{"tracked":false}\n', encoding="utf-8")
    symlink = policy_dir / "alias.json"
    symlink.symlink_to("tracked.json")
    _git(repo, "add", "objective.md", "data/policy/tracked.json", "data/policy/alias.json")
    _git(repo, "commit", "-m", "seed path evidence")

    evidence = evidence_index(
        repo,
        objective_path=objective_path,
        terms=[
            "data/policy/tracked.json",
            "data/policy/untracked.json",
            "data/policy/alias.json",
        ],
        scan_exclude_paths=["data"],
    )

    assert evidence == {
        "data/policy/tracked.json": ["data/policy/tracked.json (path)"],
        "data/policy/untracked.json": [],
        "data/policy/alias.json": [],
    }


def test_objective_scanner_enforces_source_protected_roots_in_repo_and_submodule(
    tmp_path,
    monkeypatch,
):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    protected_marker = "HSSL_SOURCE_PROTECTED_MARKER"
    artifact_source = repo / "artifacts" / "private_answer.py"
    fixture_source = (
        repo
        / "tests"
        / "unit_tests"
        / "ui"
        / "html_fixtures"
        / "private_answer.html"
    )
    artifact_source.parent.mkdir()
    fixture_source.parent.mkdir(parents=True)
    artifact_source.write_text(
        f"def {protected_marker}():\n    return 'private artifact'\n",
        encoding="utf-8",
    )
    fixture_source.write_text(
        f"<div>{protected_marker}</div>\n",
        encoding="utf-8",
    )
    protected_alias = repo / "src" / "public_answer.py"
    protected_alias.symlink_to(artifact_source)

    submodule_source = tmp_path / "protected-submodule-source"
    submodule_source.mkdir()
    _git(submodule_source, "init")
    _git(submodule_source, "checkout", "-b", "main")
    _git(submodule_source, "config", "user.name", "Test User")
    _git(submodule_source, "config", "user.email", "test@example.invalid")
    submodule_fixture = (
        submodule_source / "tests" / "model_fixtures" / "private_answer.py"
    )
    submodule_fixture.parent.mkdir(parents=True)
    submodule_fixture.write_text(
        f"def {protected_marker}():\n    return 'private submodule fixture'\n",
        encoding="utf-8",
    )
    _git(submodule_source, "add", ".")
    _git(submodule_source, "commit", "-m", "seed protected fixture")
    _git(
        repo,
        "-c",
        "protocol.file.allow=always",
        "submodule",
        "add",
        str(submodule_source),
        "vendor/protected-model",
    )
    checked_out_submodule_fixture = (
        repo
        / "vendor"
        / "protected-model"
        / "tests"
        / "model_fixtures"
        / "private_answer.py"
    )
    objective_path.write_text(
        f"""# Objective Heap

## TEST-G002 Require public evidence

- Status: active
- Parent:
- Fib priority: 1
- Track: test
- Priority: P0
- Bundle: objective/test/source-policy
- Goal: Keep source-protected evidence unavailable to the objective scanner.
- Evidence: {protected_marker}
- Outputs: src
- Validation: true
- Gap task: Produce public evidence.
""",
        encoding="utf-8",
    )
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed source-protected evidence")

    original_read_text = Path.read_text
    original_read_bytes = Path.read_bytes
    protected_paths = {
        artifact_source.resolve(),
        fixture_source.resolve(),
        checked_out_submodule_fixture.resolve(),
    }

    def guarded_read_text(path, *args, **kwargs):
        assert path.resolve() not in protected_paths
        return original_read_text(path, *args, **kwargs)

    def guarded_read_bytes(path, *args, **kwargs):
        assert path.resolve() not in protected_paths
        return original_read_bytes(path, *args, **kwargs)

    monkeypatch.setattr(Path, "read_text", guarded_read_text)
    monkeypatch.setattr(Path, "read_bytes", guarded_read_bytes)

    dataset_dir = tmp_path / "protected-policy-dataset"
    stats = {}
    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        embedding_min_score=2.0,
        dataset_dir=dataset_dir,
        dataset_id="source-policy-test",
        scan_stats=stats,
    )

    assert [finding.goal_id for finding in findings] == ["TEST-G002"]
    assert findings[0].missing_evidence == [protected_marker]
    assert stats["source_protected_scan_policy"]["deny_symlinks"] is True
    assert "artifacts" in stats["source_protected_scan_policy"]["deny_components"]
    dataset_rows = (
        dataset_dir / "source-policy-test.jsonl"
    ).read_text(encoding="utf-8")
    assert "private_answer" not in dataset_rows

    explicit_path_evidence = evidence_index(
        repo,
        objective_path=objective_path,
        terms=[
            "artifacts/private_answer.py",
            "src/public_answer.py",
            (
                "vendor/protected-model/tests/model_fixtures/"
                "private_answer.py"
            ),
        ],
        records=[],
    )
    assert explicit_path_evidence == {
        "artifacts/private_answer.py": [],
        "src/public_answer.py": [],
        (
            "vendor/protected-model/tests/model_fixtures/"
            "private_answer.py"
        ): [],
    }

    class ProtectedCachedRow(dict):
        def get(self, key, default=None):
            if key != "root_relative_path":
                raise AssertionError(
                    "protected cached evidence was inspected before denial"
                )
            return super().get(key, default)

    poisoned_cached_evidence = evidence_index(
        repo,
        objective_path=objective_path,
        terms=[protected_marker],
        embedding_min_score=2.0,
        records=[
            ProtectedCachedRow(
                root_relative_path="artifacts/private_answer.py"
            ),
            ProtectedCachedRow(
                root_relative_path=(
                    "vendor/protected-model/tests/model_fixtures/"
                    "private_answer.py"
                )
            ),
        ],
    )
    assert poisoned_cached_evidence == {protected_marker: []}


def test_tracked_file_inventory_failure_never_falls_back_to_filesystem_walk(
    tmp_path,
    monkeypatch,
):
    repo = tmp_path / "repo"
    protected = repo / "fixtures" / "private_answer.py"
    protected.parent.mkdir(parents=True)
    protected.write_text("secret = True\n", encoding="utf-8")

    def failed_git_inventory(*_args, **_kwargs):
        return subprocess.CompletedProcess(
            args=["git", "ls-files", "-z"],
            returncode=128,
            stdout=b"",
            stderr=b"not a Git worktree",
        )

    def forbidden_filesystem_walk(*_args, **_kwargs):
        raise AssertionError(
            "failed tracked-file inventory attempted a filesystem walk"
        )

    monkeypatch.setattr(subprocess, "run", failed_git_inventory)
    monkeypatch.setattr(Path, "rglob", forbidden_filesystem_walk)

    assert tracked_files(repo) == []


def test_ast_dataset_discards_protected_prior_rows_before_blob_reuse(tmp_path):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    source_blob = _git(repo, "rev-parse", ":src/runtime_router.py")
    protected_marker = "HSSL_PROTECTED_PRIOR_CACHE_PAYLOAD"
    previous_records = [
        {
            "record_schema_version": 2,
            "root_relative_path": "fixtures/private_answer.py",
            # A cache row is untrusted and must not be able to claim a public
            # source blob in order to transplant protected evidence.
            "blob_hash": source_blob,
            "source_sha1": source_blob,
            "evidence_text": protected_marker,
            "symbols_json": json.dumps([protected_marker]),
            "document_tokens_json": json.dumps([protected_marker.lower()]),
            "document_embedding_json": json.dumps([1.0]),
        }
    ]

    rows = collect_ast_dataset_records(
        repo,
        objective_path=objective_path,
        previous_records=previous_records,
    )

    serialized = json.dumps(rows, sort_keys=True)
    assert protected_marker not in serialized
    runtime_row = next(
        row
        for row in rows
        if row["root_relative_path"] == "src/runtime_router.py"
    )
    assert "CapabilityRouter" in runtime_row["evidence_text"]


def test_ast_dataset_recomputes_poisoned_benign_path_and_blob_rows(tmp_path):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    source_blob = _git(repo, "rev-parse", ":src/runtime_router.py")
    injected_text = "HSSL_INJECTED_BENIGN_CACHE_TEXT"
    injected_symbol = "HSSL_INJECTED_BENIGN_CACHE_SYMBOL"
    previous_records = [
        {
            "record_schema_version": 2,
            "root_relative_path": "src/runtime_router.py",
            "blob_hash": source_blob,
            "source_sha1": source_blob,
            "evidence_text": injected_text,
            "symbols_json": json.dumps([injected_symbol]),
            "document_tokens_json": json.dumps(
                [injected_text.lower(), injected_symbol.lower()]
            ),
            "document_embedding_json": json.dumps([1.0]),
            "ast_text": injected_text,
            "parse_elapsed_seconds": 99,
        }
    ]
    stats = {}

    rows = collect_ast_dataset_records(
        repo,
        objective_path=objective_path,
        previous_records=previous_records,
        scan_stats=stats,
    )

    serialized = json.dumps(rows, sort_keys=True)
    assert injected_text not in serialized
    assert injected_symbol not in serialized
    runtime_row = next(
        row
        for row in rows
        if row["root_relative_path"] == "src/runtime_router.py"
    )
    assert "CapabilityRouter" in runtime_row["evidence_text"]
    assert "capabilityrouter" in runtime_row["symbols_json"].lower()
    assert stats["reused_record_count"] == 0
    assert stats["parsed_record_count"] == len(rows)


def test_external_authority_goals_and_descendants_never_generate_local_work(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    objective_path.write_text(
        """# Objective Heap

## HSSL-G202 Pilot authorization gate without copied metadata

- Status: active
- Parent:
- Fib priority: 1
- Track: benchmark
- Priority: P0
- Goal: Await pilot authorization.
- Evidence: HSSL_EXTERNAL_GATE_ONLY
- Outputs: artifacts/pilot.json
- Validation: true

## HSSL-G204 Child of pilot authorization

- Status: active
- Parent: HSSL-G202
- Fib priority: 2
- Track: benchmark
- Priority: P0
- Goal: Run a child stage only after pilot authorization.
- Evidence: HSSL_EXTERNAL_CHILD_ONLY
- Outputs: artifacts/child.json
- Validation: true

## EXT-G001 Generic external gate

- Status: active
- Parent:
- Fib priority: 3
- Track: benchmark
- Priority: P0
- Completion authority: external
- Goal: Await generic external authorization.
- Evidence: HSSL_GENERIC_EXTERNAL_ONLY
- Outputs: artifacts/generic.json
- Validation: true

## EXT-G002 Child of generic external gate

- Status: active
- Parent: EXT-G001
- Fib priority: 4
- Track: benchmark
- Priority: P0
- Goal: Run a child stage only after generic authorization.
- Evidence: HSSL_GENERIC_EXTERNAL_CHILD_ONLY
- Outputs: artifacts/generic-child.json
- Validation: true

## LOCAL-G001 Independent local repair

- Status: active
- Parent:
- Fib priority: 5
- Track: benchmark
- Priority: P1
- Goal: Produce an independent local repair.
- Evidence: HSSL_LOCAL_REPAIR_ONLY
- Outputs: src/local_repair.py
- Validation: true
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective-heap.md")
    _git(repo, "commit", "-m", "seed external authority fences")

    stats = {}
    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=10,
        embedding_min_score=2.0,
        scan_stats=stats,
    )

    assert [finding.goal_id for finding in findings] == ["LOCAL-G001"]
    assert set(stats["external_authority_goal_ids"]) == {
        "EXT-G001",
        "HSSL-G202",
    }
    assert set(stats["external_authority_blocked_goal_ids"]) == {
        "EXT-G001",
        "EXT-G002",
        "HSSL-G202",
        "HSSL-G204",
    }

    decisions = {
        goal_id: {
            "verified": False,
            "state": "active",
            "actionable_reasons": ["missing_evidence"],
        }
        for goal_id in (
            "HSSL-G202",
            "HSSL-G204",
            "EXT-G001",
            "EXT-G002",
            "LOCAL-G001",
        )
    }
    proposals = objective_generation_proposals(
        objective_path=objective_path,
        completion_decisions=decisions,
    )
    assert {proposal.parent_goal_id for proposal in proposals} == {
        "LOCAL-G001"
    }
    generated_findings = objective_generation_task_findings(
        [proposal.to_dict() for proposal in proposals],
        repo_root=repo,
        objective_path=objective_path,
        generation_path=repo / "data" / "objective-generation.json",
        # The current generation pipeline admits only proposals bound to a
        # durable family/instance record.  Supplying that receipt explicitly
        # keeps this authority-fence test from relying on the removed legacy
        # fallback that treated an unrecorded proposal as executable work.
        gap_family_states={
            proposal.family_key: {
                "resolved": False,
                "outcome": "review_required",
                "instance_key": proposal.instance_key,
                "canonical_id": proposal.canonical_id,
                "occurrence": 1,
                "attempt_count": 1,
            }
            for proposal in proposals
        },
    )
    assert {finding.goal_id for finding in generated_findings} == {
        "LOCAL-G001"
    }

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=repo / "data" / "agent_supervisor" / "discovery",
        bundle_dir=repo / "data" / "agent_supervisor" / "objective_bundles",
        task_prefix="FENCE-",
        max_findings=10,
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )
    assert {record.finding.goal_id for record in records} == {
        "LOCAL-G001"
    }
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "HSSL-G202" not in todo_text
    assert "EXT-G001" not in todo_text


def test_durable_external_history_keeps_generic_goals_and_descendants_fenced(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    typed_evidence = json.dumps(
        [
            {
                "acceptance_criterion": "EXT-EVIDENCE",
                "provenance_cid": "bafkreibogusbutroutingonly",
                "metadata": {"external_operational_completion": True},
            }
        ],
        separators=(",", ":"),
    )
    objective_path.write_text(
        f"""# Objective Heap

## EXT-CID Prior authority CID with declaration removed

- Status: reopened
- External completion authority CID: bafkreiauthorityroutingonly
- Evidence: EXT-CID
- Outputs: src/external_cid.py
- Validation: true

## EXT-CID-CHILD Descendant of prior CID authority

- Status: active
- Parent: EXT-CID
- Evidence: EXT-CID-CHILD
- Outputs: src/external_cid_child.py
- Validation: true

## EXT-EVIDENCE Prior typed evidence with declaration removed

- Status: reopened
- Completion evidence records: {typed_evidence}
- Evidence: EXT-EVIDENCE
- Outputs: src/external_evidence.py
- Validation: true

## LOCAL-G001 Independent local repair

- Status: active
- Evidence: LOCAL-MISSING
- Outputs: src/local.py
- Validation: true
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective-heap.md")
    _git(repo, "commit", "-m", "seed sticky external history")
    stats = {}

    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=10,
        embedding_min_score=2.0,
        scan_stats=stats,
        trust_recorded_external_completion=False,
    )

    assert [finding.goal_id for finding in findings] == ["LOCAL-G001"]
    assert set(stats["external_authority_goal_ids"]) == {
        "EXT-CID",
        "EXT-EVIDENCE",
    }
    assert set(stats["external_authority_blocked_goal_ids"]) == {
        "EXT-CID",
        "EXT-CID-CHILD",
        "EXT-EVIDENCE",
    }
    generated = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=repo / "discovery",
        bundle_dir=repo / "bundles",
        task_prefix="STICKY-",
        max_findings=10,
        persist_ast_dataset=False,
        write_todo_vector_index=False,
        trust_recorded_external_completion=False,
    )
    assert [record.finding.goal_id for record in generated] == ["LOCAL-G001"]


def test_hsslev0097b20_empty_symbol_normalizations_are_not_ast_evidence(tmp_path):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    records = [
        {
            "root_relative_path": "assets/minified.js",
            "evidence_text": "const unrelated = true;",
            "symbols_json": json.dumps(["", " \t", "e", "v", "$"]),
            "document_tokens_json": "[]",
            "document_embedding_json": "[]",
        }
    ]

    evidence = evidence_index(
        repo,
        objective_path=objective_path,
        terms=["HSSLEV0097B20"],
        embedding_min_score=2.0,
        records=records,
    )

    assert evidence == {"HSSLEV0097B20": []}


def test_short_ast_symbols_are_not_substring_evidence_for_unique_markers(tmp_path):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    records = [
        {
            "root_relative_path": "src/unrelated.py",
            "evidence_text": "short identifiers only",
            "symbols_json": json.dumps(["le", "ssl", "hssl", "hsslev"]),
            "document_tokens_json": "[]",
            "document_embedding_json": "[]",
        }
    ]

    evidence = evidence_index(
        repo,
        objective_path=objective_path,
        terms=["HSSLEV0097B20"],
        embedding_min_score=2.0,
        records=records,
    )

    assert evidence == {"HSSLEV0097B20": []}


def test_empty_symbol_filter_preserves_nonempty_ast_and_exact_evidence(tmp_path):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    records = [
        {
            "root_relative_path": "src/proof.py",
            "evidence_text": "literal objective receipt",
            "symbols_json": json.dumps(["e", "$", "CapabilityRouter.dispatch_task"]),
            "document_tokens_json": "[]",
            "document_embedding_json": "[]",
        }
    ]

    evidence = evidence_index(
        repo,
        objective_path=objective_path,
        terms=["CapabilityRouter.dispatch_task", "literal objective receipt"],
        embedding_min_score=2.0,
        records=records,
    )

    assert evidence["CapabilityRouter.dispatch_task"] == ["src/proof.py (ast)"]
    assert evidence["literal objective receipt"] == ["src/proof.py (exact)"]


def test_empty_symbol_only_source_keeps_unique_objective_evidence_missing(tmp_path):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    minified_source = repo / "src" / "minified.js"
    minified_source.write_text(
        "const e = 1; let le = e; var ssl = le; let hssl = ssl;\n",
        encoding="utf-8",
    )
    objective_path.write_text(
        """# Objective Heap

## HSSL-G009 Reject empty AST symbols as objective evidence

- Status: active
- Parent: HSSL-G000
- Priority: P0
- Track: benchmark-protocol
- Bundle: objective/hssl/supervisor-compatibility
- Goal: Prevent token-empty AST symbols from satisfying objective evidence.
- Evidence: HSSLEV0097B20
- Validation: true
""",
        encoding="utf-8",
    )
    _git(repo, "add", "src/minified.js", "objective-heap.md")
    _git(repo, "commit", "-m", "seed token-empty AST symbols")

    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        embedding_min_score=2.0,
    )

    assert len(findings) == 1
    assert findings[0].goal_id == "HSSL-G009"
    assert findings[0].missing_evidence == ["HSSLEV0097B20"]
    assert findings[0].present_evidence == {}


def test_objective_scan_assigns_shared_evidence_to_deepest_refinement(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    objective_path = repo / "objective.md"
    objective_path.write_text(
        """# Goals

## G1 Parent
- Status: active
- Evidence: shared-proof-requirement
- Validation: true

## G1.1 Focused child
- Status: active
- Parent: G1
- Evidence: shared-proof-requirement
- Validation: true
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective.md")

    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=5,
    )

    assert [finding.goal_id for finding in findings] == ["G1.1"]
    assert findings[0].missing_evidence == ["shared-proof-requirement"]
    assert findings[0].dedupe_key.startswith("objective-evidence-obligation/v1/")


def test_objective_scan_does_not_force_provisional_goal_back_to_implementation(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    objective_path = repo / "objective.md"
    objective_path.write_text(
        """# Goals

## G1 Awaiting completion gate
- Status: provisionally_complete
- Evidence: STILL_REQUIRES_TYPED_PROOF
- Validation: true
""",
        encoding="utf-8",
    )
    proof_path = repo / "src" / "proof.py"
    proof_path.parent.mkdir()
    proof_path.write_text(
        "STILL_REQUIRES_TYPED_PROOF = True\n",
        encoding="utf-8",
    )
    _git(repo, "add", "objective.md", "src/proof.py")

    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=5,
        force_goal_ids=["G1"],
    )

    assert findings == []


def test_objective_scan_does_not_share_ownership_between_unrelated_lineages(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    objective_path = repo / "objective.md"
    objective_path.write_text(
        """# Goals

## G1 First root
- Status: active
- Evidence: shared-generic-proof
- Validation: true

## G2 Second root
- Status: active
- Evidence: shared-generic-proof
- Validation: true
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective.md")

    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=5,
    )

    assert {finding.goal_id for finding in findings} == {"G1", "G2"}


def test_refinement_does_not_repeat_an_ancestor_evidence_obligation(tmp_path):
    objective_path = tmp_path / "objective.md"
    objective_path.write_text(
        """# Goals

## G1 Parent
- Status: active
- Evidence: shared-proof-requirement

## G1.1 Focused child
- Status: active
- Parent: G1
- Evidence: shared-proof-requirement
""",
        encoding="utf-8",
    )
    finding = ObjectiveFinding(
        fingerprint="child-gap",
        goal_id="G1.1",
        title="Focused child",
        summary="Close child gap",
        priority="P1",
        track="test",
        missing_evidence=["shared-proof-requirement"],
        present_evidence={},
        evidence_methods=[],
        objective_path=str(objective_path),
        outputs=["src"],
        validation="true",
    )

    result = append_refinement_goals(objective_path, [finding])

    assert result.appended_goal_ids == []
    assert objective_path.read_text(encoding="utf-8").count("## G1.1") == 1


def test_refinement_admits_complete_candidate_with_bound_acceptance(tmp_path):
    objective_path = tmp_path / "objective.md"
    objective_path.write_text(
        """# Goals

## VFS-G000 Root

- Status: active
- Parent:
- Evidence: root-proof
""",
        encoding="utf-8",
    )
    finding = ObjectiveFinding(
        fingerprint="root-contract-gap",
        goal_id="VFS-G000",
        title="Root",
        summary="Close root contract gap",
        priority="P0",
        track="contract",
        missing_evidence=["vfs/contract-proof@1"],
        present_evidence={},
        evidence_methods=[],
        objective_path=str(objective_path),
        outputs=["src/vfs.py", "test/test_vfs.py"],
        validation="python -m pytest test/test_vfs.py -q",
        acceptance_subset=[
            "The VFS contract receipt is current and bound to the tested implementation."
        ],
    )

    result = append_refinement_goals(objective_path, [finding])

    assert result.appended_goal_ids == ["VFS-G001"]
    goals = {
        goal.goal_id: goal
        for goal in parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    }
    assert goals["VFS-G001"].fields["acceptance"] == (
        "The VFS contract receipt is current and bound to the tested implementation."
    )


def test_refinement_rejects_invalid_rendered_candidate_without_mutation(
    tmp_path,
    monkeypatch,
):
    objective_path = tmp_path / "objective.md"
    original = b"""# Goals

## VFS-G000 Root

- Status: active
- Parent:
- Evidence: root-proof
"""
    objective_path.write_bytes(original)
    finding = ObjectiveFinding(
        fingerprint="root-contract-gap",
        goal_id="VFS-G000",
        title="Root",
        summary="Close root contract gap",
        priority="P0",
        track="contract",
        missing_evidence=["vfs/contract-proof@1"],
        present_evidence={},
        evidence_methods=[],
        objective_path=str(objective_path),
        outputs=["src/vfs.py", "test/test_vfs.py"],
        validation="python -m pytest test/test_vfs.py -q",
        acceptance_subset=["The VFS contract receipt is current."],
    )
    real_render = objective_tracker_module.render_goal_block

    def render_without_acceptance(**kwargs):
        rendered = real_render(**kwargs)
        return "\n".join(
            line
            for line in rendered.splitlines()
            if not line.startswith("- Acceptance:")
        ) + "\n"

    monkeypatch.setattr(
        objective_tracker_module,
        "render_goal_block",
        render_without_acceptance,
    )

    with pytest.raises(
        ValueError,
        match=r"VFS-G001 is missing fields: .*acceptance",
    ):
        append_refinement_goals(objective_path, [finding])

    assert objective_path.read_bytes() == original


def test_refinement_atomic_rewrite_failure_preserves_original_bytes(
    tmp_path,
    monkeypatch,
):
    objective_path = tmp_path / "objective.md"
    original = b"""# Goals

## VFS-G000 Root

- Status: active
- Parent:
- Evidence: root-proof
"""
    objective_path.write_bytes(original)
    finding = ObjectiveFinding(
        fingerprint="root-contract-gap",
        goal_id="VFS-G000",
        title="Root",
        summary="Close root contract gap",
        priority="P0",
        track="contract",
        missing_evidence=["vfs/contract-proof@1"],
        present_evidence={},
        evidence_methods=[],
        objective_path=str(objective_path),
        outputs=["src/vfs.py", "test/test_vfs.py"],
        validation="python -m pytest test/test_vfs.py -q",
        acceptance_subset=["The VFS contract receipt is current."],
    )

    def interrupted_rewrite(_path, _text):
        raise InterruptedError("simulated interruption before atomic rename")

    monkeypatch.setattr(
        objective_tracker_module,
        "_atomic_rewrite",
        interrupted_rewrite,
    )

    with pytest.raises(InterruptedError, match="simulated interruption"):
        append_refinement_goals(objective_path, [finding])

    assert objective_path.read_bytes() == original


def test_refinement_without_acceptance_subset_fails_closed(tmp_path):
    objective_path = tmp_path / "objective.md"
    original = b"""# Goals

## VFS-G000 Root

- Status: active
- Parent:
- Evidence: root-proof
"""
    objective_path.write_bytes(original)
    finding = ObjectiveFinding(
        fingerprint="root-contract-gap",
        goal_id="VFS-G000",
        title="Root",
        summary="Close root contract gap",
        priority="P0",
        track="contract",
        missing_evidence=["vfs/contract-proof@1"],
        present_evidence={},
        evidence_methods=[],
        objective_path=str(objective_path),
        outputs=["src/vfs.py", "test/test_vfs.py"],
        validation="python -m pytest test/test_vfs.py -q",
    )

    with pytest.raises(ValueError, match="has no acceptance subset"):
        append_refinement_goals(objective_path, [finding])

    assert objective_path.read_bytes() == original


def test_objective_gap_scope_limits_forced_refinement_family(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    objective_path = repo / "objective.md"
    objective_path.write_text(
        """# Goals

## VFS-G001 First family

- Status: active
- Parent:
- Evidence: missing-first-proof
- Acceptance: First proof is current.

## VFS-G002 Forced family

- Status: active
- Parent:
- Evidence: missing-forced-proof
- Acceptance: Forced proof is current.
""",
        encoding="utf-8",
    )

    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=8,
        force_goal_ids=["VFS-G002"],
        scope_goal_ids=["VFS-G002"],
    )

    assert findings
    assert {finding.goal_id for finding in findings} == {"VFS-G002"}


def test_generate_objective_todos_limits_generation_to_goal_scope(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    objective_path.write_text(
        """# Goals

## VFS-G001 First family

- Status: active
- Parent:
- Track: verification
- Priority: P1
- Bundle: objective/verification/first
- Evidence: missing-first-proof
- Outputs: artifacts/first-proof.json
- Validation: true

## VFS-G002 Scoped family

- Status: active
- Parent:
- Track: verification
- Priority: P1
- Bundle: objective/verification/scoped
- Evidence: missing-scoped-proof
- Outputs: artifacts/scoped-proof.json
- Validation: true
""",
        encoding="utf-8",
    )

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=repo / "state" / "discovery",
        bundle_dir=repo / "state" / "bundles",
        task_prefix="SCOPE-",
        max_findings=8,
        scope_goal_ids=["VFS-G002"],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert records
    assert {record.finding.goal_id for record in records} == {"VFS-G002"}
    board = todo_path.read_text(encoding="utf-8")
    assert "- Goal id: VFS-G002" in board
    assert "- Goal id: VFS-G001" not in board


def test_objective_todo_result_duplicate_rescan_preserves_goal_scope(
    tmp_path,
    monkeypatch,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    duplicate_scan_kwargs = []

    monkeypatch.setattr(
        objective_graph_module,
        "generate_objective_todos",
        lambda **_kwargs: [],
    )

    def scoped_duplicate_scan(*_args, **kwargs):
        duplicate_scan_kwargs.append(kwargs)
        return []

    monkeypatch.setattr(
        objective_graph_module,
        "scan_objective_gaps",
        scoped_duplicate_scan,
    )

    result = objective_graph_module.generate_objective_todos_result(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        max_findings=1,
        seen_fingerprints=["already-seen"],
        scope_goal_ids=["VFS-G002"],
    )

    assert result.terminal_reason.value == "exhausted"
    assert len(duplicate_scan_kwargs) == 1
    assert duplicate_scan_kwargs[0]["scope_goal_ids"] == ["VFS-G002"]


def test_objective_daemon_parser_accepts_repeatable_goal_scope():
    args = build_objective_daemon_arg_parser().parse_args(
        [
            "--scope-goal-id",
            "VFS-G001",
            "--scope-goal-id",
            "VFS-G002,VFS-G003",
        ]
    )

    assert args.scope_goal_id == ["VFS-G001", "VFS-G002,VFS-G003"]


def test_launch_readiness_generated_goals_include_acceptance(tmp_path):
    objective_path = tmp_path / "objective.md"
    objective_path.write_text(
        """# Goals

## VFS-G000 Root

- Status: active
- Parent:
- Evidence: root-proof
""",
        encoding="utf-8",
    )

    result = append_launch_readiness_goals(
        objective_path,
        repo_root=tmp_path,
        max_goals=2,
        goal_prefix="VFS-G",
    )

    goals = {
        goal.goal_id: goal
        for goal in parse_goal_heap(objective_path.read_text(encoding="utf-8"))
    }
    assert len(result.appended_goal_ids) == 2
    assert all(
        str(goals[goal_id].fields.get("acceptance") or "").strip()
        for goal_id in result.appended_goal_ids
    )


def test_completion_gate_work_identity_ignores_actionable_prose_churn():
    diagnostics = {
        "uncovered_criteria": ["criterion one"],
        "stale_evidence": [],
        "analyzer_health": {"passed": False},
        "exhaustion_quorum": {"satisfied": False},
        "reopen_reasons": [],
    }
    first = {
        "reason_codes": ["missing_criterion_evidence", "analyzer_health_missing"],
        "actionable_reasons": ["Write a fresh proof for criterion one."],
        "diagnostics": diagnostics,
    }
    reworded = {
        **first,
        "actionable_reasons": ["Criterion one needs current proof."],
    }

    assert completion_gate_work_terms(first) == completion_gate_work_terms(reworded)
    assert completion_gate_work_terms(first) == (
        "completion criterion coverage",
        "completion analyzer health",
        "completion exhaustion quorum",
    )


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


def test_objective_finding_task_identity_binds_source_contract_not_alias_or_provenance():
    finding = ObjectiveFinding(
        fingerprint="finding:world-aid-storage",
        goal_id="WORLDCOIN-G002",
        title="Freeze the storage integration boundary",
        summary="Close objective gap: Freeze the storage integration boundary",
        priority="P0",
        track="world-aid-discovery",
        missing_evidence=["objective validation repair"],
        present_evidence={},
        evidence_methods=[],
        objective_path="docs/planning/WORLDCOIN_HUMAN_AID_OBJECTIVE_HEAP.md",
        outputs=["docs/reports/audit.md", "data/worldcoin/audit.json"],
        validation="python -m pytest -q tests/world_aid/test_audit.py",
        goal="Record the reviewed storage integration boundary.",
        refinement="Inventory Python/DuckDB inputs and preserve the single-writer boundary.",
        gap_task="Repair the objective validation evidence.",
        parent_goal_ids=["WORLDCOIN-G001"],
        predicted_files=["docs/reports/audit.md", "data/worldcoin/audit.json"],
        interfaces=["WorldAidDuckDBWriter", "wallet repository"],
        submodules=["ipfs_datasets_py"],
        generated_artifacts=["data/worldcoin/audit.json"],
        conflict_policy="Keep unrelated wallet work intact.",
        allow_concurrent_with=["WORLDCOIN-G041"],
    )

    original = objective_finding_task_identity("WORLDCOIN-AUTO-001", finding)
    alias = objective_finding_task_identity(
        "LOCAL-999",
        replace(
            finding,
            objective_path=(
                "/tmp/generated-root/discovery/"
                "WORLDCOIN_HUMAN_AID_OBJECTIVE_HEAP.md"
            ),
            outputs=list(reversed(finding.outputs)),
            missing_evidence=["  OBJECTIVE   validation repair  "],
        ),
    )
    revised = objective_finding_task_identity(
        "WORLDCOIN-AUTO-001",
        replace(
            finding,
            refinement=(
                "Inventory Python/PostgreSQL inputs and preserve the "
                "service boundary."
            ),
        ),
    )

    assert alias.canonical_task_key == original.canonical_task_key
    assert alias.canonical_task_cid == original.canonical_task_cid
    assert alias.namespaced_alias != original.namespaced_alias
    assert revised.canonical_task_key != original.canonical_task_key
    assert revised.canonical_task_cid != original.canonical_task_cid
    assert finding.fingerprint == "finding:world-aid-storage"


def test_objective_finding_task_identity_changes_with_execution_contract():
    finding = ObjectiveFinding(
        fingerprint="finding:runtime-contract",
        goal_id="G-RUNTIME",
        title="Verify the runtime",
        summary="Close objective gap: Verify the runtime",
        priority="P1",
        track="runtime",
        missing_evidence=["runtime receipt"],
        present_evidence={},
        evidence_methods=[],
        objective_path="docs/objectives.md",
        outputs=["runtime.lock"],
        validation="python -m pytest -q tests/test_runtime.py",
        refinement="Verify the selected runtime offline.",
    )
    original = objective_finding_task_identity("AUTO-001", finding)

    changed_output = objective_finding_task_identity(
        "AUTO-001",
        replace(finding, outputs=["runtime-v2.lock"]),
    )
    changed_validation = objective_finding_task_identity(
        "AUTO-001",
        replace(
            finding,
            validation="python -m pytest -q tests/test_runtime_v2.py",
        ),
    )

    assert changed_output.canonical_task_cid != original.canonical_task_cid
    assert changed_validation.canonical_task_cid != original.canonical_task_cid


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


def test_objective_graph_projects_lifecycle_and_completion_evidence() -> None:
    goals = parse_goal_heap(
        """# Objective Heap

## G10.S3 Evidence-backed completion

- Status: provisionally_complete
- Evidence: criterion one, criterion two
- Acceptance criterion: criterion one
- Producing task or scan: REF-206
- Validation receipt: bafy-validation
- Repository tree: sha256:tree
- Freshness: fresh
- Provenance CID: bafy-provenance

## G10.S3.1 Reopened child

- Status: reopened
- Parent: G10.S3
- Evidence: child criterion

## G10.S3.2 Legacy completed child

- Status: completed
- Parent: G10.S3
- Evidence: legacy criterion
"""
    )

    graph = goal_graph(goals)

    assert graph["node_details"]["G10.S3"] == {
        "goal_id": "G10.S3",
        "title": "Evidence-backed completion",
        "status": "provisionally_complete",
        "lifecycle_state": "provisionally_complete",
        "schedulable": True,
        "terminal": False,
        "parents": [],
        "required_evidence": ["criterion one", "criterion two"],
        "completion_evidence": {
            "acceptance_criterion": "criterion one",
            "producer": "REF-206",
            "validation_receipt": "bafy-validation",
            "repository_tree": "sha256:tree",
            "freshness": "fresh",
            "provenance_cid": "bafy-provenance",
        },
    }
    assert graph["node_details"]["G10.S3.1"]["lifecycle_state"] == "reopened"
    assert graph["node_details"]["G10.S3.1"]["schedulable"] is True
    assert graph["node_details"]["G10.S3.2"]["lifecycle_state"] == "verified_complete"
    assert graph["node_details"]["G10.S3.2"]["terminal"] is True
    assert graph["state_counts"] == {
        "provisionally_complete": 1,
        "reopened": 1,
        "verified_complete": 1,
    }
    assert graph["schedulable_goal_ids"] == ["G10.S3", "G10.S3.1"]
    assert graph["terminal_goal_ids"] == ["G10.S3.2"]
    assert {node["acceptance_criterion"] for node in graph["evidence_nodes"]} == {
        "criterion one",
        "criterion two",
        "child criterion",
        "legacy criterion",
    }
    assert {edge["kind"] for edge in graph["evidence_edges"]} == {"requires_evidence"}


def test_objective_schedule_includes_actionable_nonterminal_states() -> None:
    goals = parse_goal_heap(
        """# Objective Heap

## G-A Active
- Status: active
- Fib priority: 1

## G-P Provisional
- Status: provisionally_complete
- Fib priority: 2

## G-V Verified
- Status: verified_complete
- Fib priority: 3

## G-I Inconclusive
- Status: analysis_inconclusive
- Fib priority: 4

## G-B Blocked
- Status: blocked
- Fib priority: 5

## G-R Reopened
- Status: reopened
- Fib priority: 6
"""
    )

    schedule = objective_heap_schedule(goals)

    assert [record.goal_id for record in schedule] == ["G-A", "G-P", "G-I", "G-R"]


def test_objective_graph_links_persisted_completion_receipts() -> None:
    records = json.dumps(
        [
            {
                "acceptance_criterion": "criterion one",
                "producing_task_or_scan": "REF-206",
                "validation_receipt": "bafy-validation",
                "repository_tree": "sha256:tree",
                "freshness": True,
                "provenance_cid": "bafy-provenance",
            }
        ],
        separators=(",", ":"),
    )
    goals = parse_goal_heap(
        "\n".join(
            [
                "# Objective Heap",
                "",
                "## G10.S3 Persisted evidence",
                "",
                "- Status: provisionally_complete",
                "- Evidence: criterion one",
                f"- Completion evidence records: {records}",
            ]
        )
    )

    graph = goal_graph(goals)
    proof = next(node for node in graph["evidence_nodes"] if node.get("kind") == "completion_evidence")

    assert proof["producing_task_or_scan"] == "REF-206"
    assert proof["repository_tree"] == "sha256:tree"
    assert proof["provenance_cid"] == "bafy-provenance"
    assert any(edge["kind"] == "supported_by" and edge["to"] == proof["id"] for edge in graph["evidence_edges"])


def _truthful_completion_gate(identity, *, criterion: str, observed_at: str) -> dict[str, object]:
    binding = {
        "repository_id": identity.repository_id,
        "tree_id": identity.tree_id,
    }
    return {
        "coverage": {
            "verified": True,
            "repository_tree": identity.tree_id,
            "evaluated_at": observed_at,
            "criteria": [{"criterion": criterion, "status": "verified"}],
        },
        "analyzer_health": {
            "status": "healthy",
            "healthy": True,
            "safe_for_completion_reasoning": True,
            "exhaustive": True,
        },
        "exhaustion_quorum": {
            "satisfied": True,
            "required_members": 2,
            "member_count": 2,
            "binding": binding,
            "members": [
                {
                    "member_id": "normal-scan",
                    "evidence_channel": "exhaustive",
                    "receipt_cid": "bafy-normal-scan",
                    "scan_mode": "exhaustive",
                    "analyzer_version": "objective-graph/v1",
                    "passed": True,
                    "analyzer_health": {"status": "healthy", "healthy": True},
                    "exhaustive": True,
                    "safe_for_completion_reasoning": True,
                    "conclusive": True,
                    "contradicted": False,
                    "finished_at": observed_at,
                    "binding": binding,
                },
                {
                    "member_id": "independent-audit",
                    "evidence_channel": "audit",
                    "receipt_cid": "bafy-independent-audit",
                    "scan_mode": "audit",
                    "analyzer_version": "objective-graph/v1",
                    "passed": True,
                    "analyzer_health": {"status": "healthy", "healthy": True},
                    "exhaustive": True,
                    "safe_for_completion_reasoning": True,
                    "conclusive": True,
                    "contradicted": False,
                    "finished_at": observed_at,
                    "binding": binding,
                },
            ],
        },
    }


def test_restart_after_legacy_migration_preserves_lineage_quorum_and_dependencies(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## G10.S4 Legacy parent completion

- Status: completed
- Acceptance: current API proof
- Evidence: current API proof
- Validation: test -f proof.txt

## G10.S4.1 Verified prerequisite

- Status: verified_complete
- Parent: G10.S4
- Acceptance: prerequisite proof
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Drained task board\n", encoding="utf-8")
    (repo / "proof.txt").write_text("current API proof\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed legacy objective")

    observed_at = "2026-07-22T12:00:00+00:00"
    identity = completion_tree_identity(repo, objective_path=objective_path)
    # A rollout-era record deliberately uses the v0 aliases.  Migration must
    # preserve its source lineage rather than replacing it with an optimistic
    # modern receipt.
    legacy_evidence = {
        "version": 0,
        "criterion": "current API proof",
        "task_id": "REF-208",
        "validation": {"attempted": True, "passed": True},
        "tree_identity": identity.tree_id,
        "repository_id": identity.repository_id,
        "fresh": True,
        "generated_at": observed_at,
        "receipt_cid": "bafy-original-proof",
    }
    gate = _truthful_completion_gate(
        identity,
        criterion="current API proof",
        observed_at=observed_at,
    )

    migrated = migrate_legacy_objective_goals(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_evidence_records={"G10.S4": [legacy_evidence]},
        completion_gate_records={"G10.S4": gate},
        now=observed_at,
    )

    assert migrated.verified_goal_ids == ["G10.S4"]
    persisted_text = objective_path.read_text(encoding="utf-8")
    restarted_goals = parse_goal_heap(persisted_text)
    restarted_parent = next(goal for goal in restarted_goals if goal.goal_id == "G10.S4")
    persisted_record = restarted_parent.completion_evidence_records[0]
    assert restarted_parent.status == "verified_complete"
    assert persisted_record["producing_task_or_scan"] == "REF-208"
    assert persisted_record["provenance_cid"] == "bafy-original-proof"
    assert persisted_record["metadata"]["source_schema_version"] == 0
    persisted_quorum = json.loads(restarted_parent.fields["exhaustion_quorum"])
    assert persisted_quorum["satisfied"] is True
    assert persisted_quorum["required_members"] == 2
    assert persisted_quorum["member_count"] == 2
    persisted_gate = json.loads(restarted_parent.fields["completion_gate_record"])
    assert persisted_gate == gate
    assert {
        (member["member_id"], member["receipt_cid"], member["evidence_channel"])
        for member in persisted_gate["exhaustion_quorum"]["members"]
    } == {
        ("normal-scan", "bafy-normal-scan", "exhaustive"),
        ("independent-audit", "bafy-independent-audit", "audit"),
    }
    assert all(
        member["binding"] == gate["exhaustion_quorum"]["binding"]
        for member in persisted_gate["exhaustion_quorum"]["members"]
    )

    restarted_graph = goal_graph(restarted_goals)
    assert restarted_graph["terminal_goal_ids"] == ["G10.S4", "G10.S4.1"]
    assert {tuple((edge["from"], edge["to"], edge["kind"])) for edge in restarted_graph["edges"]} == {
        ("G10.S4", "G10.S4.1", "refines")
    }
    proof = next(
        item
        for item in restarted_graph["evidence_nodes"]
        if item.get("kind") == "completion_evidence"
    )
    assert proof["producing_task_or_scan"] == "REF-208"
    assert proof["provenance_cid"] == "bafy-original-proof"

    # Rebuild the bounded status projection only from durable markdown fields,
    # as a new supervisor process would do after losing all in-memory objects.
    restarted_diagnostic = {
        "state": restarted_parent.lifecycle_state_value,
        "confidence": float(restarted_parent.fields["completion_confidence"]),
        "uncovered_criteria": json.loads(restarted_parent.fields["uncovered_criteria"]),
        "stale_evidence": json.loads(restarted_parent.fields["stale_evidence"]),
        "analyzer_health": json.loads(restarted_parent.fields["analyzer_health"]),
        "exhaustion_quorum": json.loads(restarted_parent.fields["exhaustion_quorum"]),
        "reopen_reasons": json.loads(restarted_parent.fields["reopen_reasons"]),
    }
    projection = build_goal_completion_projection(
        {"G10.S4": restarted_diagnostic},
        migration=migrated,
    )
    operator_row = projection["by_goal_id"]["G10.S4"]
    assert operator_row["lifecycle_state"] == "verified_complete"
    assert operator_row["confidence"] == 1.0
    assert operator_row["analyzer_health"]["status"] == "healthy"
    assert operator_row["analyzer_health"]["passed"] is True
    assert operator_row["analyzer_health"]["evidence"] == {
        "status": "healthy",
        "healthy": True,
        "safe_for_completion_reasoning": True,
        "exhaustive": True,
    }
    assert operator_row["exhaustion_quorum"] == persisted_quorum
    assert operator_row["uncovered_criteria"] == []
    assert operator_row["stale_evidence"] == []

    # Dependency inputs are another restart boundary.  JSON round-tripping
    # must retain the exact edge and its blocked/claimable scheduler meaning.
    task_state_path = repo / "task-dependencies.json"
    task_state_path.write_text(
        json.dumps(
            [
                {"task_id": "REF-208", "task_cid": "cid-proof", "goal_id": "G10.S4.1"},
                {
                    "task_id": "REF-212",
                    "task_cid": "cid-regression",
                    "goal_id": "G10.S4",
                    "depends_on": ["REF-208"],
                },
            ]
        ),
        encoding="utf-8",
    )
    restarted_dependencies = materialize_task_dependency_dag(
        json.loads(task_state_path.read_text(encoding="utf-8")),
        now=0,
    )
    assert [edge.to_dict() for edge in restarted_dependencies.edges] == [
        {
            "source_task_cid": "cid-proof",
            "target_task_cid": "cid-regression",
            "kind": "goal",
                "provenance": {
                    "field": "depends_on",
                    "value": "REF-208",
                "resolution": "task_alias",
                "source_task_id": "REF-208",
                "target_task_id": "REF-212",
            },
        }
    ]
    restarted_schedule = {
        row.task_cid: row for row in restarted_dependencies.schedule
    }
    assert restarted_schedule["cid-proof"].claimable is True
    assert restarted_schedule["cid-regression"].claimable is False
    assert restarted_schedule["cid-regression"].blocking_task_cids == ["cid-proof"]

    replay = migrate_legacy_objective_goals(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        now="2026-07-22T12:01:00+00:00",
    )
    assert replay.changed is False
    assert replay.candidate_goal_ids == []
    assert objective_path.read_text(encoding="utf-8") == persisted_text


def test_changed_tree_reopens_goal_and_refills_despite_historical_fingerprint(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective.md"
    todo_path = repo / "todo.md"
    objective_path.write_text(
        """# Objective Heap

## G10.S4 Truthful completion

- Status: completed
- Priority: P0
- Track: g10
- Bundle: refactor/g10/g10-s4
- Acceptance: runtime proof marker
- Evidence: runtime proof marker
- Outputs: proof.txt, tests
- Validation: test -f proof.txt
- Gap task: Restore the runtime proof and its regression test.
""",
        encoding="utf-8",
    )
    todo_path.write_text("# Drained task board\n", encoding="utf-8")
    proof_path = repo / "proof.txt"
    proof_path.write_text("runtime proof marker\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed verified surface")

    verified_at = "2026-07-22T12:00:00+00:00"
    old_identity = completion_tree_identity(repo, objective_path=objective_path)
    evidence = CompletionEvidence(
        acceptance_criterion="runtime proof marker",
        producing_task_or_scan="REF-208",
        validation_receipt={"attempted": True, "passed": True},
        repository_id=old_identity.repository_id,
        repository_tree=old_identity.tree_id,
        freshness=True,
        observed_at=verified_at,
        provenance_cid="bafy-proof-before-regression",
        validation_passed=True,
    )
    gate = _truthful_completion_gate(
        old_identity,
        criterion="runtime proof marker",
        observed_at=verified_at,
    )
    migration = migrate_legacy_objective_goals(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_evidence_records={"G10.S4": [evidence]},
        completion_gate_records={"G10.S4": gate},
        now=verified_at,
    )
    assert migration.verified_goal_ids == ["G10.S4"]

    verified_goal = parse_goal_heap(objective_path.read_text(encoding="utf-8"))[0]
    historical_fingerprint = objective_fingerprint(
        verified_goal,
        ["runtime proof marker"],
    )

    # A later commit removes the evidence from the repository.  The old scan
    # fingerprint and its once-valid receipts remain durable history, but are
    # bound to a different tree and therefore cannot certify this one.
    proof_path.write_text("regressed surface\n", encoding="utf-8")
    _git(repo, "add", "proof.txt")
    _git(repo, "commit", "-m", "introduce later regression")
    result = reconcile_objective_goal_completion(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        now="2026-07-22T12:05:00+00:00",
    )

    assert result.reopened_goal_ids == ["G10.S4"]
    decision = result.decisions["G10.S4"]
    assert decision["state"] == "reopened"
    assert "repository_tree_mismatch" in decision["reason_codes"]
    reopened_goal = parse_goal_heap(objective_path.read_text(encoding="utf-8"))[0]
    assert reopened_goal.status == "reopened"
    assert reopened_goal.is_schedulable is True

    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    suppressed = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="REFILL-",
        max_findings=1,
        seen_fingerprints=[historical_fingerprint],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )
    assert suppressed == []

    refilled = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="REFILL-",
        max_findings=1,
        seen_fingerprints=[historical_fingerprint],
        force_goal_ids=["G10.S4"],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert [record.finding.fingerprint for record in refilled] == [historical_fingerprint]
    assert [record.finding.goal_id for record in refilled] == ["G10.S4"]
    assert "## REFILL-001 Close objective gap" in todo_path.read_text(encoding="utf-8")


def test_forced_blocked_goal_materializes_review_only_without_submission_or_lane(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    objective_path = repo / "objective-heap.md"
    todo_path = repo / "todo.md"
    discovery_dir = repo / "discovery"
    bundle_dir = repo / "objective-bundles"
    todo_path.write_text("# Review taskboard\n", encoding="utf-8")
    objective_path.write_text(
        """# Objective Heap

## G-BLOCKED Gate 0B review preparation

- Status: blocked
- Parent:
- Fib priority: 1
- Priority: P0
- Track: gate-review
- Bundle: objective/gate/review
- Goal: Preserve the review record without authorizing execution.
- Evidence: signed Gate 0B approval
- Outputs: data/gate0b/review.json
- Validation: test -f data/gate0b/review.json

## G-DONE Verified implementation

- Status: verified_complete
- Parent:
- Fib priority: 2
- Priority: P1
- Track: gate-review
- Bundle: objective/gate/done
- Goal: Keep verified work excluded from regeneration.
- Evidence: verified completion receipt
- Outputs: data/gate0b/completed.json
- Validation: test -f data/gate0b/completed.json
""",
        encoding="utf-8",
    )

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="GATE-",
        max_findings=2,
        force_goal_ids=["G-BLOCKED", "G-DONE"],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert [record.finding.goal_id for record in records] == ["G-BLOCKED"]
    assert records[0].finding.status == "blocked"
    assert records[0].finding.is_schedulable is False
    assert records[0].finding.review_only is True
    taskboard = todo_path.read_text(encoding="utf-8")
    assert "- Status: blocked" in taskboard
    assert "- Is schedulable: false" in taskboard
    assert "- Review only: true" in taskboard

    index_path = bundle_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    bundle = index["bundles"]["objective/gate/review"]
    indexed_task = bundle["tasks"][0]
    assert bundle["is_schedulable"] is False
    assert bundle["review_only"] is True
    assert indexed_task["status"] == "blocked"
    assert indexed_task["is_schedulable"] is False
    assert indexed_task["review_only"] is True

    payload = build_bundle_task_payloads(index_path)[0]
    assert payload["is_schedulable"] is False
    assert payload["review_only"] is True
    assert payload["claimable"] is False
    assert payload["ready_member_task_ids"] == []
    assert payload["execution_slice_task_ids"] == []
    assert payload["tasks"][0]["is_schedulable"] is False
    assert "profile_g" not in payload
    graph = materialize_task_dependency_dag(payload["tasks"], now=0)
    assert graph.schedule[0].claimable is False

    class RecordingQueue:
        def __init__(self):
            self.submissions = []

        def submit(self, **kwargs):
            self.submissions.append(kwargs)
            return "should-not-submit"

    queue = RecordingQueue()
    assert submit_bundle_tasks(index_path, queue=queue) == []
    assert queue.submissions == []
    assert plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=repo,
        state_root=repo / "state",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        task_prefix="GATE-",
        implement=True,
    ) == []

    assert scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        force_goal_ids=["G-DONE"],
    ) == []


def test_forced_blocked_goal_keeps_its_natural_objective_heap_position(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    objective_path = repo / "objective-heap.md"
    objective_path.write_text(
        """# Objective Heap

## G-ACTIVE Earlier active work

- Status: active
- Parent:
- Fib priority: 1
- Priority: P2
- Track: ordering
- Goal: Preserve canonical heap order.
- Evidence: active evidence
- Outputs: data/active.json
- Validation: test -f data/active.json

## G-BLOCKED Later review work

- Status: blocked
- Parent:
- Fib priority: 5
- Priority: P0
- Track: ordering
- Goal: Project this record at its natural heap position.
- Evidence: blocked review evidence
- Outputs: data/blocked.json
- Validation: test -f data/blocked.json
""",
        encoding="utf-8",
    )

    findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=2,
        force_goal_ids=["G-BLOCKED"],
        surplus_findings_per_goal=1,
    )

    assert [finding.goal_id for finding in findings] == [
        "G-ACTIVE",
        "G-BLOCKED",
    ]
    assert [finding.objective_heap_index for finding in findings] == [0, 1]
    assert findings[0].is_schedulable is True
    assert findings[1].is_schedulable is False


def test_objective_daemon_force_goal_id_projects_37_of_40_terminal_statuses_safely(
    tmp_path,
):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")

    objective_path = repo / "objective-heap.md"
    todo_path = repo / "objective.todo.md"
    discovery_dir = repo / "state" / "discovery"
    bundle_dir = repo / "state" / "bundles"
    graph_path = repo / "state" / "objective_graph.json"
    queue_path = repo / "state" / "queue.duckdb"
    sections = ["# Objective Heap"]
    for index in range(1, 38):
        goal_id = f"G{index:03d}"
        sections.append(
            f"""## {goal_id} Verified goal {index}

- Status: verified_complete
- Parent:
- Fib priority: 1
- Priority: P1
- Track: gate-review
- Bundle: objective/gate/{goal_id.lower()}
- Goal: Preserve verified terminal evidence.
- Evidence: verified completion receipt {index}
- Outputs: receipts/{goal_id.lower()}.json
- Validation: true
"""
        )
    for goal_id, fib_priority in (("G038", 8), ("G039", 2), ("G040", 5)):
        sections.append(
            f"""## {goal_id} Blocked gate review

- Status: blocked
- Parent:
- Fib priority: {fib_priority}
- Priority: P1
- Track: gate-review
- Bundle: objective/gate/{goal_id.lower()}
- Goal: Materialize terminal review evidence without execution.
- Evidence: blocked gate review {goal_id}
- Outputs: reviews/{goal_id.lower()}.json
- Validation: true
"""
        )
    objective_path.write_text("\n".join(sections), encoding="utf-8")

    args = build_objective_daemon_arg_parser().parse_args(
        [
            "--repo-root",
            str(repo),
            "--objective-path",
            str(objective_path),
            "--todo-path",
            str(todo_path),
            "--discovery-dir",
            str(discovery_dir),
            "--bundle-dir",
            str(bundle_dir),
            "--graph-path",
            str(graph_path),
            "--task-prefix",
            "GATE-",
            "--max-findings",
            "4",
            "--surplus-findings-per-goal",
            "1",
            "--force-goal-id",
            "G038",
            "--force-goal-id",
            "G001",
            "--force-goal-id",
            "G040",
            "--force-goal-id",
            "G039",
            "--no-reconcile-goal-completion",
            "--no-persist-ast-dataset",
            "--no-todo-vector-index",
            "--no-generate-bounded-work",
            "--submit-bundles",
            "--queue-path",
            str(queue_path),
        ]
    )

    result = run_objective_daemon(args)

    assert result["objective_goal_count"] == 40
    assert result["objective_completed_goal_count"] == 37
    assert result["objective_active_goal_count"] == 0
    assert result["generated_count"] == 3
    assert result["submitted_bundle_task_ids"] == []
    index_path = bundle_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed_tasks = sorted(
        (
            task
            for bundle in index["bundles"].values()
            for task in bundle["tasks"]
        ),
        key=lambda task: task["task_id"],
    )
    assert [task["goal_id"] for task in indexed_tasks] == [
        "G039",
        "G040",
        "G038",
    ]
    assert [task["objective_heap_index"] for task in indexed_tasks] == [0, 1, 2]
    assert {task["status"] for task in indexed_tasks} == {"blocked"}
    assert all(task["is_schedulable"] is False for task in indexed_tasks)
    assert all(task["review_only"] is True for task in indexed_tasks)
    assert plan_bundle_lanes(
        bundle_index_path=index_path,
        repo_root=repo,
        state_root=repo / "state" / "lanes",
        worktree_root=repo / "worktrees",
        log_dir=repo / "logs",
        task_prefix="GATE-",
        implement=True,
    ) == []


def test_goal_packet_aggregate_does_not_mix_active_and_review_only_scope():
    active = ObjectiveFinding(
        fingerprint="active-finding",
        goal_id="G-ACTIVE",
        title="Active work",
        summary="Active work",
        priority="P1",
        track="gate",
        missing_evidence=["active evidence"],
        present_evidence={},
        evidence_methods=[],
        objective_path="objective.md",
        outputs=["active.json"],
        validation="test -f active.json",
        goal_packet_key="goal_packet/gate/shared",
        goal_packet_goal_ids=["G-ACTIVE", "G-BLOCKED"],
    )
    review = replace(
        active,
        fingerprint="review-finding",
        goal_id="G-BLOCKED",
        title="Blocked review",
        summary="Blocked review",
        missing_evidence=["review evidence"],
        outputs=["review.json"],
        status="blocked",
        is_schedulable=False,
        review_only=True,
    )

    findings = add_goal_packet_aggregate_findings(
        [active, review],
        max_findings=3,
    )

    assert findings == [active, review]
    assert not any(
        finding.candidate_kind == "goal_packet_aggregate"
        for finding in findings
    )


def test_generate_objective_todos_omits_evidenced_packet_internal_dependencies(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G101 Packet anchor

- Status: active
- Parent:
- Evidence: anchor.json

## VAIOS-G102 Packet member

- Status: active
- Parent:
- Evidence: member.json
""",
        encoding="utf-8",
    )
    finding = ObjectiveFinding(
        fingerprint="packet-aggregate",
        goal_id="VAIOS-G101",
        title="Packet aggregate",
        summary="Implement packet aggregate",
        priority="P1",
        track="runtime",
        missing_evidence=["anchor.json", "member.json"],
        present_evidence={},
        evidence_methods=[],
        objective_path=str(objective_path),
        outputs=["src/runtime_router.py"],
        validation="true",
        dependencies=[
            "ACCEL-001",
            "VAIOS-G101",
            "VAIOS-G102",
        ],
        candidate_kind="goal_packet_aggregate",
        goal_packet_key="goal_packet/runtime/shared",
        goal_packet_role="packet_aggregate",
        goal_packet_goal_ids=["VAIOS-G101", "VAIOS-G102"],
        completion_goal_bindings={
            "VAIOS-G101": ["anchor.json"],
            "VAIOS-G102": ["member.json"],
        },
        semantic_identity="objective-evidence-packet/v1/packet",
        dedupe_key="objective-evidence-packet/v1/packet",
    )

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=repo / "discovery",
        bundle_dir=repo / "bundles",
        task_prefix="ACCEL-",
        precomputed_findings=[finding],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert len(records) == 1
    assert records[0].depends_on == ("ACCEL-001",)
    generated_block = records[0].task_block
    assert "- Depends on: ACCEL-001" in generated_block
    assert "Depends on: ACCEL-001, VAIOS-G101" not in generated_block


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


def test_goal_packets_respect_explicit_bundles_and_preserve_implicit_grouping():
    def finding(goal_id: str, bundle_key: str, *, explicit: bool) -> ObjectiveFinding:
        return ObjectiveFinding(
            fingerprint=f"fingerprint-{goal_id}",
            goal_id=goal_id,
            title=f"{goal_id} gap",
            summary=f"{goal_id} summary",
            priority="P1",
            track="runtime",
            missing_evidence=[f"missing-{goal_id}"],
            present_evidence={},
            evidence_methods=["exact"],
            objective_path="objective-heap.md",
            outputs=["src/bridge.py"],
            validation="true",
            parent_goal_ids=["VAIOS-G100"],
            bundle_key=bundle_key,
            parallel_lane=bundle_key,
            bundle_explicit=explicit,
            work_item_count=1,
        )

    explicit_findings = [
        finding("VAIOS-G101", "objective/runtime/shard-a", explicit=True),
        finding("VAIOS-G102", "objective/runtime/shard-a", explicit=True),
        finding("VAIOS-G103", "objective/runtime/shard-b", explicit=True),
        finding("VAIOS-G104", "objective/runtime/shard-b", explicit=True),
    ]
    packeted = assign_goal_subgoal_packets(explicit_findings)

    packets_by_bundle = {
        bundle_key: {
            item.goal_packet_key
            for item in packeted
            if item.bundle_key == bundle_key
        }
        for bundle_key in ("objective/runtime/shard-a", "objective/runtime/shard-b")
    }
    assert all(len(packet_keys) == 1 for packet_keys in packets_by_bundle.values())
    assert packets_by_bundle["objective/runtime/shard-a"] != packets_by_bundle["objective/runtime/shard-b"]
    assert {
        tuple(item.goal_packet_goal_ids)
        for item in packeted
        if item.bundle_key == "objective/runtime/shard-a"
    } == {("VAIOS-G101", "VAIOS-G102")}
    assert {
        tuple(item.goal_packet_goal_ids)
        for item in packeted
        if item.bundle_key == "objective/runtime/shard-b"
    } == {("VAIOS-G103", "VAIOS-G104")}

    expanded = add_goal_packet_aggregate_findings(packeted, max_findings=6)
    aggregates = [item for item in expanded if item.candidate_kind == "goal_packet_aggregate"]
    assert {
        (item.bundle_key, tuple(item.goal_packet_goal_ids))
        for item in aggregates
    } == {
        ("objective/runtime/shard-a", ("VAIOS-G101", "VAIOS-G102")),
        ("objective/runtime/shard-b", ("VAIOS-G103", "VAIOS-G104")),
    }

    implicit = assign_goal_subgoal_packets(
        [
            finding("VAIOS-G201", "objective/runtime/semantic-a", explicit=False),
            finding("VAIOS-G202", "objective/runtime/semantic-b", explicit=False),
        ]
    )
    assert implicit[0].goal_packet_key
    assert implicit[0].goal_packet_key == implicit[1].goal_packet_key


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


def test_objective_graph_generates_forced_launch_validation_gate_when_evidence_present(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective-heap.md"
    readiness = repo / "docs" / "launch" / "phone_desktop_glasses_readiness.md"
    readiness.parent.mkdir(parents=True)
    readiness.write_text(
        "launch_readiness_receipt_v1 covers phone desktop Swissknife Meta glasses readiness.\n",
        encoding="utf-8",
    )
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
- Evidence: docs/launch/phone_desktop_glasses_readiness.md
- Outputs: docs/launch/phone_desktop_glasses_readiness.md, tests
- Validation: test -f docs/launch/phone_desktop_glasses_readiness.md
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective-heap.md", "docs/launch/phone_desktop_glasses_readiness.md")
    _git(repo, "commit", "-m", "seed launch objective")

    unforced = scan_objective_gaps(repo, objective_path=objective_path, max_findings=1)
    forced = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        force_goal_ids=["VAIOS-G697"],
    )
    assert len(forced) == 1
    suppressed = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        force_goal_ids=["VAIOS-G697"],
        seen_fingerprints=[forced[0].fingerprint],
    )

    assert unforced == []
    assert forced[0].candidate_kind == "validation_gate"
    assert forced[0].missing_evidence == ["launch Playwright validation gate"]
    assert forced[0].work_scope == "launch_validation_gate"
    assert "npm --prefix swissknife run test:e2e:meta-glasses" in forced[0].validation
    assert [finding.goal_id for finding in suppressed] == ["VAIOS-G697"]


def test_objective_graph_generates_forced_interoperability_validation_repair_when_evidence_present(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "checkout", "-b", "main")
    _git(repo, "config", "user.name", "Test User")
    _git(repo, "config", "user.email", "test@example.invalid")
    objective_path = repo / "objective-heap.md"
    receipt = repo / "docs" / "integration" / "swissknife_mobile.md"
    receipt.parent.mkdir(parents=True)
    receipt.write_text("swissknife mobile interoperability adapter contract test receipt\n", encoding="utf-8")
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G700 Interoperate swissknife with mobile

- Status: active
- Parent:
- Fib priority: 1
- Track: interoperability
- Priority: P1
- Bundle: objective/interoperability/swissknife-mobile
- Goal: Prove swissknife and mobile can be used together.
- Evidence: docs/integration/swissknife_mobile.md
- Outputs: docs/integration/swissknife_mobile.md, tests
- Validation: python -m pytest tests/integration -q
""",
        encoding="utf-8",
    )
    _git(repo, "add", "objective-heap.md", "docs/integration/swissknife_mobile.md")
    _git(repo, "commit", "-m", "seed interoperability objective")

    unforced = scan_objective_gaps(repo, objective_path=objective_path, max_findings=1)
    forced = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        force_goal_ids=["VAIOS-G700"],
    )

    assert unforced == []
    assert len(forced) == 1
    assert forced[0].candidate_kind == "validation_gate"
    assert forced[0].missing_evidence == ["objective validation repair"]
    assert forced[0].work_scope == "objective_validation_repair"
    assert forced[0].validation == "python -m pytest tests/integration -q"


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
    assert records[0].board_namespace == "todo.md"
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## ACCEL-002 Close objective gap" in todo_text
    assert "- Board namespace: todo.md" in todo_text
    assert "- Bundle: objective/ops/root" in todo_text
    generated_block = todo_text.split("## ACCEL-002 ", 1)[1]
    outputs_line = next(
        line for line in generated_block.splitlines() if line.startswith("- Outputs:")
    )
    assert "objective-heap.md" not in outputs_line
    assert "data/agent_supervisor/discovery" not in outputs_line
    assert "- Evidence inputs: data/agent_supervisor/discovery" in generated_block
    assert "- Discovery evidence:" in generated_block

    shard = bundle_dir / "objective-ops-root.todo.md"
    assert shard.exists()
    shard_text = shard.read_text(encoding="utf-8")
    assert "## ACCEL-002 Close objective gap" in shard_text
    assert "- Board namespace: todo.md" in shard_text
    index_path = bundle_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed_task = index["bundles"]["objective/ops/root"]["tasks"][0]
    assert indexed_task["task_id"] == "ACCEL-002"
    assert indexed_task["board_namespace"] == "todo.md"
    assert index["task_conflict_graph"]["surfaces"]
    assert index["task_planning_graph"]["planning_decisions"] == []
    dataset_manifest = bundle_dir.parent / "objective_datasets" / "accel-objective-ast.manifest.json"
    assert dataset_manifest.exists()
    dataset_payload = json.loads(dataset_manifest.read_text(encoding="utf-8"))
    assert dataset_payload["row_count"] >= 2
    assert Path(dataset_payload["jsonl_path"]).exists()

    payloads = build_bundle_task_payloads(index_path)
    assert payloads[0]["bundle_key"] == "objective/ops/root"
    assert payloads[0]["todo_path"].endswith("objective-ops-root.todo.md")
    assert payloads[0]["planning_evidence_ref"]["conflict_edge_table"] == "conflict_edges"
    assert "task_conflict_graph" not in payloads[0]

    submitted: list[dict[str, object]] = []

    class FakeQueue:
        def submit(self, **kwargs):
            submitted.append(kwargs)
            return "queued-1"

    task_ids = submit_bundle_tasks(index_path, queue=FakeQueue())

    assert task_ids == ["queued-1"]
    assert submitted[0]["task_type"] == "codex.todo_bundle"
    assert submitted[0]["payload"]["bundle_key"] == "objective/ops/root"


def test_generate_objective_todos_projects_dscon_path_evidence_as_typed_outputs(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = (
        repo
        / "data"
        / "datasets_contract_analysis"
        / "agent_supervisor"
        / "discovery"
    )
    bundle_dir = (
        repo
        / "data"
        / "datasets_contract_analysis"
        / "agent_supervisor"
        / "objective_bundles"
    )
    manifests = [
        "data/datasets_contract_analysis/manifests/repository-root.json",
        "data/datasets_contract_analysis/manifests/coverage.json",
    ]
    outputs = [
        "ipfs_datasets_py/processors/datasets/repository.py",
        "ipfs_datasets_py/processors/datasets/coverage.py",
        "test/datasets/test_repository_coverage.py",
    ]
    finding = ObjectiveFinding(
        fingerprint="dscon-g020-repository-inventory",
        goal_id="DSCON-G020",
        title="Inventory repository coverage",
        summary="Close objective gap: Inventory repository coverage",
        priority="P0",
        track="datasets-contract-analysis",
        missing_evidence=[
            *manifests,
            "operator approval",
            "123456789012345678901234567890",
            "https://example.invalid/receipt.json",
            "../outside.json",
            "data/*/forged.json",
            "objective-heap.md",
            (
                "data/datasets_contract_analysis/agent_supervisor/"
                "discovery/forged.json"
            ),
        ],
        present_evidence={},
        evidence_methods=[],
        objective_path="objective-heap.md",
        outputs=outputs,
        validation="python -m pytest -q test/datasets/test_repository_coverage.py",
        evidence_subset=[
            *manifests,
            "operator approval",
            "123456789012345678901234567890",
            "https://example.invalid/receipt.json",
            "../outside.json",
            "data/*/forged.json",
            "objective-heap.md",
            (
                "data/datasets_contract_analysis/agent_supervisor/"
                "discovery/forged.json"
            ),
        ],
        bundle_key="objective/datasets/repository-inventory",
    )

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="DSCON-",
        precomputed_findings=[finding],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
        discovery_output_path=(
            "data/datasets_contract_analysis/agent_supervisor/discovery"
        ),
    )

    assert len(records) == 1
    assert records[0].evidence_outputs == tuple(manifests)
    generated_block = todo_path.read_text(encoding="utf-8").split(
        "## DSCON-001 ",
        1,
    )[1]
    assert f"- Outputs: {', '.join(outputs)}" in generated_block
    assert f"- Evidence outputs: {', '.join(manifests)}" in generated_block
    index = json.loads(
        (bundle_dir / "index.json").read_text(encoding="utf-8")
    )
    task = index["bundles"][
        "objective/datasets/repository-inventory"
    ]["tasks"][0]
    assert task["evidence_outputs"] == sorted(manifests)
    assert task["outputs"] == [*outputs, *sorted(manifests)]
    assert set(task["files"]) == {*outputs, *manifests}
    assert "objective-heap.md" not in task["files"]
    assert not any(
        "agent_supervisor/discovery" in path for path in task["files"]
    )

    assert (
        objective_finding_task_identity("DSCON-001", finding).canonical_task_cid
        != objective_finding_task_identity(
            "DSCON-001",
            finding,
            evidence_outputs=[],
        ).canonical_task_cid
    )


def _seed_legacy_dscon_evidence_output_card(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = (
        repo
        / "data"
        / "datasets_contract_analysis"
        / "agent_supervisor"
        / "discovery"
    )
    bundle_dir = (
        repo
        / "data"
        / "datasets_contract_analysis"
        / "agent_supervisor"
        / "bundles"
    )
    manifests = [
        "data/datasets_contract_analysis/manifests/repository-root.json",
        "data/datasets_contract_analysis/manifests/coverage.json",
    ]
    finding = ObjectiveFinding(
        fingerprint="4f91244e4ecbba7c3c22c1c2a2f1da27e59c41a0",
        goal_id="DSCON-G020",
        title="Build recursive tracked-object and coverage manifests",
        summary=(
            "Implement datasets symbolic contract objective: "
            "Build recursive tracked-object and coverage manifests"
        ),
        priority="P0",
        track="bootstrap",
        missing_evidence=manifests,
        present_evidence={},
        evidence_methods=[],
        objective_path="objective-heap.md",
        outputs=[
            "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/repository.py",
            "ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/coverage.py",
            (
                "ipfs_datasets_py/tests/unit/logic/software_contracts/"
                "test_repository_manifest.py"
            ),
        ],
        validation=(
            "python -m pytest -q "
            "ipfs_datasets_py/tests/unit/logic/software_contracts/"
            "test_repository_manifest.py"
        ),
        evidence_subset=manifests,
        bundle_key="datasets-contract/bootstrap",
        parallel_lane="bootstrap-coverage",
        dedupe_key=(
            "objective-evidence-obligation/v1/"
            "f56244a68715f27b32683260e3b53b8ff915bc8dcefee0b3c5b91b6b8f2dbd96"
        ),
        semantic_identity=(
            "objective-evidence-obligation/v1/"
            "f56244a68715f27b32683260e3b53b8ff915bc8dcefee0b3c5b91b6b8f2dbd96"
        ),
    )
    created = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="DSCON-",
        precomputed_findings=[finding],
        persist_ast_dataset=False,
        write_todo_vector_index=True,
        discovery_output_path=(
            "data/datasets_contract_analysis/agent_supervisor/discovery"
        ),
    )
    assert len(created) == 1
    task_id = created[0].task_id
    current_identity = objective_finding_task_identity(task_id, finding)
    legacy_identity = objective_finding_task_identity(
        task_id,
        finding,
        evidence_outputs=[],
    )

    def downgrade(markdown):
        downgraded = markdown.replace(
            f"- Evidence outputs: {', '.join(manifests)}\n",
            "",
            1,
        )
        downgraded = downgraded.replace(
            current_identity.canonical_task_key,
            legacy_identity.canonical_task_key,
            1,
        )
        downgraded = downgraded.replace(
            current_identity.canonical_task_cid,
            legacy_identity.canonical_task_cid,
            1,
        )
        obligation_line = f"- Evidence obligation key: {finding.dedupe_key}"
        return downgraded.replace(
            obligation_line,
            (
                obligation_line
                + "\n- Projection history: preserve this operator note"
            ),
            1,
        )

    todo_path.write_text(
        downgrade(todo_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    shard_path = bundle_dir / "datasets-contract-bootstrap.todo.md"
    shard_path.write_text(
        downgrade(shard_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    discovery_text = created[0].discovery_path.read_text(encoding="utf-8")
    for line in (
        f"Evidence outputs: {', '.join(manifests)}\n",
        f"Canonical task key: {current_identity.canonical_task_key}\n",
        f"Canonical task CID: {current_identity.canonical_task_cid}\n",
    ):
        discovery_text = discovery_text.replace(line, "", 1)
    created[0].discovery_path.write_text(discovery_text, encoding="utf-8")
    return {
        "repo": repo,
        "objective_path": objective_path,
        "todo_path": todo_path,
        "discovery_dir": discovery_dir,
        "bundle_dir": bundle_dir,
        "shard_path": shard_path,
        "vector_path": bundle_dir / "todo_vector_index.json",
        "finding": finding,
        "manifests": manifests,
        "task_id": task_id,
        "legacy_identity": legacy_identity,
        "current_identity": current_identity,
        "discovery_path": created[0].discovery_path,
    }


def test_objective_refill_reprojects_legacy_dscon_card_and_artifacts_in_place(
    tmp_path,
):
    seeded = _seed_legacy_dscon_evidence_output_card(tmp_path)
    old_vector_bytes = seeded["vector_path"].read_bytes()
    old_vector = json.loads(
        seeded["vector_path"].read_text(encoding="utf-8")
    )
    old_vector_task = next(
        item
        for item in old_vector["records"]
        if item["task_id"] == seeded["task_id"]
    )

    refilled = generate_objective_todos(
        repo_root=seeded["repo"],
        objective_path=seeded["objective_path"],
        todo_path=seeded["todo_path"],
        discovery_dir=seeded["discovery_dir"],
        bundle_dir=seeded["bundle_dir"],
        task_prefix="DSCON-",
        precomputed_findings=[seeded["finding"]],
        persist_ast_dataset=False,
        write_todo_vector_index=True,
        discovery_output_path=(
            "data/datasets_contract_analysis/agent_supervisor/discovery"
        ),
    )

    # Obligation dedupe still reports zero newly generated tasks.  The exact
    # existing display ID is rotated under the generator's taskboard lock.
    assert refilled == []
    todo_text = seeded["todo_path"].read_text(encoding="utf-8")
    assert todo_text.count(f"## {seeded['task_id']} ") == 1
    assert "- Status: todo" in todo_text
    assert "- Projection history: preserve this operator note" in todo_text
    assert (
        f"- Evidence outputs: {', '.join(seeded['manifests'])}"
        in todo_text
    )
    assert seeded["current_identity"].canonical_task_key in todo_text
    assert seeded["current_identity"].canonical_task_cid in todo_text
    assert seeded["legacy_identity"].canonical_task_cid not in todo_text

    discovery_text = seeded["discovery_path"].read_text(encoding="utf-8")
    assert (
        f"Evidence outputs: {', '.join(seeded['manifests'])}"
        in discovery_text
    )
    assert (
        f"Canonical task CID: {seeded['current_identity'].canonical_task_cid}"
        in discovery_text
    )
    shard_text = seeded["shard_path"].read_text(encoding="utf-8")
    assert "- Projection history: preserve this operator note" in shard_text
    assert (
        f"- Evidence outputs: {', '.join(seeded['manifests'])}"
        in shard_text
    )

    bundle_index = json.loads(
        (seeded["bundle_dir"] / "index.json").read_text(encoding="utf-8")
    )
    indexed_task = bundle_index["bundles"][
        "datasets-contract/bootstrap"
    ]["tasks"][0]
    assert indexed_task["task_id"] == seeded["task_id"]
    assert (
        indexed_task["canonical_task_cid"]
        == seeded["current_identity"].canonical_task_cid
    )
    assert indexed_task["evidence_outputs"] == sorted(seeded["manifests"])

    new_vector = json.loads(
        seeded["vector_path"].read_text(encoding="utf-8")
    )
    new_vector_task = next(
        item
        for item in new_vector["records"]
        if item["task_id"] == seeded["task_id"]
    )
    assert seeded["vector_path"].read_bytes() != old_vector_bytes
    assert new_vector_task["task_id"] == old_vector_task["task_id"]

    generated_artifacts = {
        path: path.read_bytes()
        for path in (
            seeded["todo_path"],
            seeded["discovery_path"],
            seeded["shard_path"],
            seeded["bundle_dir"] / "index.json",
            seeded["vector_path"],
        )
    }
    repeated = generate_objective_todos(
        repo_root=seeded["repo"],
        objective_path=seeded["objective_path"],
        todo_path=seeded["todo_path"],
        discovery_dir=seeded["discovery_dir"],
        bundle_dir=seeded["bundle_dir"],
        task_prefix="DSCON-",
        precomputed_findings=[seeded["finding"]],
        persist_ast_dataset=False,
        write_todo_vector_index=True,
        discovery_output_path=(
            "data/datasets_contract_analysis/agent_supervisor/discovery"
        ),
    )
    assert repeated == []
    assert {
        path: path.read_bytes() for path in generated_artifacts
    } == generated_artifacts


def test_normal_refill_sweeps_seen_legacy_card_once(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = (
        repo
        / "data"
        / "datasets_contract_analysis"
        / "agent_supervisor"
        / "discovery"
    )
    bundle_dir = (
        repo
        / "data"
        / "datasets_contract_analysis"
        / "agent_supervisor"
        / "bundles"
    )
    manifests = [
        "data/datasets_contract_analysis/manifests/repository-root.json",
        "data/datasets_contract_analysis/manifests/coverage.json",
    ]
    objective_path.write_text(
        f"""# Objective Heap

## DSCON-G020 Build recursive tracked-object and coverage manifests

- Status: active
- Parent:
- Fib priority: 1
- Track: bootstrap
- Priority: P0
- Bundle: datasets-contract/bootstrap
- Goal: Build recursive tracked-object and coverage manifests.
- Evidence: {", ".join(manifests)}
- Outputs: ipfs_datasets_py/ipfs_datasets_py/logic/software_contracts/repository.py, ipfs_datasets_py/tests/unit/logic/software_contracts/test_repository_manifest.py
- Validation: true
- Gap task: Generate deterministic repository coverage manifests.
""",
        encoding="utf-8",
    )
    created = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="DSCON-",
        max_findings=4,
        persist_ast_dataset=False,
        write_todo_vector_index=False,
        discovery_output_path=(
            "data/datasets_contract_analysis/agent_supervisor/discovery"
        ),
    )
    assert len(created) == 1
    record = created[0]
    finding = record.finding
    current_identity = objective_finding_task_identity(
        record.task_id,
        finding,
    )
    legacy_identity = objective_finding_task_identity(
        record.task_id,
        finding,
        evidence_outputs=[],
    )

    def downgrade(markdown):
        value = markdown.replace(
            f"- Evidence outputs: {', '.join(manifests)}\n",
            "",
            1,
        )
        value = value.replace(
            current_identity.canonical_task_key,
            legacy_identity.canonical_task_key,
            1,
        )
        return value.replace(
            current_identity.canonical_task_cid,
            legacy_identity.canonical_task_cid,
            1,
        )

    todo_path.write_text(
        downgrade(todo_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    shard_path = bundle_dir / "datasets-contract-bootstrap.todo.md"
    shard_path.write_text(
        downgrade(shard_path.read_text(encoding="utf-8")),
        encoding="utf-8",
    )
    discovery_text = record.discovery_path.read_text(encoding="utf-8")
    for line in (
        f"Evidence outputs: {', '.join(manifests)}\n",
        f"Canonical task key: {current_identity.canonical_task_key}\n",
        f"Canonical task CID: {current_identity.canonical_task_cid}\n",
    ):
        discovery_text = discovery_text.replace(line, "", 1)
    record.discovery_path.write_text(discovery_text, encoding="utf-8")

    # This is the real daemon path: discovery has already recorded the
    # fingerprint, so the ordinary new-task scan returns no candidate.
    assert scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=4,
        seen_fingerprints=[finding.fingerprint],
    ) == []
    refilled = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="DSCON-",
        max_findings=4,
        seen_fingerprints=[finding.fingerprint],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
        discovery_output_path=(
            "data/datasets_contract_analysis/agent_supervisor/discovery"
        ),
    )

    assert refilled == []
    migrated = todo_path.read_text(encoding="utf-8")
    assert migrated.count(f"## {record.task_id} ") == 1
    assert f"- Evidence outputs: {', '.join(manifests)}" in migrated
    assert current_identity.canonical_task_cid in migrated
    assert legacy_identity.canonical_task_cid not in migrated

    generated_artifacts = {
        path: path.read_bytes()
        for path in (
            todo_path,
            record.discovery_path,
            shard_path,
            bundle_dir / "index.json",
        )
    }
    repeated = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="DSCON-",
        max_findings=4,
        seen_fingerprints=[finding.fingerprint],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
        discovery_output_path=(
            "data/datasets_contract_analysis/agent_supervisor/discovery"
        ),
    )
    assert repeated == []
    assert {
        path: path.read_bytes() for path in generated_artifacts
    } == generated_artifacts


def test_retained_reprojection_fingerprint_cannot_be_starved_by_surplus(
    tmp_path,
):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    objective_path.write_text(
        """# Objective Heap

## DSCON-G001 Early wide goal

- Status: active
- Parent:
- Fib priority: 1
- Track: bootstrap
- Priority: P0
- Goal: Fill the ordinary candidate pool first.
- Evidence: data/manifests/early-a.json, data/manifests/early-b.json, data/manifests/early-c.json
- Outputs: src/early.py
- Validation: true

## DSCON-G999 Late retained goal

- Status: active
- Parent:
- Fib priority: 999
- Track: drift
- Priority: P3
- Goal: Retain this exact legacy migration candidate.
- Evidence: data/manifests/late.json
- Outputs: src/late.py
- Validation: true
""",
        encoding="utf-8",
    )
    all_findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=16,
        surplus_findings_per_goal=3,
        surplus_min_terms_per_todo=1,
    )
    retained = next(
        finding
        for finding in all_findings
        if finding.goal_id == "DSCON-G999"
    )

    constrained = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
        seen_fingerprints=[retained.fingerprint],
        retain_fingerprints=[retained.fingerprint],
        surplus_findings_per_goal=3,
        surplus_min_terms_per_todo=1,
    )

    assert constrained[0].fingerprint == retained.fingerprint
    assert constrained[0].goal_id == "DSCON-G999"
    assert len(
        [
            finding
            for finding in constrained
            if finding.fingerprint == retained.fingerprint
        ]
    ) == 1


def test_objective_refill_repairs_artifacts_after_interrupted_card_rotation(
    tmp_path,
    monkeypatch,
):
    seeded = _seed_legacy_dscon_evidence_output_card(tmp_path)
    write_bundles = objective_graph_module.write_bundle_shards

    def interrupt_after_board(**_kwargs):
        raise RuntimeError("simulated interruption after locked board rotation")

    monkeypatch.setattr(
        objective_graph_module,
        "write_bundle_shards",
        interrupt_after_board,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        generate_objective_todos(
            repo_root=seeded["repo"],
            objective_path=seeded["objective_path"],
            todo_path=seeded["todo_path"],
            discovery_dir=seeded["discovery_dir"],
            bundle_dir=seeded["bundle_dir"],
            task_prefix="DSCON-",
            precomputed_findings=[seeded["finding"]],
            persist_ast_dataset=False,
            write_todo_vector_index=True,
            discovery_output_path=(
                "data/datasets_contract_analysis/agent_supervisor/discovery"
            ),
        )
    assert seeded["current_identity"].canonical_task_cid in (
        seeded["todo_path"].read_text(encoding="utf-8")
    )
    assert "Evidence outputs:" not in (
        seeded["discovery_path"].read_text(encoding="utf-8")
    )

    monkeypatch.setattr(
        objective_graph_module,
        "write_bundle_shards",
        write_bundles,
    )
    recovered = generate_objective_todos(
        repo_root=seeded["repo"],
        objective_path=seeded["objective_path"],
        todo_path=seeded["todo_path"],
        discovery_dir=seeded["discovery_dir"],
        bundle_dir=seeded["bundle_dir"],
        task_prefix="DSCON-",
        precomputed_findings=[seeded["finding"]],
        persist_ast_dataset=False,
        write_todo_vector_index=True,
        discovery_output_path=(
            "data/datasets_contract_analysis/agent_supervisor/discovery"
        ),
    )
    assert recovered == []
    assert (
        f"Canonical task CID: "
        f"{seeded['current_identity'].canonical_task_cid}"
        in seeded["discovery_path"].read_text(encoding="utf-8")
    )
    assert (
        seeded["current_identity"].canonical_task_cid
        in seeded["shard_path"].read_text(encoding="utf-8")
    )


@pytest.mark.parametrize("status", ["completed", "in_progress", "running"])
def test_objective_refill_does_not_reproject_terminal_or_active_cards(
    tmp_path,
    status,
):
    seeded = _seed_legacy_dscon_evidence_output_card(tmp_path)
    board = seeded["todo_path"].read_text(encoding="utf-8").replace(
        "- Status: todo",
        f"- Status: {status}",
        1,
    )
    seeded["todo_path"].write_text(board, encoding="utf-8")
    before = {
        path: path.read_bytes()
        for path in (
            seeded["todo_path"],
            seeded["discovery_path"],
            seeded["shard_path"],
            seeded["bundle_dir"] / "index.json",
            seeded["vector_path"],
        )
    }

    records = generate_objective_todos(
        repo_root=seeded["repo"],
        objective_path=seeded["objective_path"],
        todo_path=seeded["todo_path"],
        discovery_dir=seeded["discovery_dir"],
        bundle_dir=seeded["bundle_dir"],
        task_prefix="DSCON-",
        precomputed_findings=[seeded["finding"]],
        persist_ast_dataset=False,
        write_todo_vector_index=True,
        discovery_output_path=(
            "data/datasets_contract_analysis/agent_supervisor/discovery"
        ),
    )

    assert records == []
    assert {path: path.read_bytes() for path in before} == before


def test_objective_refill_fails_closed_for_ambiguous_legacy_card_binding(
    tmp_path,
):
    seeded = _seed_legacy_dscon_evidence_output_card(tmp_path)
    board = seeded["todo_path"].read_text(encoding="utf-8")
    original_block = board.split(f"## {seeded['task_id']} ", 1)[1]
    ambiguous_block = (
        f"## DSCON-099 {original_block}"
    )
    seeded["todo_path"].write_text(
        board.rstrip() + "\n\n" + ambiguous_block,
        encoding="utf-8",
    )
    before = seeded["todo_path"].read_bytes()

    records = generate_objective_todos(
        repo_root=seeded["repo"],
        objective_path=seeded["objective_path"],
        todo_path=seeded["todo_path"],
        discovery_dir=seeded["discovery_dir"],
        bundle_dir=seeded["bundle_dir"],
        task_prefix="DSCON-",
        precomputed_findings=[seeded["finding"]],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
        discovery_output_path=(
            "data/datasets_contract_analysis/agent_supervisor/discovery"
        ),
    )

    assert records == []
    assert seeded["todo_path"].read_bytes() == before


@pytest.mark.parametrize(
    "unsafe_requirement",
    [
        "operator approval",
        "123456789012345678901234567890",
        "https://example.invalid/proof.json",
        "/absolute/proof.json",
        r"C:\proof.json",
        "../proof.json",
        "./proof.json",
        "data/proof.json/",
        "data//proof.json",
        "data/*/proof.json",
        "data/\nproof.json",
        ".git/config.json",
        "operator/approval",
    ],
)
def test_objective_evidence_output_projection_rejects_noncanonical_authority(
    unsafe_requirement,
):
    finding = ObjectiveFinding(
        fingerprint="unsafe-path-evidence",
        goal_id="DSCON-G020",
        title="Reject unsafe path evidence",
        summary="Reject unsafe path evidence",
        priority="P0",
        track="datasets-contract-analysis",
        missing_evidence=[
            "data/manifests/coverage.json",
            unsafe_requirement,
        ],
        present_evidence={},
        evidence_methods=[],
        objective_path="docs/objectives.md",
        outputs=["src/coverage.py"],
        validation="true",
        evidence_subset=[
            "data/manifests/coverage.json",
            unsafe_requirement,
        ],
    )

    assert objective_finding_evidence_output_paths(finding) == [
        "data/manifests/coverage.json"
    ]


def test_generate_objective_todos_inherits_explicit_board_namespace(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    todo_path.write_text(
        todo_path.read_text(encoding="utf-8").replace(
            "- Validation: true",
            "- Validation: true\n"
            "- Board namespace: ipfs-kit-vfs-symbolic-assurance-v1",
        ),
        encoding="utf-8",
    )
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

    namespace = "ipfs-kit-vfs-symbolic-assurance-v1"
    assert records[0].board_namespace == namespace
    assert f"- Board namespace: {namespace}" in records[0].task_block
    index = json.loads((bundle_dir / "index.json").read_text(encoding="utf-8"))
    indexed_task = index["bundles"]["objective/ops/root"]["tasks"][0]
    assert indexed_task["board_namespace"] == namespace


def test_taskboard_namespace_rejects_conflicting_explicit_metadata(tmp_path):
    todo_path = tmp_path / "todo.md"
    todo_text = """# Todos

## AUTO-001 First

- Board namespace: first-v1

## AUTO-002 Second

- Board namespace: second-v1
"""

    with pytest.raises(ValueError, match="conflicting board namespaces"):
        taskboard_namespace_from_todo(todo_text, todo_path)


def test_generate_objective_todos_projects_goal_dependencies_to_task_ids(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    todo_path.write_text(
        todo_path.read_text(encoding="utf-8").rstrip()
        + """

- Goal id: VAIOS-G000
""",
        encoding="utf-8",
    )
    finding = ObjectiveFinding(
        fingerprint="dependent-goal-gap",
        goal_id="VAIOS-G001",
        title="Dependent objective",
        summary="Implement dependent objective",
        priority="P1",
        track="runtime",
        missing_evidence=["dependent proof"],
        present_evidence={},
        evidence_methods=[],
        objective_path=str(objective_path),
        outputs=["src"],
        validation="true",
        dependencies=["VAIOS-G000"],
    )

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="ACCEL-",
        precomputed_findings=[finding],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert len(records) == 1
    assert records[0].depends_on == ("ACCEL-001",)
    assert records[0].finding.dependencies == ["ACCEL-001"]
    generated = todo_path.read_text(encoding="utf-8")
    assert "- Depends on: ACCEL-001" in generated
    assert "- Depends on: VAIOS-G000" not in generated
    index = json.loads(
        (bundle_dir / "index.json").read_text(encoding="utf-8")
    )
    indexed_task = next(
        task
        for bundle in index["bundles"].values()
        for task in bundle["tasks"]
        if task["task_id"] == "ACCEL-002"
    )
    assert indexed_task["depends_on"] == ["ACCEL-001"]
    assert indexed_task["dependency_task_ids"] == ["ACCEL-001"]


def test_manual_review_finding_without_edit_targets_is_visible_but_not_executable(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    common = {
        "goal_id": "VAIOS-G010",
        "title": "Virtual AI operating system",
        "priority": "P1",
        "track": "ops",
        "present_evidence": {},
        "objective_path": "objective-heap.md",
        "validation": "git diff --check",
        "goal": "Keep objective evidence current.",
        "parent_goal_ids": [],
        "graph_depth": 1,
        "bundle_key": "objective/ops/review",
        "parallel_lane": "objective/ops/review",
        "bundle_strategy": "bounded_objective_generation",
        "candidate_kind": "generated_task",
        "surplus_group": "VAIOS-G010",
        "work_scope": "bounded_objective_generation",
    }
    manual_review = ObjectiveFinding(
        **common,
        fingerprint="manual-review-no-edit-target",
        summary="Review completion evidence",
        missing_evidence=["independent completion proof"],
        evidence_methods=[
            "bounded_objective_generation",
            "completion_gate_gap_manual_review",
        ],
        outputs=[],
        merge_key="objective-family/v1/manual-review",
        merge_family="VAIOS-G010",
        merge_role="completion_gate_gap_manual_review",
        predicted_files=[],
    )
    actionable = ObjectiveFinding(
        **common,
        fingerprint="actionable-typed-gap",
        summary="Align completion evidence",
        missing_evidence=["documentation validator proof"],
        evidence_methods=[
            "bounded_objective_generation",
            "completion_gate_gap",
        ],
        outputs=["docs/runtime_notes.md"],
        merge_key="objective-family/v1/actionable-gap",
        merge_family="VAIOS-G010",
        merge_role="completion_gate_gap",
        predicted_files=["docs/runtime_notes.md"],
    )

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="ACCEL-",
        precomputed_findings=[manual_review, actionable],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert [record.task_id for record in records] == ["ACCEL-002", "ACCEL-003"]
    todo_text = todo_path.read_text(encoding="utf-8")
    manual_block, actionable_block = todo_text.split("## ACCEL-002 ", 1)[1].split(
        "## ACCEL-003 ", 1
    )
    assert "- Status: blocked" in manual_block
    assert (
        "- Blocked reason: manual review required because no precise edit "
        "targets were authorized"
    ) in manual_block
    assert "authorize precise repository-relative edit targets" in manual_block
    assert "- Status: todo" in actionable_block
    assert "- Blocked reason:" not in actionable_block
    assert all(
        line == line.rstrip()
        for line in (manual_block + actionable_block).splitlines()
    )

    index_path = bundle_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    indexed = {
        task["task_id"]: task
        for task in index["bundles"]["objective/ops/review"]["tasks"]
    }
    assert indexed["ACCEL-002"]["status"] == "blocked"
    assert indexed["ACCEL-003"].get("status", "todo") == "todo"

    bundle = build_bundle_task_payloads(index_path)[0]
    assert bundle["blocked_member_task_ids"] == ["ACCEL-002"]
    assert bundle["ready_member_task_ids"] == ["ACCEL-003"]
    assert bundle["execution_slice_task_ids"] == ["ACCEL-003"]
    assert bundle["claimable"] is True

    from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
        TodoImplementationDaemon,
    )

    state_dir = repo / "data" / "implementation"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## ACCEL-",
    )
    state = daemon.run_once()
    persisted_state = json.loads(Path(state["state_path"]).read_text(encoding="utf-8"))
    assert state["blocked_count"] == 1
    assert state["ready_count"] == 1
    assert state["active_task_id"] == "ACCEL-003"
    assert persisted_state["blocked_task_ids"] == ["ACCEL-002"]
    assert persisted_state["ready_task_ids"] == ["ACCEL-003"]
    assert persisted_state["recommended_task_id"] == "ACCEL-003"


def test_generate_objective_todos_skips_existing_canonical_task(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    first = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="ACCEL-",
        max_findings=1,
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )
    original_todo = todo_path.read_text(encoding="utf-8")
    original_discoveries = sorted(discovery_dir.iterdir())

    repeated = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="ACCEL-",
        precomputed_findings=[first[0].finding],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert repeated == []
    assert todo_path.read_text(encoding="utf-8") == original_todo
    assert sorted(discovery_dir.iterdir()) == original_discoveries


def test_generate_objective_todos_recognizes_legacy_refinement_obligation(
    tmp_path,
):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    todo_path.write_text(
        todo_path.read_text(encoding="utf-8").rstrip()
        + """

## ACCEL-002 Legacy refined evidence task

- Status: todo
- Completion: manual
- Priority: P0
- Track: ops
- Depends on:
- Goal id: VAIOS-G999
- Graph parents: VAIOS-G000
- Missing evidence: missing_meta_glasses_contract
- Candidate kind: aggregate
- Canonical task key: task/v1/legacy
- Canonical task CID: baguqeera-legacy
- Acceptance: Close the inherited evidence obligation.
""",
        encoding="utf-8",
    )
    original_todo = todo_path.read_text(encoding="utf-8")
    finding = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
    )[0]

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="ACCEL-",
        precomputed_findings=[finding],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert records == []
    assert todo_path.read_text(encoding="utf-8") == original_todo
    assert not discovery_dir.exists()


def test_legacy_atomic_tasks_cover_reordered_aggregate_obligation(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G000 Aggregate outcome

- Status: active
- Track: ops
- Bundle: objective/ops/root
- Evidence: requirement_alpha, requirement_beta, requirement_gamma
- Outputs: src, tests
- Validation: true
""",
        encoding="utf-8",
    )
    task_blocks = []
    for index, requirement in enumerate(
        ("requirement_alpha", "requirement_beta", "requirement_gamma"),
        start=2,
    ):
        task_blocks.append(
            f"""## ACCEL-{index:03d} Legacy atomic task

- Status: todo
- Goal id: VAIOS-G{index:03d}
- Graph parents: VAIOS-G000
- Missing evidence: {requirement}
- Candidate kind: aggregate
- Acceptance: Produce one exact requirement.
"""
        )
    todo_path.write_text(
        "# Objective Todos\n\n" + "\n".join(task_blocks),
        encoding="utf-8",
    )
    original_todo = todo_path.read_text(encoding="utf-8")
    finding = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
    )[0]
    assert set(finding.missing_evidence) == {
        "requirement_alpha",
        "requirement_beta",
        "requirement_gamma",
    }

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="ACCEL-",
        precomputed_findings=[replace(
            finding,
            missing_evidence=list(reversed(finding.missing_evidence)),
        )],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert records == []
    assert todo_path.read_text(encoding="utf-8") == original_todo


def test_existing_packet_task_covers_each_sibling_goal_obligation(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G100 Runtime parent
- Status: completed

## VAIOS-G101 Scheduler child
- Status: active
- Parent: VAIOS-G100
- Evidence: scheduler_policy, scheduler_metrics

## VAIOS-G102 Fallback child
- Status: active
- Parent: VAIOS-G100
- Evidence: fallback_route, fallback_metrics
""",
        encoding="utf-8",
    )
    todo_path.write_text(
        """# Objective Todos

## ACCEL-001 Existing packet task

- Status: todo
- Goal id: VAIOS-G101
- Goal packet goals: VAIOS-G101, VAIOS-G102
- Completion goal bindings: {"VAIOS-G101":["scheduler_policy","scheduler_metrics"],"VAIOS-G102":["fallback_route","fallback_metrics"]}
- Graph parents: VAIOS-G100
- Missing evidence: scheduler_policy, scheduler_metrics, fallback_route, fallback_metrics
- Candidate kind: goal_packet_aggregate
- Acceptance: Close both sibling obligations in one packet.
""",
        encoding="utf-8",
    )
    original_todo = todo_path.read_text(encoding="utf-8")
    fallback_finding = ObjectiveFinding(
        fingerprint="fallback-gap",
        goal_id="VAIOS-G102",
        title="Fallback child",
        summary="Close fallback evidence",
        priority="P1",
        track="runtime",
        missing_evidence=["fallback_route", "fallback_metrics"],
        present_evidence={},
        evidence_methods=[],
        objective_path=str(objective_path),
        outputs=["src"],
        validation="true",
    )

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="ACCEL-",
        precomputed_findings=[fallback_finding],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
    )

    assert records == []
    assert todo_path.read_text(encoding="utf-8") == original_todo
    assert not discovery_dir.exists()


def test_bundle_regeneration_preserves_projected_task_status(tmp_path):
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
    index_path = bundle_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    task = index["bundles"]["objective/ops/root"]["tasks"][0]
    task["status"] = "completed"
    index["completed_task_ids"] = [task["task_id"]]
    index_path.write_text(json.dumps(index), encoding="utf-8")

    write_bundle_shards(
        bundle_dir=bundle_dir,
        repo_root=repo,
        todo_path=todo_path,
        records=records,
    )

    regenerated = json.loads(index_path.read_text(encoding="utf-8"))
    regenerated_task = regenerated["bundles"]["objective/ops/root"]["tasks"][0]
    assert regenerated_task["status"] == "completed"
    assert regenerated["completed_task_ids"] == ["ACCEL-002"]


def test_empty_objective_scan_refreshes_existing_todo_and_bundle_projections(tmp_path):
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
        persist_ast_dataset=False,
    )
    task_id = records[0].task_id
    todo_text = todo_path.read_text(encoding="utf-8")
    before_task, task_block = todo_text.split(f"## {task_id} ", 1)
    todo_path.write_text(
        before_task
        + f"## {task_id} "
        + task_block.replace(
            "- Status: todo",
            "- Status: completed",
            1,
        ),
        encoding="utf-8",
    )

    repeated = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="ACCEL-",
        max_findings=1,
        persist_ast_dataset=False,
    )

    assert repeated == []
    vector_index = json.loads(
        (bundle_dir / "todo_vector_index.json").read_text(encoding="utf-8")
    )
    assert vector_index["active_task_count"] == 0
    assert {
        record["task_id"]: record["status"]
        for record in vector_index["records"]
    }[task_id] == "completed"
    bundle_index = json.loads(
        (bundle_dir / "index.json").read_text(encoding="utf-8")
    )
    bundle_tasks = {
        task["task_id"]: task
        for bundle in bundle_index["bundles"].values()
        for task in bundle["tasks"]
    }
    assert bundle_tasks[task_id]["status"] == "completed"


def test_generate_objective_todos_reserves_ids_from_discovery_artifacts(tmp_path):
    repo, objective_path, todo_path = _seed_repo(tmp_path)
    discovery_dir = repo / "data" / "agent_supervisor" / "discovery"
    bundle_dir = repo / "data" / "agent_supervisor" / "objective_bundles"
    discovery_dir.mkdir(parents=True)
    (discovery_dir / "2026-07-22-accel-009-codebase-scan-deadbeef.md").write_text(
        "# Prior durable finding\n",
        encoding="utf-8",
    )

    records = generate_objective_todos(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="ACCEL-",
        max_findings=1,
    )

    assert [record.task_id for record in records] == ["ACCEL-010"]
    assert "## ACCEL-010 Close objective gap" in todo_path.read_text(encoding="utf-8")


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


def test_task_dependency_dag_materializes_all_prerequisite_kinds_with_provenance():
    graph = materialize_task_dependency_dag(
        [
            {
                "task_id": "TASK-A",
                "task_cid": "cid-a",
                "goal_id": "G0",
                "outputs": ["pkg/runtime.py"],
                "provides_imports": ["pkg.runtime"],
                "provides_interfaces": ["Runtime@1"],
                "provides_migrations": ["schema-2"],
                "provides_validations": ["runtime-green"],
            },
            {
                "task_id": "TASK-B",
                "task_cid": "cid-b",
                "goal_id": "G1",
                "parent_goal_ids": ["G0"],
                "inputs": ["pkg/runtime.py"],
                "required_imports": ["pkg.runtime"],
                "required_interfaces": ["Runtime@1"],
                "required_migrations": ["schema-2"],
                "validation_prerequisites": ["runtime-green"],
            },
        ],
        now=10_000,
    )

    assert {edge.kind for edge in graph.edges} == {
        "goal",
        "import",
        "interface",
        "output_input",
        "migration",
        "validation",
    }
    assert all(edge.provenance["field"] and edge.provenance["value"] for edge in graph.edges)
    assert graph.schedule[0].task_cid == "cid-a"
    assert graph.schedule[0].claimable is True
    assert graph.schedule[1].blocking_task_cids == ["cid-a"]


def test_task_dependency_dag_does_not_require_an_abstract_parent_goal_task():
    graph = materialize_task_dependency_dag(
        [
            {
                "task_id": "TASK-A",
                "task_cid": "cid-a",
                "goal_id": "G1.S1",
                "parent_goal_ids": ["G1"],
            }
        ]
    )

    assert graph.edges == []
    assert graph.repair_evidence == []
    assert graph.invalid_task_cids == []
    assert graph.schedule[0].claimable is True


def test_task_dependency_dag_resolves_explicit_goal_dependency_to_all_goal_tasks():
    graph = materialize_task_dependency_dag(
        [
            {
                "task_id": "TASK-A1",
                "task_cid": "cid-a1",
                "goal_id": "G0",
            },
            {
                "task_id": "TASK-A2",
                "task_cid": "cid-a2",
                "goal_id": "G0",
            },
            {
                "task_id": "TASK-B",
                "task_cid": "cid-b",
                "goal_id": "G1",
                "depends_on": ["G0"],
            },
        ]
    )

    goal_edges = [edge for edge in graph.edges if edge.target_task_cid == "cid-b"]
    assert {edge.source_task_cid for edge in goal_edges} == {"cid-a1", "cid-a2"}
    assert {edge.provenance["resolution"] for edge in goal_edges} == {"goal_id"}
    assert graph.repair_evidence == []
    scheduled = {item.task_cid: item for item in graph.schedule}
    assert scheduled["cid-b"].blocking_task_cids == ["cid-a1", "cid-a2"]


def test_bundle_payload_admits_only_the_dependency_closed_ready_member_slice(tmp_path):
    index_path = tmp_path / "slice-index.json"
    payload = {
        "source_todo": "tasks.todo.md",
        "bundles": {
            "objective/mixed": {
                "shard_path": "mixed.todo.md",
                "tasks": [
                    {
                        "task_id": "A",
                        "task_cid": "cid-a",
                        "outputs": ["ready.py"],
                    },
                    {
                        "task_id": "B",
                        "task_cid": "cid-b",
                        "depends_on": ["X"],
                        "outputs": ["deferred.py"],
                    },
                ],
            },
            "objective/prerequisite": {
                "shard_path": "prerequisite.todo.md",
                "tasks": [
                    {
                        "task_id": "X",
                        "task_cid": "cid-x",
                        "outputs": ["prerequisite.py"],
                    }
                ],
            },
        },
    }
    index_path.write_text(json.dumps(payload), encoding="utf-8")

    initial = {
        item["bundle_key"]: item for item in build_bundle_task_payloads(index_path)
    }
    mixed = initial["objective/mixed"]
    assert mixed["claimable"] is True
    assert mixed["ready_member_task_ids"] == ["A"]
    assert mixed["deferred_member_task_ids"] == ["B"]
    assert mixed["execution_slice_task_ids"] == ["A"]
    assert mixed["dependency_task_cids"] == []

    payload["bundles"]["objective/mixed"]["tasks"][0]["status"] = "completed"
    index_path.write_text(json.dumps(payload), encoding="utf-8")
    waiting = {
        item["bundle_key"]: item for item in build_bundle_task_payloads(index_path)
    }
    mixed = waiting["objective/mixed"]
    assert mixed["claimable"] is False
    assert mixed["ready_member_task_ids"] == []
    assert mixed["deferred_member_task_ids"] == ["B"]
    assert mixed["execution_slice_task_ids"] == ["B"]
    assert mixed["dependency_task_cids"] == ["cid-x"]

    replenished = {
        item["bundle_key"]: item
        for item in build_bundle_task_payloads(
            index_path,
            merge_receipts={
                "cid-x": {
                    "status": "succeeded",
                    "receipt_cid": "receipt-x",
                }
            },
        )
    }
    mixed = replenished["objective/mixed"]
    assert mixed["claimable"] is True
    assert mixed["ready_member_task_ids"] == ["B"]
    assert mixed["execution_slice_task_ids"] == ["B"]
    assert mixed["dependency_task_cids"] == []
    prerequisite = replenished["objective/prerequisite"]
    assert prerequisite["completed_member_task_ids"] == ["X"]
    assert prerequisite["ready_member_task_ids"] == []
    assert prerequisite["execution_slice_task_ids"] == []
    assert prerequisite["tasks"][0]["claimable"] is False


def test_active_member_fences_bundle_slice_from_duplicate_launch(tmp_path):
    index_path = tmp_path / "active-index.json"
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {
                    "objective/active": {
                        "shard_path": "active.todo.md",
                        "tasks": [
                            {
                                "task_id": "A",
                                "task_cid": "cid-a",
                                "status": "in_progress",
                            },
                            {"task_id": "B", "task_cid": "cid-b"},
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = build_bundle_task_payloads(index_path)[0]

    assert bundle["claimable"] is False
    assert bundle["active_member_task_ids"] == ["A"]
    assert bundle["execution_slice_task_ids"] == []
    assert bundle["ready_member_task_ids"] == ["B"]


def test_blocked_member_is_not_admitted_as_a_ready_bundle_slice(tmp_path):
    index_path = tmp_path / "blocked-index.json"
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {
                    "objective/blocked": {
                        "shard_path": "blocked.todo.md",
                        "tasks": [
                            {
                                "task_id": "A",
                                "task_cid": "cid-a",
                                "status": "blocked",
                            },
                            {
                                "task_id": "B",
                                "task_cid": "cid-b",
                                "depends_on": ["A"],
                            },
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    bundle = build_bundle_task_payloads(index_path)[0]

    assert bundle["claimable"] is False
    assert bundle["blocked_member_task_ids"] == ["A"]
    assert bundle["ready_member_task_ids"] == []
    assert bundle["deferred_member_task_ids"] == ["A", "B"]


def test_bundle_projection_absorbs_internal_cycles_but_keeps_cross_bundle_cycles(tmp_path):
    internal_index = tmp_path / "internal-index.json"
    internal_index.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {
                    "objective/internal": {
                        "shard_path": "internal.todo.md",
                        "tasks": [
                            {"task_id": "A", "task_cid": "cid-a", "depends_on": ["B"]},
                            {"task_id": "B", "task_cid": "cid-b", "depends_on": ["A"]},
                        ],
                    }
                },
            }
        ),
        encoding="utf-8",
    )

    internal = build_bundle_task_payloads(internal_index)[0]
    assert internal["dependency_task_cids"] == []
    assert internal["dependency_repair_evidence"] == []
    assert internal["claimable"] is True

    external_index = tmp_path / "external-index.json"
    external_index.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {
                    "objective/a": {
                        "shard_path": "a.todo.md",
                        "tasks": [{"task_id": "A", "task_cid": "cid-a", "depends_on": ["B"]}],
                    },
                    "objective/b": {
                        "shard_path": "b.todo.md",
                        "tasks": [{"task_id": "B", "task_cid": "cid-b", "depends_on": ["A"]}],
                    },
                },
            }
        ),
        encoding="utf-8",
    )

    external = build_bundle_task_payloads(external_index)
    assert all(payload["claimable"] is False for payload in external)
    assert all(
        any(item["kind"] == "dependency_cycle" for item in payload["dependency_repair_evidence"])
        for payload in external
    )

    resolved_index = tmp_path / "resolved-index.json"
    resolved_payload = json.loads(external_index.read_text(encoding="utf-8"))
    resolved_payload["bundles"]["objective/a"]["tasks"][0]["status"] = "completed"
    resolved_index.write_text(json.dumps(resolved_payload), encoding="utf-8")

    resolved = build_bundle_task_payloads(resolved_index)
    assert all(payload["claimable"] is True for payload in resolved)
    assert all(payload["dependency_task_cids"] == [] for payload in resolved)
    assert all(payload["dependency_repair_evidence"] == [] for payload in resolved)


def test_task_dependency_dag_requires_successful_merge_receipts_and_scores_critical_path():
    tasks = [
        {"task_id": "A", "task_cid": "cid-a", "depends_on": [], "priority": "P2", "created_at_ms": 1_000},
        {"task_id": "B", "task_cid": "cid-b", "depends_on": ["A"], "priority": "P0", "created_at_ms": 2_000},
        {"task_id": "C", "task_cid": "cid-c", "depends_on": ["B"], "priority": "P1", "created_at_ms": 3_000},
        {"task_id": "D", "task_cid": "cid-d", "priority": "P0", "created_at_ms": 4_000},
    ]

    blocked = materialize_task_dependency_dag(tasks, now=10_000)
    scheduled = {item.task_cid: item for item in blocked.schedule}
    assert scheduled["cid-a"].critical_path_length == 3
    assert scheduled["cid-a"].downstream_unlock_value == 2
    assert scheduled["cid-d"].slack == 2
    assert scheduled["cid-b"].claimable is False

    unblocked = materialize_task_dependency_dag(
        tasks,
        merge_receipts={"cid-a": {"status": "succeeded", "receipt_cid": "receipt-a"}},
        now=10_000,
    )
    unblocked_schedule = {item.task_cid: item for item in unblocked.schedule}
    assert unblocked_schedule["cid-a"].claimable is False
    assert unblocked_schedule["cid-b"].claimable is True


def test_external_authority_fence_rejects_local_status_and_merge_receipts(
    tmp_path,
):
    assert EXTERNAL_AUTHORITY_BENCHMARK_GOAL_IDS == {
        "HSSL-G201",
        "HSSL-G202",
        "HSSL-G203",
        "HSSL-G212",
        "HSSL-G220",
        "HSSL-G232",
        "HSSL-G241",
        "HSSL-G242",
        "HSSL-G243",
    }
    local_readiness = materialize_task_dependency_dag(
        [
            {
                "task_id": "LOCAL-G231-READINESS",
                "task_cid": "cid-local-g231-readiness",
                "goal_id": "HSSL-G231",
            },
            {
                "task_id": "LOCAL-G240-READINESS",
                "task_cid": "cid-local-g240-readiness",
                "goal_id": "HSSL-G240",
            },
        ]
    )
    assert local_readiness.invalid_task_cids == []
    assert all(item.claimable for item in local_readiness.schedule)

    duplicate_external = materialize_task_dependency_dag(
        [
            {
                "task_id": "FIRST-LOCAL-ALIAS",
                "task_cid": "cid-duplicate-authority",
                "goal_id": "LOCAL-G001",
            },
            {
                "task_id": "SECOND-EXTERNAL-ALIAS",
                "task_cid": "cid-duplicate-authority",
                "goal_id": "HSSL-G242",
                "status": "completed",
            },
        ],
        merge_receipts={
            "cid-duplicate-authority": {
                "status": "succeeded",
                "receipt_cid": "local-duplicate-receipt",
            }
        },
    )
    assert duplicate_external.invalid_task_cids == [
        "cid-duplicate-authority"
    ]
    assert duplicate_external.schedule[0].claimable is False
    assert any(
        item.kind == "external_authority_required"
        for item in duplicate_external.repair_evidence
    )

    tasks = [
        {
            "task_id": "KNOWN-GATE",
            "task_cid": "cid-known-gate",
            "goal_id": "HSSL-G202",
            "status": "completed",
        },
        {
            "task_id": "KNOWN-CHILD",
            "task_cid": "cid-known-child",
            "goal_id": "HSSL-G204",
            "parent_goal_ids": ["HSSL-G202"],
            "depends_on": ["KNOWN-GATE"],
        },
        {
            "task_id": "LEGACY-G242",
            "task_cid": "cid-legacy-g242",
            "goal_id": "HSSL-G242",
            "status": "completed",
        },
        {
            "task_id": "LEGACY-G243-CHILD",
            "task_cid": "cid-legacy-g243-child",
            "goal_id": "HSSL-G244",
            "parent_goal_ids": ["HSSL-G243"],
            "status": "completed",
        },
        {
            "task_id": "GENERIC-GATE",
            "task_cid": "cid-generic-gate",
            "goal_id": "EXT-G001",
            "completion_authority": "external",
            "status": "completed",
        },
        {
            "task_id": "GENERIC-CHILD",
            "task_cid": "cid-generic-child",
            "goal_id": "LOCAL-G002",
            "depends_on": ["GENERIC-GATE"],
        },
        {
            "task_id": "INDEPENDENT",
            "task_cid": "cid-independent",
            "goal_id": "LOCAL-G001",
        },
    ]
    forged_local_receipts = {
        "cid-known-gate": {
            "status": "succeeded",
            "receipt_cid": "local-known-receipt",
        },
        "cid-generic-gate": {
            "status": "succeeded",
            "receipt_cid": "local-generic-receipt",
        },
        "cid-legacy-g242": {
            "status": "succeeded",
            "receipt_cid": "local-g242-receipt",
        },
        "cid-legacy-g243-child": {
            "status": "succeeded",
            "receipt_cid": "local-g243-child-receipt",
        },
    }

    graph = materialize_task_dependency_dag(
        tasks,
        merge_receipts=forged_local_receipts,
        now=10_000,
    )
    schedule = {item.task_cid: item for item in graph.schedule}

    assert {
        "cid-known-gate",
        "cid-known-child",
        "cid-legacy-g242",
        "cid-legacy-g243-child",
        "cid-generic-gate",
    } <= set(graph.invalid_task_cids)
    assert schedule["cid-known-gate"].claimable is False
    assert schedule["cid-known-child"].claimable is False
    assert schedule["cid-legacy-g242"].claimable is False
    assert schedule["cid-legacy-g243-child"].claimable is False
    assert schedule["cid-generic-gate"].claimable is False
    assert schedule["cid-generic-child"].claimable is False
    assert schedule["cid-generic-child"].blocking_task_cids == [
        "cid-generic-gate"
    ]
    assert schedule["cid-independent"].claimable is True
    assert {
        item.task_cid
        for item in graph.repair_evidence
        if item.kind == "external_authority_required"
    } == {
        "cid-generic-gate",
        "cid-known-child",
        "cid-known-gate",
        "cid-legacy-g242",
        "cid-legacy-g243-child",
    }

    index_path = tmp_path / "external-authority-index.json"
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {
                    "objective/known": {
                        "shard_path": "known.todo.md",
                        "tasks": tasks[:4],
                    },
                    "objective/generic": {
                        "shard_path": "generic.todo.md",
                        "tasks": tasks[4:6],
                    },
                    "objective/local": {
                        "shard_path": "local.todo.md",
                        "tasks": tasks[6:],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    payloads = {
        payload["bundle_key"]: payload
        for payload in build_bundle_task_payloads(
            index_path,
            merge_receipts=forged_local_receipts,
        )
    }

    assert payloads["objective/known"]["completed_member_task_ids"] == []
    assert payloads["objective/known"]["claimable"] is False
    assert payloads["objective/generic"]["completed_member_task_ids"] == []
    assert payloads["objective/generic"]["claimable"] is False
    assert payloads["objective/local"]["ready_member_task_ids"] == [
        "INDEPENDENT"
    ]
    assert payloads["objective/local"]["claimable"] is True


def test_task_dag_keeps_durable_external_history_fenced_from_local_receipts():
    typed_evidence = [
        {
            "acceptance_criterion": "EXT-EVIDENCE",
            "provenance_cid": "bafkreiroutinghint",
            "metadata": {"external_operational_completion": True},
        }
    ]
    tasks = [
        {
            "task_id": "DURABLE-CID",
            "task_cid": "cid-durable-cid",
            "goal_id": "EXT-CID",
            "status": "completed",
            "external_completion_authority_cid": "bafkreiauthorityroutinghint",
        },
        {
            "task_id": "DURABLE-EVIDENCE",
            "task_cid": "cid-durable-evidence",
            "goal_id": "EXT-EVIDENCE",
            "status": "completed",
            "completion_evidence_records": typed_evidence,
        },
        {
            "task_id": "LOCAL",
            "task_cid": "cid-local",
            "goal_id": "LOCAL-G001",
        },
    ]
    graph = materialize_task_dependency_dag(
        tasks,
        merge_receipts={
            "cid-durable-cid": {"status": "succeeded"},
            "cid-durable-evidence": {"status": "succeeded"},
        },
    )
    schedule = {item.task_cid: item for item in graph.schedule}

    assert {
        "cid-durable-cid",
        "cid-durable-evidence",
    } <= set(graph.invalid_task_cids)
    assert schedule["cid-durable-cid"].claimable is False
    assert schedule["cid-durable-evidence"].claimable is False
    assert schedule["cid-local"].claimable is True


@pytest.mark.parametrize("reverse_records", (False, True))
def test_semantically_conflicting_duplicate_task_cids_fail_closed_in_all_orders(
    tmp_path,
    reverse_records,
):
    completed = {
        "task_id": "COMPLETED-A",
        "task_cid": "cid-conflicting-duplicate",
        "goal_id": "LOCAL-A",
        "status": "completed",
        "outputs": ["src/a.py"],
        "validation": ["test -f src/a.py"],
    }
    blocked = {
        "task_id": "BLOCKED-B",
        "task_cid": "cid-conflicting-duplicate",
        "goal_id": "LOCAL-B",
        "status": "todo",
        "outputs": ["src/b.py"],
        "depends_on": ["MISSING-PREREQUISITE"],
        "validation": ["test -f src/b.py"],
    }
    tasks = [completed, blocked]
    if reverse_records:
        tasks.reverse()

    graph = materialize_task_dependency_dag(tasks)

    assert graph.invalid_task_cids == ["cid-conflicting-duplicate"]
    assert graph.schedule[0].claimable is False
    assert any(
        repair.kind == "conflicting_duplicate_task_identity"
        for repair in graph.repair_evidence
    )

    index_path = tmp_path / f"duplicate-{int(reverse_records)}.json"
    index_path.write_text(
        json.dumps(
            {
                "source_todo": "tasks.todo.md",
                "bundles": {
                    "objective/a": {
                        "shard_path": "a.todo.md",
                        "tasks": [tasks[0]],
                    },
                    "objective/b": {
                        "shard_path": "b.todo.md",
                        "tasks": [tasks[1]],
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    payloads = build_bundle_task_payloads(index_path)

    assert all(
        payload["completed_member_task_ids"] == []
        and payload["claimable"] is False
        for payload in payloads
    )


@pytest.mark.parametrize("completed_record_first", (False, True))
def test_semantically_equivalent_duplicate_task_cids_merge_terminal_status(
    completed_record_first,
):
    todo = {
        "task_id": "ALIAS-TODO",
        "task_cid": "cid-equivalent-duplicate",
        "goal_id": "LOCAL-G001",
        "status": "todo",
        "outputs": ["src/shared.py"],
        "depends_on": [],
        "validation": ["test -f src/shared.py"],
    }
    completed = {
        **todo,
        "task_id": "ALIAS-COMPLETED",
        "status": "completed",
    }
    tasks = [completed, todo] if completed_record_first else [todo, completed]

    graph = materialize_task_dependency_dag(tasks)

    assert graph.invalid_task_cids == []
    assert graph.repair_evidence == []
    assert graph.nodes["cid-equivalent-duplicate"].status == "completed"
    assert graph.nodes["cid-equivalent-duplicate"].metadata[
        "task_id_aliases"
    ] == ["ALIAS-COMPLETED", "ALIAS-TODO"]
    assert graph.schedule[0].claimable is False


def test_task_dependency_dag_bounds_cycle_and_missing_dependency_repairs_without_deadlock():
    graph = materialize_task_dependency_dag(
        [
            {"task_id": "A", "task_cid": "cid-a", "depends_on": ["B"]},
            {"task_id": "B", "task_cid": "cid-b", "depends_on": ["A"]},
            {"task_id": "C", "task_cid": "cid-c", "depends_on": ["not-present"]},
            {"task_id": "D", "task_cid": "cid-d"},
        ],
        max_repair_evidence=2,
    )

    assert len(graph.repair_evidence) == 2
    assert {"cid-a", "cid-b", "cid-c"}.issubset(graph.invalid_task_cids)
    schedule = {item.task_cid: item for item in graph.schedule}
    assert schedule["cid-d"].claimable is True
    assert schedule["cid-a"].claimable is False
    assert schedule["cid-b"].claimable is False
    assert schedule["cid-c"].claimable is False


def test_task_dependency_dag_handles_long_generated_chains_without_recursion():
    tasks = [
        {
            "task_id": f"TASK-{index}",
            "task_cid": f"cid-{index}",
            "depends_on": [f"TASK-{index - 1}"] if index else [],
        }
        for index in range(1_250)
    ]

    graph = materialize_task_dependency_dag(tasks, max_repair_evidence=4)

    assert graph.repair_evidence == []
    assert len(graph.schedule) == 1_250
    assert graph.schedule[0].task_cid == "cid-0"
    assert graph.schedule[0].critical_path_length == 1_250


def test_objective_findings_preserve_complete_conflict_surface_metadata(tmp_path):
    repo, objective_path, _todo_path = _seed_repo(tmp_path)
    objective_path.write_text(
        """# Objective Heap

## VAIOS-G100 Conflict-aware objective

- Status: active
- Priority: P0
- Track: runtime
- Goal: Add a generated runtime interface.
- Evidence: missing_runtime_contract
- Outputs: src/runtime_router.py, test/runtime_router_test.py
- Predicted files: schemas/runtime.json
- AST query: CapabilityRouter.dispatch_task, RuntimeAdapter
- Interfaces: RuntimeAPI@2
- Submodules: vendor/runtime
- Generated artifacts: dist/runtime-schema.json
- Allow concurrent with: TASK-SAFE
- Validation: true
""",
        encoding="utf-8",
    )

    finding = scan_objective_gaps(repo, objective_path=objective_path, max_findings=1)[0]
    record = objective_finding_conflict_record("AUTO-100", finding)

    assert finding.predicted_files == [
        "src/runtime_router.py",
        "test/runtime_router_test.py",
        "schemas/runtime.json",
    ]
    assert finding.ast_symbols == ["CapabilityRouter.dispatch_task", "RuntimeAdapter"]
    assert finding.interfaces == ["RuntimeAPI@2"]
    assert finding.submodules == ["vendor/runtime"]
    assert finding.generated_artifacts == ["dist/runtime-schema.json"]
    assert record["files"] == finding.predicted_files
    assert record["allow_concurrent_with"] == ["TASK-SAFE"]


def test_task_planning_graph_combines_dependency_readiness_and_conflict_coloring():
    planning = materialize_task_planning_graph(
        [
            {
                "task_id": "TASK-A",
                "task_cid": "cid-a",
                "predicted_files": ["src/shared.py"],
                "ast_symbols": ["Shared.update"],
            },
            {
                "task_id": "TASK-B",
                "task_cid": "cid-b",
                "depends_on": ["TASK-A"],
                "predicted_files": ["src/shared.py"],
                "ast_symbols": ["Shared.update"],
            },
        ],
        now=10_000,
    )

    assert planning.claimable_task_cids == ["cid-a"]
    assert set(planning.conflict_graph.surfaces) == {"cid-a", "cid-b"}
    assert len(planning.conflict_graph.lanes) == 2
    payload = planning.to_dict()
    assert payload["dependency_dag"]["claimable_task_cids"] == ["cid-a"]
    assert payload["planning_decisions"]
