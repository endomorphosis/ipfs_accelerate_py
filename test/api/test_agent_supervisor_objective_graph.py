from __future__ import annotations

import json
import subprocess
import sys
import types
from pathlib import Path

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
from ipfs_accelerate_py.agent_supervisor.merge_resolver import (
    MergeResolverCliConfig,
    build_llm_merge_resolver_invoker,
    build_merge_prompt_callback,
    build_namespace_merge_resolver_runner,
    build_resolver_payload_callback,
    main as merge_resolver_main,
    run_configured_merge_resolver_cli,
)
from ipfs_accelerate_py.agent_supervisor.goal_completion import CompletionEvidence
from ipfs_accelerate_py.agent_supervisor.implementation_supervisor_runner import (
    build_goal_completion_projection,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    materialize_task_dependency_dag,
    materialize_task_planning_graph,
    objective_fingerprint,
    objective_finding_conflict_record,
)
from ipfs_accelerate_py.agent_supervisor.objective_tracker import (
    completion_tree_identity,
    migrate_legacy_objective_goals,
    reconcile_objective_goal_completion,
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
        "schedulable": False,
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
    assert graph["schedulable_goal_ids"] == ["G10.S3.1"]
    assert graph["terminal_goal_ids"] == ["G10.S3.2"]
    assert {node["acceptance_criterion"] for node in graph["evidence_nodes"]} == {
        "criterion one",
        "criterion two",
        "child criterion",
        "legacy criterion",
    }
    assert {edge["kind"] for edge in graph["evidence_edges"]} == {"requires_evidence"}


def test_objective_schedule_only_includes_active_and_reopened_states() -> None:
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

    assert [record.goal_id for record in schedule] == ["G-A", "G-R"]


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
        "analyzer_health": {"status": "healthy"},
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
                    "finished_at": observed_at,
                    "binding": binding,
                },
                {
                    "member_id": "independent-audit",
                    "evidence_channel": "audit",
                    "receipt_cid": "bafy-independent-audit",
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
    assert operator_row["analyzer_health"]["evidence"] == {"status": "healthy"}
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
    todo_text = todo_path.read_text(encoding="utf-8")
    assert "## ACCEL-002 Close objective gap" in todo_text
    assert "- Bundle: objective/ops/root" in todo_text

    shard = bundle_dir / "objective-ops-root.todo.md"
    assert shard.exists()
    assert "## ACCEL-002 Close objective gap" in shard.read_text(encoding="utf-8")
    index_path = bundle_dir / "index.json"
    index = json.loads(index_path.read_text(encoding="utf-8"))
    assert index["bundles"]["objective/ops/root"]["tasks"][0]["task_id"] == "ACCEL-002"
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
    assert payloads[0]["task_conflict_graph"]["surfaces"]

    submitted: list[dict[str, object]] = []

    class FakeQueue:
        def submit(self, **kwargs):
            submitted.append(kwargs)
            return "queued-1"

    task_ids = submit_bundle_tasks(index_path, queue=FakeQueue())

    assert task_ids == ["queued-1"]
    assert submitted[0]["task_type"] == "codex.todo_bundle"
    assert submitted[0]["payload"]["bundle_key"] == "objective/ops/root"


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
    assert next(item for item in unblocked.schedule if item.task_cid == "cid-b").claimable is True


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
