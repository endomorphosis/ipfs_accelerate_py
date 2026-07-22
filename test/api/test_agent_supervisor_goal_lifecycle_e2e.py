from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.goal_completion import (
    CompletionEvidence,
    ContradictionEvidence,
    GoalState,
    evaluate_goal_completion,
    reopen_goal_for_contradictions,
)
from ipfs_accelerate_py.agent_supervisor.implementation_supervisor_runner import (
    persist_goal_completion_projection,
    persist_supervisor_scan_receipt,
)
from ipfs_accelerate_py.agent_supervisor.objective_graph import (
    generate_objective_todos_result,
    scan_objective_gaps,
)
from ipfs_accelerate_py.agent_supervisor.objective_tracker import (
    completion_tree_identity,
    migrate_legacy_objective_goals,
)
from ipfs_accelerate_py.agent_supervisor.scan_receipts import ScanTerminalReason


CRITERION = "ref212canary9f8e7d"


def _git(repo: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr or completed.stdout
    return completed.stdout.strip()


def _seed_repository(tmp_path: Path, *, legacy_state: str = "active") -> tuple[Path, Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Regression Test")
    _git(repo, "config", "user.email", "regression@example.invalid")
    source = repo / "src" / "dispatch.py"
    source.parent.mkdir()
    source.write_text("def dispatch(value):\n    return value\n", encoding="utf-8")
    objective_path = repo / "objective.md"
    objective_path.write_text(
        f"""# Objective

## G10.S4 Truthful completion

- Status: {legacy_state}
- Evidence: {CRITERION}
- Acceptance: {CRITERION}
- Outputs: src/dispatch.py
- Validation: python -m pytest tests/test_dispatch.py -q
- Predicted files: src/dispatch.py
- AST symbols: dispatch
""",
        encoding="utf-8",
    )
    todo_path = repo / "todo.md"
    todo_path.write_text("# Autonomous work board\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed objective")
    return repo, objective_path, todo_path


def _completion_gate(identity: object, observed_at: datetime) -> dict[str, object]:
    binding = {
        "repository_id": identity.repository_id,
        "tree_id": identity.tree_id,
        "analyzer_version": "objective-e2e/v1",
        "configuration_revision": "config-v1",
        "objective_revision": "objective-v1",
    }
    members = [
        {
            "member_id": "normal-exhaustive",
            "evidence_channel": "exhaustive",
            "receipt_cid": "bafy-normal-exhaustive",
            "finished_at": observed_at.isoformat(),
            "binding": binding,
        },
        {
            "member_id": "independent-audit",
            "evidence_channel": "audit",
            "receipt_cid": "bafy-independent-audit",
            "finished_at": observed_at.isoformat(),
            "binding": binding,
        },
    ]
    return {
        "coverage": {
            "verified": True,
            "repository_tree": identity.tree_id,
            "evaluated_at": observed_at.isoformat(),
            "criteria": [{"criterion": CRITERION, "status": "verified"}],
        },
        "analyzer_health": {
            "status": "healthy",
            "healthy": True,
            "safe_for_completion_reasoning": True,
        },
        "exhaustion_quorum": {
            "satisfied": True,
            "quorum_met": True,
            "required_members": 2,
            "member_count": 2,
            "binding": binding,
            "members": members,
        },
    }


def _evidence(identity: object, observed_at: datetime) -> CompletionEvidence:
    return CompletionEvidence(
        acceptance_criterion=CRITERION,
        producing_task_or_scan="REF-212-e2e",
        validation_receipt="bafy-dispatch-validation",
        validation_passed=True,
        repository_id=identity.repository_id,
        repository_tree=identity.tree_id,
        observed_at=observed_at,
        freshness={"fresh": True},
        provenance_cid="bafy-completion-lineage",
    )


def test_stale_fingerprints_cannot_complete_goal_and_reopened_goal_refills_board(
    tmp_path: Path,
) -> None:
    repo, objective_path, todo_path = _seed_repository(tmp_path)
    discovery_dir = repo / "state" / "discovery"
    bundle_dir = repo / "state" / "bundles"
    state_dir = repo / "state"
    state_dir.mkdir(exist_ok=True)
    strategy_path = state_dir / "serial_strategy.json"
    events_path = state_dir / "events.jsonl"
    strategy_path.write_text("{}\n", encoding="utf-8")

    initial_findings = scan_objective_gaps(
        repo,
        objective_path=objective_path,
        max_findings=1,
    )
    assert len(initial_findings) == 1
    stale_fingerprint = initial_findings[0].fingerprint

    duplicate_only = generate_objective_todos_result(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="E2E-",
        max_findings=1,
        seen_fingerprints=[stale_fingerprint],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
        scan_mode="drained_exhaustive",
    )
    assert duplicate_only.terminal_reason is ScanTerminalReason.DUPLICATE_ONLY
    assert duplicate_only.safe_for_completion_reasoning is False
    projection = persist_supervisor_scan_receipt(
        duplicate_only,
        scan_kind="objective",
        state_dir=state_dir,
        state_prefix="serial",
        strategy_path=strategy_path,
        events_path=events_path,
    )
    assert projection["terminal_reason"] == "duplicate_only"
    assert projection["health"] == "healthy"
    assert projection["safe_for_completion_reasoning"] is False

    now = datetime.now(timezone.utc)
    identity = completion_tree_identity(repo, objective_path=objective_path)
    gate = _completion_gate(identity, now)
    completion = evaluate_goal_completion(
        current_state=GoalState.VERIFIED_COMPLETE,
        acceptance_criteria=[CRITERION],
        evidence=[_evidence(identity, now)],
        tasks_complete=True,
        repository_id=identity.repository_id,
        repository_tree=identity.tree_id,
        now=now,
        analysis_result=duplicate_only.to_dict(),
        **gate,
    )
    assert completion.state is GoalState.REOPENED
    assert completion.verified is False
    assert "analysis_not_completion_safe" in completion.reason_codes

    contradiction = ContradictionEvidence(
        goal_id="G10.S4",
        kind="mapped_finding",
        summary="A later independent scan still finds the dispatch proof gap.",
        impacted_criteria=(CRITERION,),
        invalidated_evidence=("bafy-completion-lineage",),
        source_receipt={
            "receipt_cid": "bafy-later-relevant-finding",
            "finding_id": stale_fingerprint,
            "tree_id": identity.tree_id,
        },
    )
    reopening = reopen_goal_for_contradictions(
        goal_id="G10.S4",
        current_state=GoalState.VERIFIED_COMPLETE,
        contradictions=[contradiction],
        historical_completion_receipts=[
            {"goal_id": "G10.S4", "receipt_id": "bafy-completion-lineage"}
        ],
        now=now,
    )
    assert reopening.reopened is True
    assert reopening.reopening_receipt["historical_completion_receipt_ids"] == [
        "bafy-completion-lineage"
    ]

    objective_path.write_text(
        objective_path.read_text(encoding="utf-8").replace(
            "- Status: active", "- Status: reopened"
        ),
        encoding="utf-8",
    )
    generated = generate_objective_todos_result(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        discovery_dir=discovery_dir,
        bundle_dir=bundle_dir,
        task_prefix="E2E-",
        depends_on=["REF-212"],
        max_findings=1,
        seen_fingerprints=[stale_fingerprint],
        force_goal_ids=["G10.S4"],
        persist_ast_dataset=False,
        write_todo_vector_index=False,
        scan_mode="reopened_goal_refill",
    )
    assert generated.terminal_reason is ScanTerminalReason.GENERATED
    assert generated.generated_count == 1
    assert generated.items[0].finding.goal_id == "G10.S4"
    assert generated.items[0].finding.fingerprint == stale_fingerprint
    board = todo_path.read_text(encoding="utf-8")
    assert board.count("## E2E-001 ") == 1
    assert "- Depends on: REF-212" in board
    assert "- Canonical task key: task/v1/" in board
    assert stale_fingerprint[:12] in generated.items[0].discovery_path.name

    completion_projection = persist_goal_completion_projection(
        {"G10.S4": {**completion.to_dict(), "reopen_reasons": [contradiction.summary]}},
        state_dir=state_dir,
        state_prefix="serial",
        strategy_path=strategy_path,
    )
    restarted_state = json.loads(strategy_path.read_text(encoding="utf-8"))
    assert completion_projection["by_goal_id"]["G10.S4"]["lifecycle_state"] == "reopened"
    assert restarted_state["goal_completion_by_goal_id"]["G10.S4"]["stale_evidence"] == []
    assert restarted_state["goal_completion_by_goal_id"]["G10.S4"]["reopen_reasons"]
    assert restarted_state["latest_attempted_scan"]["receipt_cid"] == duplicate_only.receipt_cid


def test_restart_after_legacy_migration_preserves_lineage_quorum_and_operator_truth(
    tmp_path: Path,
) -> None:
    repo, objective_path, todo_path = _seed_repository(tmp_path, legacy_state="completed")
    state_dir = repo / "state"
    state_dir.mkdir()
    strategy_path = state_dir / "bundle_strategy.json"
    strategy_path.write_text('{"scheduler_mode": "bundle"}\n', encoding="utf-8")
    now = datetime.now(timezone.utc)
    identity = completion_tree_identity(repo, objective_path=objective_path)
    gate = _completion_gate(identity, now)
    evidence = _evidence(identity, now)

    migration = migrate_legacy_objective_goals(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_evidence_records={"G10.S4": [evidence]},
        completion_gate_records={"G10.S4": gate},
        now=now.isoformat(),
    )
    assert migration.verified_goal_ids == ["G10.S4"]
    migrated_text = objective_path.read_text(encoding="utf-8")
    assert "- Status: verified_complete" in migrated_text
    assert "bafy-completion-lineage" in migrated_text
    assert "bafy-normal-exhaustive" in migrated_text
    assert "bafy-independent-audit" in migrated_text

    projection = persist_goal_completion_projection(
        {record["goal_id"]: record["completion_decision"] for record in migration.records},
        state_dir=state_dir,
        state_prefix="bundle",
        strategy_path=strategy_path,
        migration=migration,
    )
    restarted_payload = json.loads(strategy_path.read_text(encoding="utf-8"))
    restarted_goal = restarted_payload["goal_completion_by_goal_id"]["G10.S4"]
    assert restarted_goal["lifecycle_state"] == "verified_complete"
    assert restarted_goal["confidence"] == 1.0
    assert restarted_goal["analyzer_health"]["status"] == "healthy"
    assert restarted_goal["exhaustion_quorum"]["satisfied"] is True
    assert restarted_payload["scheduler_mode"] == "bundle"
    assert projection["migration"]["verified_goal_ids"] == ["G10.S4"]

    replay = migrate_legacy_objective_goals(
        repo_root=repo,
        objective_path=objective_path,
        todo_path=todo_path,
        completion_evidence_records={"G10.S4": [evidence]},
        completion_gate_records={"G10.S4": gate},
        now=now.isoformat(),
    )
    assert replay.changed is False
    assert replay.candidate_goal_ids == []
    assert objective_path.read_text(encoding="utf-8") == migrated_text
