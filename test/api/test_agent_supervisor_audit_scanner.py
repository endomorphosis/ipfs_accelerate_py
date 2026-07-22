from __future__ import annotations

import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.audit_scanner import (
    AuditFindingStatus,
    audit_codebase_findings,
    classify_audit_findings,
)
from ipfs_accelerate_py.agent_supervisor.backlog_refinery import (
    CodebaseFinding,
    record_codebase_scan_findings,
)
from ipfs_accelerate_py.agent_supervisor.dataset_store import ObjectiveDatasetStore
from ipfs_accelerate_py.agent_supervisor.scan_receipts import (
    ExhaustionBinding,
    RefillScanResult,
    ScanTerminalReason,
    evaluate_exhaustion_quorum,
    objective_revision,
    scan_configuration_revision,
)


def _git(repo: Path, *args: str) -> None:
    result = subprocess.run(["git", *args], cwd=repo, text=True, capture_output=True)
    assert result.returncode == 0, result.stderr


def _repo(tmp_path: Path, source: str = "VALUE = 1\n") -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init")
    _git(repo, "config", "user.name", "Audit Test")
    _git(repo, "config", "user.email", "audit@example.invalid")
    (repo / "source.py").write_text(source, encoding="utf-8")
    (repo / "todo.md").write_text("# Agent Todos\n", encoding="utf-8")
    _git(repo, "add", ".")
    _git(repo, "commit", "-m", "seed audit repository")
    return repo


def _finding(path: str, line: int, snippet: str, fingerprint: str) -> CodebaseFinding:
    return CodebaseFinding(
        fingerprint=fingerprint,
        kind="annotated_followup",
        priority="P3",
        track="runtime",
        root_relative_path=path,
        line_number=line,
        snippet=snippet,
        summary=f"Resolve code annotation in {path}:{line}",
        validation=f"python3 -m py_compile {path}",
    )


def test_audit_classification_reports_known_stale_changed_and_novel_separately() -> None:
    baseline = [
        _finding("known.py", 1, "# TODO: same", "normal-a"),
        _finding("stale.py", 2, "# TODO: removed", "normal-b"),
        _finding("changed.py", 3, "# TODO: old", "normal-c"),
    ]
    current = [
        _finding("known.py", 1, "# TODO: same", "poisoned-different-normal-hash"),
        _finding("changed.py", 3, "# TODO: new", "normal-d"),
        _finding("novel.py", 4, "# TODO: new path", "normal-e"),
    ]

    records = classify_audit_findings(current, baseline)
    by_status = {
        status: [record for record in records if record.status is status]
        for status in AuditFindingStatus
    }

    assert {status: len(items) for status, items in by_status.items()} == {
        AuditFindingStatus.KNOWN: 1,
        AuditFindingStatus.STALE: 1,
        AuditFindingStatus.CHANGED: 1,
        AuditFindingStatus.NOVEL: 1,
    }
    assert by_status[AuditFindingStatus.KNOWN][0].current_content_revision == by_status[AuditFindingStatus.KNOWN][0].prior_content_revision
    assert by_status[AuditFindingStatus.CHANGED][0].current_content_revision != by_status[AuditFindingStatus.CHANGED][0].prior_content_revision


def test_audit_ignores_normal_seen_set_and_persists_deduplicated_baseline(tmp_path: Path) -> None:
    repo = _repo(tmp_path, "# TODO: independent evidence\n")
    store = ObjectiveDatasetStore(repo / "data" / "agent_supervisor" / "objective_datasets")
    normal_seen = {"0" * 40, "f" * 40}

    first = audit_codebase_findings(
        repo,
        dataset_store=store,
        normal_seen_fingerprints=normal_seen,
    )
    frozen_seen = set(normal_seen)
    second = audit_codebase_findings(
        repo,
        dataset_store=store,
        normal_seen_fingerprints=normal_seen,
    )

    assert normal_seen == frozen_seen
    assert first.counts == {"known": 0, "stale": 0, "changed": 0, "novel": 1}
    assert second.counts == {"known": 1, "stale": 0, "changed": 0, "novel": 0}
    assert first.binding.tree_id == second.binding.tree_id
    assert first.snapshot_artifact is not None and second.snapshot_artifact is not None
    assert first.snapshot_artifact.jsonl_path == second.snapshot_artifact.jsonl_path
    assert second.quorum.count == 1


def _receipt(binding: ExhaustionBinding, channel: str, *, offset: int = 0) -> RefillScanResult[object]:
    started = datetime(2026, 1, 1, tzinfo=timezone.utc) + timedelta(seconds=offset)
    return RefillScanResult(
        terminal_reason=ScanTerminalReason.EXHAUSTED,
        scan_mode="audit" if channel == "audit" else "drained_exhaustive",
        analyzer_version=binding.analyzer_version,
        repository_id=binding.repository_id,
        tree_id=binding.tree_id,
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        metadata={
            "exhaustion_binding": binding.to_dict(),
            "evidence_channel": channel,
            "exhaustive": True,
            "coverage_complete": True,
            "analyzer_health": {"status": "healthy"},
        },
    )


def test_exhaustion_quorum_deduplicates_repeats_and_invalidates_changed_context() -> None:
    binding = ExhaustionBinding(
        "repo",
        "tree-a",
        "analyzer/v1",
        scan_configuration_revision({"limit": 5, "flags": ["a", "b"]}),
        objective_revision("objective-a"),
    )
    normal = _receipt(binding, "normal", offset=0)
    repeated_normal = _receipt(binding, "normal", offset=5)
    audit = _receipt(binding, "audit", offset=10)

    quorum = evaluate_exhaustion_quorum(
        [normal, repeated_normal, audit], binding=binding, required_members=2
    )
    assert quorum.satisfied
    assert quorum.count == 2
    assert len(quorum.duplicates) == 1

    changed = ExhaustionBinding(
        binding.repository_id,
        "tree-b",
        binding.analyzer_version,
        binding.configuration_revision,
        binding.objective_revision,
    )
    invalidated = evaluate_exhaustion_quorum(
        quorum.members, binding=changed, required_members=2
    )
    assert not invalidated.satisfied
    assert invalidated.count == 0
    assert {reason for item in invalidated.invalidated for reason in item["reasons"]} == {"tree_mismatch"}


def test_configuration_and_objective_revisions_are_canonical_and_binding_sensitive() -> None:
    assert scan_configuration_revision({"a": 1, "b": [2, 3]}) == scan_configuration_revision({"b": [2, 3], "a": 1})
    assert scan_configuration_revision({"a": 1}) != scan_configuration_revision({"a": 2})
    assert objective_revision("one") != objective_revision("two")


def test_normal_and_independent_audit_channels_form_completion_quorum(tmp_path: Path) -> None:
    repo = _repo(tmp_path)
    dataset_dir = repo / "datasets"
    thresholds = {"max_excluded_file_ratio": 0.75}
    audit = audit_codebase_findings(
        repo,
        dataset_dir=dataset_dir,
        health_thresholds=thresholds,
    )
    assert audit.receipt.terminal_reason is ScanTerminalReason.EXHAUSTED
    assert audit.quorum.count == 1
    assert audit.receipt.safe_for_completion_reasoning is False

    normal = record_codebase_scan_findings(
        todo_path=repo / "todo.md",
        state_path=None,
        strategy_path=repo / "state" / "strategy.json",
        discovery_dir=repo / "discovery",
        dataset_dir=dataset_dir,
        repo_root=repo,
        max_findings=5,
        health_thresholds=thresholds,
    )
    assert normal.terminal_reason is ScanTerminalReason.EXHAUSTED
    assert normal.metadata["exhaustion_quorum"]["member_count"] == 2
    assert normal.safe_for_completion_reasoning is True
