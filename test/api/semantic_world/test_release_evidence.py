"""SAWM-044 release, migration, and limitation report."""

from __future__ import annotations

import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_release import (
    SemanticWorldMigrationReceipt,
    SemanticWorldReleaseReport,
    SemanticWorldRollbackTarget,
    build_current_tree_release_report,
)


REPORT = (
    Path(__file__).resolve().parents[3]
    / "artifacts"
    / "agent_supervisor"
    / "semantic_addressed_world_model"
    / "SAWM-044-release.json"
)
MARKDOWN = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "architecture"
    / "SEMANTIC_ADDRESSED_WORLD_MODEL_RELEASE.md"
)


def test_release_report_does_not_claim_board_completion() -> None:
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["schema"] == "SemanticWorldReleaseReport@1"
    assert payload["completion_authority"] is False
    assert payload["cas_completed"] is False
    assert payload["generation_published"] is False
    assert payload["remaining_todo"] == 25
    assert MARKDOWN.exists()
    text = MARKDOWN.read_text(encoding="utf-8")
    assert "not DuckDB completion" in text
    assert "todo" in text
    report = build_current_tree_release_report()
    assert isinstance(report, SemanticWorldReleaseReport)
    assert isinstance(report.rollback, SemanticWorldRollbackTarget)
    assert isinstance(report.migration, SemanticWorldMigrationReceipt)
    assert report.completion_authority is False
    assert report.released is False
    assert report.safety_floor_violations == 0
    assert "overlay tests are not DuckDB completion evidence" in report.blockers
    assert report.migration.migrated is False
    assert report.rollback.extra_gate_generation == 48
