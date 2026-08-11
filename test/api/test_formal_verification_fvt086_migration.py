from __future__ import annotations

import importlib.util
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_PATH = REPO_ROOT / "tools/logic/migrate_fvt086_to_fvt101.py"
SPEC = importlib.util.spec_from_file_location(
    "migrate_fvt086_to_fvt101", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
migration = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(migration)


def _card(task_id: str, dependencies: str, *, source: bool = False) -> str:
    completion = """
- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
"""
    authority = "- Completion authority: local\n"
    return (
        f"## {task_id} Example\n"
        f"{completion}"
        f"- Depends on: {dependencies}\n"
        f"{authority}"
        "- Acceptance: exact\n"
    )


def _board_and_shards() -> tuple[str, dict[str, str]]:
    blocks = {
        "FVT-086": _card("FVT-086", "FVT-055, FVT-073, FVT-087", source=True),
        "FVT-088": _card("FVT-088", "FVT-084, FVT-087, FVT-086, FVT-085"),
        "FVT-092": _card("FVT-092", "FVT-086"),
        "FVT-099": _card("FVT-099", "FVT-086, FVT-092, FVT-093, FVT-095"),
    }
    board = "\n\n".join(block.rstrip() for block in blocks.values()) + "\n"
    shards = {
        task_id: f"# shard\n\n{block}" for task_id, block in blocks.items()
    }
    return board, shards


def _indexed_task(task_id: str, dependencies: list[str]) -> dict:
    return {
        "task_id": task_id,
        "canonical_task_cid": migration.TASK_CIDS[task_id],
        "canonical_task_key": f"task/v1/{task_id.lower()}",
        "status": "todo",
        "is_schedulable": True,
        "review_only": False,
        "completion_authority": "local",
        "depends_on": list(dependencies),
        "dependency_task_ids": list(dependencies),
        "dependency_task_cids": list(dependencies),
        "validation_receipts": [],
        "work_contract": {"old": task_id},
        "work_contract_id": f"work-{task_id}",
        "task_work_contract": {"old": task_id},
        "task_work_contract_id": f"task-work-{task_id}",
        "conflict_surface": {
            "work_contract": {"old": task_id},
            "work_contract_id": f"work-{task_id}",
            "task_work_contract": {"old": task_id},
            "task_work_contract_id": f"task-work-{task_id}",
        },
    }


def _index_payload() -> dict:
    dependencies = {
        "FVT-086": ["FVT-055", "FVT-073", "FVT-087"],
        "FVT-088": ["FVT-084", "FVT-087", "FVT-086", "FVT-085"],
        "FVT-092": ["FVT-086"],
        "FVT-099": ["FVT-086", "FVT-092", "FVT-093", "FVT-095"],
        "FVT-101": ["FVT-055", "FVT-073", "FVT-087"],
    }
    return {
        "bundles": {
            task_id: {
                "tasks": [_indexed_task(task_id, task_dependencies)]
            }
            for task_id, task_dependencies in dependencies.items()
        }
    }


def test_exact_markdown_migration_retires_source_and_rewires_only_reviewed_edges():
    board, shards = _board_and_shards()

    rewritten, rewritten_shards = migration.rewrite_markdown_documents(
        board, shards
    )

    source = migration.task_block(rewritten, "FVT-086")
    assert "- Status: blocked" in source
    assert "- Completion: superseded:FVT-101" in source
    assert "- Is schedulable: false" in source
    assert "- Review only: true" in source
    assert "- Completion authority: none" in source
    assert "- Supersession completion authority: none" in source
    assert "- Acceptance: exact\n\n## FVT-088" in rewritten
    for task_id in migration.DIRECT_DEPENDENT_TASK_IDS:
        block = migration.task_block(rewritten, task_id)
        depends = next(
            line for line in block.splitlines() if line.startswith("- Depends on:")
        )
        assert "FVT-086" not in depends
        assert depends.count("FVT-101") == 1
        assert migration.task_block(
            rewritten_shards[task_id], task_id
        ).strip() == block.strip()


def test_exact_markdown_migration_rejects_an_unreviewed_dependency_user():
    board, shards = _board_and_shards()
    board += _card("FVT-777", "FVT-086")

    with pytest.raises(migration.MigrationError, match="closure changed"):
        migration.rewrite_markdown_documents(board, shards)


def test_index_migration_preserves_cids_and_invalidates_only_downstream_contracts():
    payload = _index_payload()

    rewritten, before = migration.rewrite_bundle_index(payload)
    tasks = migration._bundle_tasks(rewritten)

    source = tasks["FVT-086"]
    assert source["status"] == "blocked"
    assert source["is_schedulable"] is False
    assert source["review_only"] is True
    assert source["completion_authority"] == "none"
    assert source["superseded_by"] == "FVT-101"
    assert source["work_contract_id"] == "work-FVT-086"
    for task_id in migration.DIRECT_DEPENDENT_TASK_IDS:
        task = tasks[task_id]
        assert task["canonical_task_cid"] == migration.TASK_CIDS[task_id]
        assert "FVT-086" not in task["depends_on"]
        assert task["depends_on"].count("FVT-101") == 1
        assert task["dependency_task_ids"] == task["depends_on"]
        assert task["dependency_task_cids"] == task["depends_on"]
        assert "work_contract_id" not in task
        assert "task_work_contract_id" not in task
        assert "work_contract_id" not in task["conflict_surface"]
        assert before[task_id]["work_contract_id"] == f"work-{task_id}"


def test_index_migration_rejects_prior_downstream_receipts():
    payload = _index_payload()
    tasks = migration._bundle_tasks(payload)
    tasks["FVT-092"]["validation_receipts"] = ["receipt"]

    with pytest.raises(migration.MigrationError, match="validation receipts"):
        migration.rewrite_bundle_index(payload)


def test_receipt_evidence_paths_are_repository_relative(tmp_path):
    repo_root = tmp_path / "checkout"
    event_path = repo_root / "data/live/events.jsonl"
    backup_path = repo_root / "data/live/migrations/run/backup/index.json"
    preflight = {
        "fvt101_completion_receipt": {
            "event_path": str(event_path),
            "status": "succeeded",
        }
    }
    snapshot = [
        {
            "path": "bundles/index.json",
            "backup_path": str(backup_path),
            "existed": True,
        }
    ]

    portable_preflight = migration._portable_preflight_evidence(
        repo_root, preflight
    )
    portable_snapshot = migration._portable_snapshot_evidence(
        repo_root, snapshot
    )

    assert portable_preflight["fvt101_completion_receipt"]["event_path"] == (
        "data/live/events.jsonl"
    )
    assert portable_snapshot[0]["backup_path"] == (
        "data/live/migrations/run/backup/index.json"
    )
    assert preflight["fvt101_completion_receipt"]["event_path"] == str(
        event_path
    )
    assert snapshot[0]["backup_path"] == str(backup_path)


def test_projection_refresh_failure_restores_every_snapshot(
    tmp_path, monkeypatch
):
    board_relative = Path("board.md")
    projection_relative = Path("projection.json")
    receipt_relative = Path("receipt.json")
    original_journal_relative = Path("state/original-journal.json")
    board_path = tmp_path / board_relative
    projection_path = tmp_path / projection_relative
    receipt_path = tmp_path / receipt_relative
    original_journal_path = tmp_path / original_journal_relative
    board_path.write_text(
        "## FVT-086 historical\n\n"
        "- Completion: superseded:FVT-101\n",
        encoding="utf-8",
    )
    projection_path.write_text('{"state":"good"}\n', encoding="utf-8")
    direct_dependents = {
        task_id: {
            "canonical_task_cid": migration.TASK_CIDS[task_id],
            "canonical_task_key": f"task/v1/{task_id}",
            "previous_depends_on": ["FVT-086"],
            "previous_work_contract_id": f"work/{task_id}",
            "previous_task_work_contract_id": f"task-work/{task_id}",
        }
        for task_id in migration.DIRECT_DEPENDENT_TASK_IDS
    }
    receipt = {
        "schema": migration.MIGRATION_SCHEMA,
        "migration_id": "migration-1",
        "journal_path": original_journal_relative.as_posix(),
        "direct_dependents": direct_dependents,
    }
    original_journal = {
        "schema": migration.JOURNAL_SCHEMA,
        "migration_id": "migration-1",
        "phase": "completed",
    }
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    original_journal_path.parent.mkdir(parents=True)
    original_journal_path.write_text(
        json.dumps(original_journal) + "\n", encoding="utf-8"
    )
    expected = {
        path: path.read_bytes()
        for path in (
            board_path,
            projection_path,
            receipt_path,
            original_journal_path,
        )
    }

    monkeypatch.setattr(migration, "BOARD_RELATIVE", board_relative)
    monkeypatch.setattr(migration, "RECEIPT_RELATIVE", receipt_relative)
    monkeypatch.setattr(
        migration,
        "SNAPSHOT_RELATIVES",
        (board_relative, projection_relative, receipt_relative),
    )
    monkeypatch.setattr(migration, "preflight", lambda _root: {"ok": True})
    monkeypatch.setattr(
        migration,
        "checkout_mutation_lock_path",
        lambda *_args, **_kwargs: tmp_path / "maintenance.lock",
    )

    class FakeLease:
        def __init__(self, **_kwargs):
            pass

        @contextmanager
        def exclusive_section(self, **_kwargs):
            yield {"within_bound": True}

    monkeypatch.setattr(migration, "CheckoutMaintenanceLease", FakeLease)

    def fail_after_mutation(_root):
        board_path.write_text("corrupt board\n", encoding="utf-8")
        projection_path.write_text("corrupt projection\n", encoding="utf-8")
        receipt_path.write_text("corrupt receipt\n", encoding="utf-8")
        original_journal_path.write_text(
            "corrupt journal\n", encoding="utf-8"
        )
        raise RuntimeError("injected projection failure")

    monkeypatch.setattr(migration, "_writer_pass", fail_after_mutation)

    with pytest.raises(RuntimeError, match="injected projection failure"):
        migration.refresh_migrated_projections(tmp_path)

    for path, content in expected.items():
        assert path.read_bytes() == content
    refresh_journals = list(
        original_journal_path.parent.glob(
            "fvt086-to-fvt101-projection-refresh-*/journal.json"
        )
    )
    assert len(refresh_journals) == 1
    assert json.loads(refresh_journals[0].read_text())["phase"] == "rolled_back"
