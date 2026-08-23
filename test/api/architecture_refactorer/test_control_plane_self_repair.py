"""Hermetic proof that database authority cannot drift into Markdown."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.authoritative_board_projection import (
    BoardProjectionRepairError,
    classify_projection_drift,
    repair_authoritative_board_projection,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    _repair_authoritative_board_projection_before_launch,
)

SEALED_BOARD = b"""# Tasks

## PCAR-000 Canonical zero

- Status: todo
- Completion: auto

## PCAR-001 Canonical one

- Status: todo
- Completion: auto
"""


def _generated_reconciliation(task_id: str = "PCAR-002") -> bytes:
    return f"""

## {task_id} Resolve generated reconciliation finding

- Status: blocked
- Completion: manual
- Is schedulable: false
- Review only: true
- Blocked reason: operator_reconciliation_required
- Priority: P1
- Track: ops
- Fingerprint: deadbeef
- Dedupe key: reconciliation_guardrail:preflight_merge_conflict
- Depends on:
- Outputs: evidence, docs/tasks.md
- Validation: test -f evidence/finding.json
- Acceptance: Reconciliation guardrail filed this because one candidate is blocked.
""".encode()


def _generated_retry(task_id: str = "PCAR-003") -> bytes:
    return f"""

## {task_id} Resolve generated retry finding

- Status: todo
- Completion: manual
- Priority: P1
- Track: ops
- Depends on: PCAR-000
- Outputs: evidence
- Validation: test -f evidence/retry.json
- Acceptance: Retry-budget guardrail filed this from repeated implementation failures.
""".encode()


def _git(repo: Path, *arguments: str) -> str:
    result = subprocess.run(
        ["git", *arguments],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _seed_repository(tmp_path: Path, *, trusted_subject: bool = True) -> tuple[Path, Path]:
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    board = repo / "docs/tasks.md"
    board.parent.mkdir()
    board.write_bytes(SEALED_BOARD)
    _git(repo, "add", "docs/tasks.md")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "seal board",
    )
    source_head = _git(repo, "rev-parse", "HEAD")

    receipt = repo / "runtime/bootstrap.json"
    receipt.parent.mkdir()
    receipt.write_text(
        json.dumps(
            {
                "source_head": source_head,
                "source_identities": {
                    "taskboard": "sha256:" + hashlib.sha256(SEALED_BOARD).hexdigest()
                },
            }
        ),
        encoding="utf-8",
    )
    config = repo / "config.json"
    config.write_text(
        json.dumps(
            {
                "taskboard_path": "docs/tasks.md",
                "task_prefix": "PCAR-",
                "merge_target_branch": "main",
                "operational_control_plane": {"markdown_is_authority": False},
                "authoritative_board_projection_repair": {
                    "mode": "sealed_bootstrap_projection",
                    "automatic_repair_before_launch": True,
                    "allowed_drift": "supervisor_generated_guardrail_suffix_only",
                    "bootstrap_receipt_path": "runtime/bootstrap.json",
                    "repair_receipt_path": "runtime/repair.json",
                    "canonical_block_mutation_permitted": False,
                    "markdown_task_mutation_permitted": False,
                },
            }
        ),
        encoding="utf-8",
    )
    _git(repo, "add", "config.json", "runtime/bootstrap.json")
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        "configure authority",
    )
    board.write_bytes(SEALED_BOARD + _generated_reconciliation())
    _git(repo, "add", "docs/tasks.md")
    subject = (
        "Agent: generated guardrail [agent-supervisor:generated-protected-board]"
        if trusted_subject
        else "untrusted board edit"
    )
    _git(
        repo,
        "-c",
        "user.name=Test",
        "-c",
        "user.email=test@example.invalid",
        "commit",
        "-m",
        subject,
    )
    return repo, config


def test_classifier_accepts_only_noncanonical_generated_suffixes() -> None:
    observed = classify_projection_drift(
        sealed_board=SEALED_BOARD,
        current_board=(
            SEALED_BOARD + _generated_reconciliation() + _generated_retry()
        ),
        canonical_task_ids=["PCAR-000", "PCAR-001"],
        task_prefix="PCAR-",
        taskboard_relative="docs/tasks.md",
    )

    assert observed["drift"] is True
    assert observed["generated_task_ids"] == ["PCAR-002", "PCAR-003"]


def test_classifier_rejects_canonical_or_ambiguous_changes() -> None:
    with pytest.raises(BoardProjectionRepairError, match="canonical task"):
        classify_projection_drift(
            sealed_board=SEALED_BOARD,
            current_board=SEALED_BOARD + _generated_reconciliation("PCAR-001"),
            canonical_task_ids=["PCAR-000", "PCAR-001"],
            task_prefix="PCAR-",
            taskboard_relative="docs/tasks.md",
        )
    with pytest.raises(BoardProjectionRepairError, match="sealed bootstrap bytes"):
        classify_projection_drift(
            sealed_board=SEALED_BOARD,
            current_board=SEALED_BOARD.replace(b"todo", b"blocked", 1),
            canonical_task_ids=["PCAR-000", "PCAR-001"],
            task_prefix="PCAR-",
            taskboard_relative="docs/tasks.md",
        )


def test_repair_restores_exact_receipt_bytes_and_preserves_database_blocks(
    tmp_path: Path,
) -> None:
    repo, config = _seed_repository(tmp_path)
    starting_head = _git(repo, "rev-parse", "HEAD")

    report = repair_authoritative_board_projection(
        config,
        repo_root=repo,
        canonical_snapshot_provider=lambda: {
            "task_ids": ["PCAR-000", "PCAR-001"],
            "blocked_task_ids": ["PCAR-001"],
        },
    )

    assert report["repaired"] is True
    assert report["starting_head"] == starting_head
    assert report["removed_generated_task_ids"] == ["PCAR-002"]
    assert report["canonical_blocked_task_ids"] == ["PCAR-001"]
    assert report["canonical_blocks_mutated"] is False
    assert (repo / "docs/tasks.md").read_bytes() == SEALED_BOARD
    assert report["repair_commit"] == _git(repo, "rev-parse", "HEAD")
    assert "[agent-supervisor:generated-protected-board]" in _git(
        repo, "show", "-s", "--format=%s", "HEAD"
    )
    receipt = json.loads((repo / "runtime/repair.json").read_text(encoding="utf-8"))
    assert receipt["repair_commit"] == report["repair_commit"]

    replay = repair_authoritative_board_projection(
        config,
        repo_root=repo,
        canonical_snapshot_provider=lambda: {
            "task_ids": ["PCAR-000", "PCAR-001"],
            "blocked_task_ids": ["PCAR-001"],
        },
    )
    assert replay["repaired"] is False
    assert replay["reason_code"] == "projection_current"
    assert replay["canonical_blocked_task_ids"] == ["PCAR-001"]
    assert json.loads(
        (repo / "runtime/repair.json").read_text(encoding="utf-8")
    )["reason_code"] == "projection_current"
    assert _git(repo, "rev-parse", "HEAD") == report["repair_commit"]


def test_current_projection_still_requires_exact_live_canonical_identity(
    tmp_path: Path,
) -> None:
    repo, config = _seed_repository(tmp_path)
    repair_authoritative_board_projection(
        config,
        repo_root=repo,
        canonical_snapshot_provider=lambda: {
            "task_ids": ["PCAR-000", "PCAR-001"],
            "blocked_task_ids": [],
        },
    )

    with pytest.raises(BoardProjectionRepairError, match="canonical task identities"):
        repair_authoritative_board_projection(
            config,
            repo_root=repo,
            canonical_snapshot_provider=lambda: {
                "task_ids": ["PCAR-000", "PCAR-001", "PCAR-002"],
                "blocked_task_ids": ["PCAR-002"],
            },
        )


def test_repair_rejects_untrusted_commit_history(tmp_path: Path) -> None:
    repo, config = _seed_repository(tmp_path, trusted_subject=False)

    with pytest.raises(BoardProjectionRepairError, match="trusted supervisor"):
        repair_authoritative_board_projection(
            config,
            repo_root=repo,
            canonical_snapshot_provider=lambda: {
                "task_ids": ["PCAR-000", "PCAR-001"],
                "blocked_task_ids": [],
            },
        )

    assert (repo / "docs/tasks.md").read_bytes() != SEALED_BOARD


def test_scheduler_repairs_only_before_effectful_launch(monkeypatch, tmp_path: Path) -> None:
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        authoritative_board_projection as projection_module,
    )

    calls: list[tuple[Path, Path]] = []

    def repair(config_path: Path, *, repo_root: Path):
        calls.append((config_path, repo_root))
        return {"enabled": True, "repaired": True}

    monkeypatch.setattr(
        projection_module,
        "repair_authoritative_board_projection",
        repair,
    )
    config = tmp_path / "config.json"

    read_only = _repair_authoritative_board_projection_before_launch(
        config_path=config,
        repo_root=tmp_path,
        command="preflight",
        dry_run=False,
    )
    dry_run = _repair_authoritative_board_projection_before_launch(
        config_path=config,
        repo_root=tmp_path,
        command="launch",
        dry_run=True,
    )
    launch = _repair_authoritative_board_projection_before_launch(
        config_path=config,
        repo_root=tmp_path,
        command="launch",
        dry_run=False,
    )

    assert read_only["reason_code"] == "read_only_command"
    assert dry_run["reason_code"] == "read_only_command"
    assert launch == {"enabled": True, "repaired": True}
    assert calls == [(config, tmp_path)]
