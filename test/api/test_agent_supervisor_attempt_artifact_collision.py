from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTask,
    TodoImplementationDaemon,
)


def _daemon_and_task(tmp_path: Path) -> tuple[TodoImplementationDaemon, PortalTask]:
    repo = tmp_path / "repo"
    repo.mkdir()
    todo_path = repo / "todo.md"
    todo_path.write_text("# Todos\n", encoding="utf-8")
    state_dir = repo / "state"
    daemon = TodoImplementationDaemon(
        todo_path=todo_path,
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix="## REV-",
    )
    task = PortalTask(
        task_id="REV-001",
        title="Preserve the first attempt revision",
        status="ready",
        completion="manual",
        priority="P0",
        track="runtime",
        outputs=["src/revision.py"],
        validation=["test -f src/revision.py"],
        acceptance="Preserve revision one.",
    )
    return daemon, task


def test_same_attempt_task_revision_preserves_prior_log_and_context_receipt(
    tmp_path: Path,
) -> None:
    daemon, first_task = _daemon_and_task(tmp_path)
    attempt = 1
    log_dir = daemon.implementation_log_dir
    legacy_log = log_dir / "rev-001-attempt-1.log"

    assert daemon._implementation_attempt_artifact_path(
        first_task,
        legacy_log,
    ) == legacy_log
    log_dir.mkdir(parents=True, exist_ok=True)
    first_log_bytes = b"first task revision log\n"
    legacy_log.write_bytes(first_log_bytes)

    daemon._compile_implementation_context(first_task, attempt=attempt)
    first_receipt = daemon._persist_implementation_context_receipt(
        first_task,
        attempt=attempt,
    )
    first_receipt_bytes = first_receipt.read_bytes()
    assert first_receipt.name == (
        "rev-001-attempt-1-context-receipt.json"
    )

    revised_task = replace(
        first_task,
        title="Preserve a materially revised attempt",
        acceptance="Preserve revision two without erasing revision one.",
    )
    revised_suffix = daemon._identity_for_task(revised_task).short_id
    assert revised_suffix != daemon._identity_for_task(first_task).short_id

    revised_log = daemon._implementation_attempt_artifact_path(
        revised_task,
        legacy_log,
    )
    assert revised_log.name == f"rev-001-attempt-1-{revised_suffix}.log"
    revised_log.write_bytes(b"second task revision log\n")

    daemon._compile_implementation_context(revised_task, attempt=attempt)
    revised_receipt = daemon._persist_implementation_context_receipt(
        revised_task,
        attempt=attempt,
    )
    assert revised_receipt.name == (
        "rev-001-attempt-1-context-receipt-"
        f"{revised_suffix}.json"
    )
    assert revised_receipt != first_receipt
    assert legacy_log.read_bytes() == first_log_bytes
    assert first_receipt.read_bytes() == first_receipt_bytes
    # Base context files remain the unsuffixed latest projection. Only the
    # immutable attempt-owned receipt gains the canonical revision suffix.
    base_receipt = log_dir / "rev-001-base-context-receipt.json"
    assert json.loads(base_receipt.read_text(encoding="utf-8")) == json.loads(
        revised_receipt.read_text(encoding="utf-8")
    )
    assert not (
        log_dir / f"rev-001-base-context-receipt-{revised_suffix}.json"
    ).exists()

    with pytest.raises(FileExistsError, match="attempt artifact already exists"):
        daemon._implementation_attempt_artifact_path(
            revised_task,
            legacy_log,
        )
    with pytest.raises(FileExistsError, match="attempt artifact already exists"):
        daemon._persist_implementation_context_receipt(
            revised_task,
            attempt=attempt,
        )

    assert legacy_log.read_bytes() == first_log_bytes
    assert first_receipt.read_bytes() == first_receipt_bytes
