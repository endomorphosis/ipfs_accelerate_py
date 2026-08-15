from __future__ import annotations

import json
import os
from dataclasses import replace

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import (
    persistent_task_queue as queue_module,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.task_identity import (
    canonical_bundle_identity,
    canonical_task_identity,
)
from ipfs_accelerate_py.agent_supervisor.task_sources.persistent_task_queue import PersistentTaskQueue
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTaskState,
    parse_task_file,
)


def _task(task_id: str) -> dict[str, object]:
    return {
        "task_id": task_id,
        "title": "Add a durable task ledger",
        "outputs": ["src/ledger.py", "tests/test_ledger.py"],
        "acceptance": "Retries and receipts retain canonical identity.",
        "metadata": {"goal id": "G9.S1"},
    }


def _write_allowed_path_board(path) -> None:
    path.write_text(
        """# Tasks

## REF-001 Add a durable task ledger

- Status: todo
- Priority: P0
- Track: agent
- Outputs: src/ledger.py, tests/test_ledger.py
- Allowed paths: src/ledger.py, tests/test_ledger.py
- Acceptance: Retries and receipts retain canonical identity.
- Goal id: G9.S1
""",
        encoding="utf-8",
    )


def _identity_daemon(todo_path, tmp_path) -> PortalImplementationDaemon:
    return PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## REF-",
    )


def _write_governed_board(path) -> None:
    command_set = json.dumps(
        {
            "commands": [
                {
                    "argv": ["python", "-m", "pytest", "tests/test_inventory.py", "-q"],
                    "cwd": ".",
                    "env": {},
                    "id": "inventory-smoke",
                    "repository": "control",
                    "repository_root": ".",
                    "timeout_seconds": 120,
                }
            ],
            "schema": (
                "ipfs_accelerate_py/agent-supervisor/"
                "governed-phase-command-set@1"
            ),
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    path.write_text(
        f"""# Governed tasks

## REF-010 Inventory governed runtime authority

- Status: todo
- Priority: P0
- Track: inventory
- Outputs: artifacts/inventory.json, artifacts/receipts/REF-010.json
- Provider effects: artifacts/inventory.json
- Supervisor outputs: artifacts/receipts/REF-010.json
- Objective: Inspect canonical runtime implementations.
- Acceptance criteria: Inventory rows cite exact code and tests.
- Required evidence: inventory artifact identity
- Required tests: inventory-smoke
- Rollback procedure: Revert the exact candidate commit.
- Provider role: grok-implement, codex-review
- Pre-change validation: {command_set}
- Pre-change validation policy: require-pass
- Post-change validation: {command_set}
- Acceptance: Inventory evidence is retained.
""",
        encoding="utf-8",
    )


def test_governed_v2_identity_is_parse_to_daemon_idempotent(tmp_path) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_governed_board(todo_path)
    [task] = parse_task_file(todo_path, "## REF-")
    daemon = _identity_daemon(todo_path, tmp_path)

    runtime_identity = daemon._identity_for_task(task)

    assert task.canonical_task_key.startswith("task/v2/")
    assert runtime_identity.to_dict() == canonical_task_identity(
        task,
        board_namespace=task.board_namespace,
        source_path=todo_path,
    ).to_dict()
    assert runtime_identity.canonical_task_key == task.canonical_task_key
    assert runtime_identity.canonical_task_cid == task.canonical_task_cid
    assert runtime_identity.task_intent_cid


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("Objective", "Trust an unverified model assertion."),
        ("Provider effects", "artifacts/widened.json"),
        ("Rollback procedure", "No rollback is required."),
        ("Pre-change validation policy", "record-baseline"),
    ],
)
def test_governed_v2_rejects_stale_supplied_identity_after_intent_change(
    tmp_path,
    field: str,
    replacement: str,
) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_governed_board(todo_path)
    [task] = parse_task_file(todo_path, "## REF-")
    daemon = _identity_daemon(todo_path, tmp_path)
    stale_claim = replace(
        task,
        metadata={**task.metadata, field: replacement},
    )

    with pytest.raises(ValueError, match="does not bind current intent"):
        daemon._identity_for_task(stale_claim)


def test_daemon_rejects_forged_v2_native_key_cid_pair(tmp_path) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_governed_board(todo_path)
    [task] = parse_task_file(todo_path, "## REF-")
    daemon = _identity_daemon(todo_path, tmp_path)
    other = canonical_task_identity(
        {
            **_task("REF-OTHER"),
            "metadata": {
                "Provider role": "grok-implement, codex-review",
                "Objective": "A different governed objective.",
            },
        }
    )
    assert other.canonical_task_key.startswith("task/v2/")
    forged = replace(task, canonical_task_cid=other.canonical_task_cid)

    with pytest.raises(ValueError, match="key/CID claim is inconsistent"):
        daemon._identity_for_task(forged)


def test_parser_identity_is_daemon_idempotent_with_allowed_paths(tmp_path) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_allowed_path_board(todo_path)
    [task] = parse_task_file(todo_path, "## REF-")
    daemon = _identity_daemon(todo_path, tmp_path)

    runtime_identity = daemon._identity_for_task(task)

    assert runtime_identity.canonical_task_key == task.canonical_task_key
    assert runtime_identity.canonical_task_cid == task.canonical_task_cid


def test_allowed_path_authority_change_readdresses_runtime_identity(tmp_path) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_allowed_path_board(todo_path)
    [task] = parse_task_file(todo_path, "## REF-")
    daemon = _identity_daemon(todo_path, tmp_path)
    widened = replace(
        task,
        metadata={
            **task.metadata,
            "allowed paths": (
                "src/ledger.py, tests/test_ledger.py, src/new-authority.py"
            ),
        },
    )

    widened_identity = daemon._identity_for_task(widened)

    assert widened_identity.canonical_task_key != task.canonical_task_key
    assert widened_identity.canonical_task_cid != task.canonical_task_cid


def test_daemon_rejects_forged_native_provided_identity(tmp_path) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_allowed_path_board(todo_path)
    [task] = parse_task_file(todo_path, "## REF-")
    daemon = _identity_daemon(todo_path, tmp_path)
    other = canonical_task_identity(
        {
            "task_id": "REF-999",
            "title": "Different task",
            "outputs": ["src/different.py"],
        }
    )
    forged = replace(task, canonical_task_cid=other.canonical_task_cid)

    with pytest.raises(ValueError, match="key/CID claim is inconsistent"):
        daemon._identity_for_task(forged)


def test_canonical_identity_rejects_forged_native_key_cid_pair() -> None:
    first = canonical_task_identity(_task("REF-001"))
    second_task = _task("REF-002")
    second_task["title"] = "Different task"
    second = canonical_task_identity(second_task)

    with pytest.raises(ValueError, match="key/CID claim is inconsistent"):
        canonical_task_identity(
            {
                **_task("REF-FORGED"),
                "canonical_task_key": first.canonical_task_key,
                "canonical_task_cid": second.canonical_task_cid,
                "metadata": {"allowed paths": "src/ledger.py"},
            }
        )


def test_task_identity_is_independent_of_display_id_and_board_path() -> None:
    first = canonical_task_identity(
        _task("REF-001"),
        board_namespace="board-a",
        source_path="data/board-a.todo.md",
    )
    second = canonical_task_identity(
        _task("LOCAL-987"),
        board_namespace="board-b",
        source_path="worktrees/attempt/data/board-b.todo.md",
    )

    assert first.canonical_task_key == second.canonical_task_key
    assert first.canonical_task_cid == second.canonical_task_cid
    assert first.namespaced_alias == "board-a::REF-001"
    assert second.namespaced_alias == "board-b::LOCAL-987"


def test_task_identity_changes_when_semantic_acceptance_changes() -> None:
    first = _task("REF-001")
    second = _task("REF-001")
    second["acceptance"] = "A different implementation contract."

    assert canonical_task_identity(first).canonical_task_cid != canonical_task_identity(second).canonical_task_cid


def test_task_identity_binds_explicit_evidence_outputs() -> None:
    first = _task("REF-001")
    second = _task("REF-001")
    first["metadata"] = {
        **first["metadata"],
        "evidence outputs": "data/manifests/coverage.json",
    }
    second["metadata"] = {
        **second["metadata"],
        "evidence outputs": "data/manifests/repository-root.json",
    }

    assert (
        canonical_task_identity(first).canonical_task_cid
        != canonical_task_identity(second).canonical_task_cid
    )


def test_explicit_dedupe_key_migrates_legacy_aliases_idempotently() -> None:
    first = _task("REF-001")
    second = _task("OTHER-002")
    first["metadata"] = {"dedupe key": "supervisor:durable-ledger"}
    second["metadata"] = {"dedupe key": "supervisor:durable-ledger"}

    assert canonical_task_identity(first).canonical_task_cid == canonical_task_identity(second).canonical_task_cid


def test_bundle_identity_is_stable_across_bundle_and_display_names() -> None:
    first = {
        "bundle_key": "objective/g9/one",
        "source_todo": "one.todo.md",
        "tasks": [_task("REF-001")],
    }
    second = {
        "bundle_key": "objective/g9/two",
        "source_todo": "two.todo.md",
        "tasks": [_task("LOCAL-009")],
    }

    assert canonical_bundle_identity(first).canonical_task_cid == canonical_bundle_identity(second).canonical_task_cid


def test_bundle_identity_preserves_metadata_poor_task_cardinality() -> None:
    single = {"bundle_key": "objective/refactor", "tasks": [{"task_id": "ONE"}]}
    pair = {"bundle_key": "objective/refactor", "tasks": [{"task_id": "ONE"}, {"task_id": "TWO"}]}

    assert canonical_bundle_identity(single).canonical_task_cid != canonical_bundle_identity(pair).canonical_task_cid


def test_bundle_identity_uses_nonempty_execution_slice() -> None:
    first_task = _task("FIRST")
    second_task = {
        **_task("SECOND"),
        "title": "Add dependency-aware task scheduling",
        "outputs": ["src/scheduler.py", "tests/test_scheduler.py"],
    }
    full_bundle = {
        "bundle_key": "objective/g9",
        "tasks": [first_task, second_task],
    }
    first_slice = {
        **full_bundle,
        "execution_slice_task_ids": ["FIRST"],
    }
    second_cid = canonical_task_identity(second_task).canonical_task_cid
    second_slice = {
        **full_bundle,
        "execution_slice_task_cids": [second_cid],
    }

    first_identity = canonical_bundle_identity(first_slice).canonical_task_cid
    second_identity = canonical_bundle_identity(second_slice).canonical_task_cid

    assert first_identity != second_identity
    assert first_identity == canonical_bundle_identity(
        {"bundle_key": "objective/g9", "tasks": [first_task]}
    ).canonical_task_cid
    assert second_identity == canonical_bundle_identity(
        {"bundle_key": "objective/g9", "tasks": [second_task]}
    ).canonical_task_cid
    assert canonical_bundle_identity(
        {**full_bundle, "execution_slice_task_ids": []}
    ).canonical_task_cid == canonical_bundle_identity(full_bundle).canonical_task_cid


def test_bundle_slice_identity_ignores_member_status_changes() -> None:
    task = _task("FIRST")
    bundle = {
        "bundle_key": "objective/g9",
        "tasks": [task, {**_task("SECOND"), "title": "A deferred task"}],
        "execution_slice_task_ids": ["FIRST"],
    }
    changed = {
        **bundle,
        "tasks": [
            {**task, "status": "completed"},
            {**bundle["tasks"][1], "status": "blocked"},
        ],
    }

    assert canonical_bundle_identity(bundle).canonical_task_cid == canonical_bundle_identity(
        changed
    ).canonical_task_cid


def test_provided_identity_uses_git_safe_execution_fingerprint() -> None:
    identity = canonical_task_identity(
        {
            "task_id": "TASK-001",
            "canonical_task_key": "external:key/with:git-invalid-characters",
            "canonical_task_cid": "bafyexternalidentity",
        }
    )

    assert len(identity.semantic_fingerprint) == 64
    assert set(identity.semantic_fingerprint) <= set("0123456789abcdef")


def test_persistent_queue_migrates_legacy_display_id_to_canonical_identity(tmp_path) -> None:
    path = tmp_path / "task_queue.json"
    path.write_text(
        """{
  "schema": "persistent_task_queue_v1",
  "entries": {
    "REF-001": {"task_id": "REF-001", "attempt_count": 2, "selection_penalty": 100}
  }
}
""",
        encoding="utf-8",
    )
    identity = canonical_task_identity(
        _task("REF-001"),
        board_namespace="main",
        source_path="tasks.todo.md",
    )

    queue = PersistentTaskQueue.load(path, save_interval=0)
    first = queue.register_task(identity, priority="P0", track="agent")
    second = queue.register_task(identity, priority="P0", track="agent")
    queue.save()
    restored = PersistentTaskQueue.load(path)

    assert first is second
    assert list(queue.entries) == [identity.canonical_task_cid]
    assert first.attempt_count == 2
    assert first.selection_penalty == 100
    assert first.provenance == [
        {
            "board_namespace": "main",
            "display_task_id": "REF-001",
            "source_path": "tasks.todo.md",
        }
    ]
    assert restored.resolve_key("main::REF-001") == identity.canonical_task_cid

    restored.register_task(identity, priority="P0", track="agent")
    assert restored.dirty is False


def test_persistent_queue_selection_preserves_registered_scheduling_metadata(tmp_path) -> None:
    queue = PersistentTaskQueue.load(tmp_path / "task_queue.json", save_interval=0)
    identity = canonical_task_identity(
        _task("REF-001"),
        board_namespace="main",
        source_path="tasks.todo.md",
    )
    entry = queue.register_task(identity, priority="P0", track="foundation")

    queue.record_selection(identity.canonical_task_cid)

    assert entry.priority == "P0"
    assert entry.track == "foundation"
    assert entry.attempt_count == 1


def test_authority_renewal_retry_state_is_durable_bounded_and_content_scoped(
    tmp_path,
    monkeypatch,
) -> None:
    path = tmp_path / "task_queue.json"
    clock = {"now": 1000.0}
    monkeypatch.setattr(queue_module.time, "time", lambda: clock["now"])
    original = canonical_task_identity(
        _task("REF-001"),
        board_namespace="main",
        source_path="tasks.todo.md",
    )
    revised_task = _task("REF-001")
    revised_task["acceptance"] = "A revised authority contract."
    revised = canonical_task_identity(
        revised_task,
        board_namespace="main",
        source_path="tasks.todo.md",
    )
    task_cid = original.canonical_task_cid
    renewal_key = "renewal-key-generation-a"

    # Seed every ordinary retry dimension so renewal failures must preserve
    # non-zero values rather than merely leaving zero-valued defaults alone.
    queue = PersistentTaskQueue.load(path)
    queue.register_task(original, priority="P0", track="authority")
    queue.record_selection(task_cid)
    queue.record_failure(task_cid, "ordinary implementation failure")
    queue.record_no_change(task_cid)
    queue.record_merge_failure(task_cid)
    queue.save()
    ordinary_retry_state = {
        field: getattr(queue.entries[task_cid], field)
        for field in (
            "attempt_count",
            "selection_penalty",
            "consecutive_failures",
            "consecutive_no_change",
            "merge_failure_count",
            "cooldown_until",
            "notes",
        )
    }

    for strike in range(1, 4):
        recorded = queue.record_authority_renewal_failure(
            task_cid,
            renewal_key,
            reason=f"renewal failure {strike}",
            max_failures=3,
            base_cooldown_seconds=10,
            max_cooldown_seconds=100,
        )

        # Reload after every strike. In particular, strike two occurs less
        # than the default 30-second save interval after strike one.
        queue = PersistentTaskQueue.load(path)
        restored = queue.authority_renewal_state(task_cid, renewal_key)
        assert restored == recorded
        assert restored["failure_count"] == strike
        assert restored["last_failure_at"] == clock["now"]
        assert restored["cooldown_until"] > clock["now"]
        assert restored["cooled_down"] is True
        assert restored["quarantined"] is (strike == 3)
        assert restored["requires_operator_reset"] is (strike == 3)
        assert restored["reason"] == f"renewal failure {strike}"
        assert {
            field: getattr(queue.entries[task_cid], field)
            for field in ordinary_retry_state
        } == ordinary_retry_state

        if strike < 3:
            clock["now"] = restored["cooldown_until"] + 1

    clock["now"] = restored["cooldown_until"] + 1
    queue = PersistentTaskQueue.load(path)
    expired = queue.authority_renewal_state(task_cid, renewal_key)
    assert expired["cooled_down"] is False
    assert expired["quarantined"] is True
    assert expired["requires_operator_reset"] is True

    changed_key = "renewal-key-generation-b"
    assert queue.authority_renewal_state(task_cid, changed_key) == {
        "renewal_key": changed_key,
        "failure_count": 0,
        "last_failure_at": 0.0,
        "cooldown_until": 0.0,
        "cooled_down": False,
        "quarantined": False,
        "requires_operator_reset": False,
        "reason": "",
    }
    queue.record_authority_renewal_failure(
        task_cid,
        changed_key,
        reason="new authority generation",
        max_failures=3,
        base_cooldown_seconds=10,
        max_cooldown_seconds=100,
    )
    queue = PersistentTaskQueue.load(path)
    changed_key_state = queue.authority_renewal_state(task_cid, changed_key)
    assert changed_key_state["failure_count"] == 1
    assert changed_key_state["quarantined"] is False
    assert queue.authority_renewal_state(task_cid, renewal_key)["failure_count"] == 0
    assert {
        field: getattr(queue.entries[task_cid], field)
        for field in ordinary_retry_state
    } == ordinary_retry_state

    assert revised.canonical_task_cid != task_cid
    queue.register_task(revised, priority="P0", track="authority")
    assert queue.authority_renewal_state(revised.canonical_task_cid, renewal_key)[
        "failure_count"
    ] == 0
    queue.record_authority_renewal_failure(
        revised.canonical_task_cid,
        renewal_key,
        reason="revised task failure",
        max_failures=3,
        base_cooldown_seconds=10,
        max_cooldown_seconds=100,
    )
    queue = PersistentTaskQueue.load(path)
    revised_state = queue.authority_renewal_state(
        revised.canonical_task_cid,
        renewal_key,
    )
    assert revised_state["failure_count"] == 1
    assert revised_state["quarantined"] is False
    revised_entry = queue.entries[revised.canonical_task_cid]
    assert revised_entry.attempt_count == 0
    assert revised_entry.consecutive_failures == 0
    assert revised_entry.consecutive_no_change == 0
    assert revised_entry.merge_failure_count == 0
    assert revised_entry.selection_penalty == 0
    assert revised_entry.cooldown_until == 0.0


def test_legacy_markdown_pending_status_normalizes_to_todo(tmp_path) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    todo_path.write_text(
        """# Tasks

## REF-001 Ready under the common pending spelling

- Status: pending
- Priority: P0
- Track: foundation
""",
        encoding="utf-8",
    )

    [task] = parse_task_file(todo_path, "## REF-")

    assert task.status == "todo"


def test_persistent_queue_recovers_from_malformed_numeric_state(tmp_path) -> None:
    path = tmp_path / "task_queue.json"
    path.write_text(
        json.dumps(
            {
                "schema": "persistent_task_queue_v1",
                "entries": {"TASK-001": {"task_id": "TASK-001", "attempt_count": "invalid"}},
            }
        ),
        encoding="utf-8",
    )

    queue = PersistentTaskQueue.load(path)

    assert queue.entries == {}


def test_persistent_queue_coalesces_two_board_aliases_for_same_work(tmp_path) -> None:
    queue = PersistentTaskQueue.load(tmp_path / "task_queue.json", save_interval=0)
    first = canonical_task_identity(_task("REF-001"), board_namespace="main")
    second = canonical_task_identity(_task("LOCAL-009"), board_namespace="bundle")

    queue.register_task(first).record_selection()
    queue.register_task(second)

    assert len(queue.entries) == 1
    assert queue.get_penalty(first.canonical_task_cid) == queue.get_penalty(second.canonical_task_cid)
    assert queue.entries[first.canonical_task_cid].attempt_count == 1
    assert queue.resolve_key("main::REF-001") == queue.resolve_key("bundle::LOCAL-009")


def test_persistent_queue_keeps_reused_display_ids_separate_across_boards(tmp_path) -> None:
    queue = PersistentTaskQueue.load(tmp_path / "task_queue.json", save_interval=0)
    first = canonical_task_identity(
        {"task_id": "TASK-001", "title": "Refactor parser", "outputs": ["parser.py"]},
        board_namespace="backend",
    )
    second = canonical_task_identity(
        {"task_id": "TASK-001", "title": "Refactor dashboard", "outputs": ["dashboard.ts"]},
        board_namespace="frontend",
    )

    queue.register_task(first)
    queue.register_task(second)

    assert set(queue.entries) == {first.canonical_task_cid, second.canonical_task_cid}
    assert queue.resolve_key(first.namespaced_alias) == first.canonical_task_cid
    assert queue.resolve_key(second.namespaced_alias) == second.canonical_task_cid


def test_persistent_queue_resets_history_when_task_semantics_change(tmp_path) -> None:
    queue = PersistentTaskQueue.load(tmp_path / "task_queue.json", save_interval=0)
    original = canonical_task_identity(
        {"task_id": "TASK-001", "title": "Refactor parser", "acceptance": "Keep API stable."},
        board_namespace="main",
    )
    replacement = canonical_task_identity(
        {"task_id": "TASK-001", "title": "Refactor parser", "acceptance": "Permit API changes."},
        board_namespace="main",
    )

    queue.register_task(original).record_failure("old failure")
    replacement_entry = queue.register_task(replacement)

    assert replacement_entry.consecutive_failures == 0
    assert set(queue.entries) == {original.canonical_task_cid, replacement.canonical_task_cid}
    assert queue.resolve_key(replacement.namespaced_alias) == replacement.canonical_task_cid
    assert queue.resolve_key(replacement.display_task_id) == replacement.canonical_task_cid


def _write_duplicate_board(path) -> None:
    path.write_text(
        """# Tasks

## REF-001 Add a durable task ledger

- Status: todo
- Priority: P0
- Track: agent
- Outputs: src/ledger.py, tests/test_ledger.py
- Acceptance: Retries and receipts retain canonical identity.
- Goal id: G9.S1

## REF-009 Add a durable task ledger

- Status: todo
- Priority: P0
- Track: agent
- Outputs: src/ledger.py, tests/test_ledger.py
- Acceptance: Retries and receipts retain canonical identity.
- Goal id: G9.S1
""",
        encoding="utf-8",
    )


def test_implementation_daemon_coalesces_duplicate_work_before_selection(tmp_path) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_duplicate_board(todo_path)
    state_path = tmp_path / "state" / "task_state.json"
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=state_path,
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## REF-",
    )

    result = daemon.run_once()
    state = PortalTaskState.load(state_path)

    assert result["task_count"] == 2
    assert result["canonical_task_count"] == 1
    assert result["selectable_ready_count"] == 1
    assert state.active_task_cid
    assert state.task_identities["REF-001"]["canonical_task_cid"] == state.task_identities["REF-009"][
        "canonical_task_cid"
    ]
    events = [json.loads(line) for line in (tmp_path / "state" / "events.jsonl").read_text().splitlines()]
    selected = next(event for event in events if event["type"] == "task_selected")
    assert selected["canonical_task_cid"] == state.active_task_cid
    queue = PersistentTaskQueue.load(tmp_path / "state" / "task_queue.json")
    assert len(queue.entries) == 1
    [entry] = queue.entries.values()
    assert entry.priority == "P0"
    assert entry.attempt_count == 1


def test_claim_lock_and_retry_history_follow_canonical_identity_across_aliases(tmp_path) -> None:
    first_path = tmp_path / "first.todo.md"
    second_path = tmp_path / "second.todo.md"
    _write_duplicate_board(first_path)
    second_path.write_text(
        first_path.read_text(encoding="utf-8").replace("REF-001", "OTHER-777"),
        encoding="utf-8",
    )
    first = parse_task_file(first_path, "## REF-")[0]
    second = parse_task_file(second_path, "## OTHER-")[0]
    daemon = PortalImplementationDaemon(
        todo_path=first_path,
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## REF-",
    )
    state = PortalTaskState(implementation_attempts={"REF-001": 3})

    assert first.canonical_task_cid == second.canonical_task_cid
    first_lock = daemon._implementation_task_claim_path(
        first.task_id,
        canonical_task_cid=first.canonical_task_cid,
    )
    second_lock = daemon._implementation_task_claim_path(
        second.task_id,
        canonical_task_cid=second.canonical_task_cid,
    )
    assert first_lock == second_lock
    assert daemon._task_attempt(state, first) == 4
    daemon._record_task_attempt(state, first, 4)
    assert daemon._task_attempt(state, second) == 5


def test_legacy_claim_lock_blocks_every_canonical_alias(tmp_path) -> None:
    todo_path = tmp_path / "tasks.todo.md"
    _write_duplicate_board(todo_path)
    tasks = parse_task_file(todo_path, "## REF-")
    daemon = PortalImplementationDaemon(
        todo_path=todo_path,
        state_path=tmp_path / "state" / "task_state.json",
        strategy_path=tmp_path / "state" / "strategy.json",
        events_path=tmp_path / "state" / "events.jsonl",
        repo_root=tmp_path,
        task_header_prefix="## REF-",
    )
    claim_path = daemon._implementation_task_claim_path(tasks[0].task_id)
    claim_path.parent.mkdir(parents=True, exist_ok=True)
    claim_path.write_text(
        json.dumps(
            {
                "kind": "implementation_task_claim",
                "pid": os.getpid(),
                "repo_root": str(tmp_path.resolve()),
                "task_id": tasks[0].task_id,
            }
        ),
        encoding="utf-8",
    )

    claims = daemon._active_implementation_task_claims(tasks)

    assert set(claims) == {"REF-001", "REF-009"}
