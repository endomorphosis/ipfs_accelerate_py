from __future__ import annotations

import json
import os

from ipfs_accelerate_py.agent_supervisor.task_identity import (
    canonical_bundle_identity,
    canonical_task_identity,
)
from ipfs_accelerate_py.agent_supervisor.persistent_task_queue import PersistentTaskQueue
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
