"""A dead unpublished dispatch intent can be released without task authority."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (
    current_process_birth,
    read_process_birth,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    IMPLEMENTATION_DISPATCH_INTENT_LEASE_ROLE,
    IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
    PortalImplementationDaemon,
    PortalTaskState,
)


CID = "baguqeeratestciddeadintent000000000000000000000000000000"
KEY = "task/v1/" + "ab" * 32
NAMESPACE = "complaint_repository_reorganization_task_board.md"
TASK_ID = "CIMP-MOD-02"


def _isolate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR",
        str(tmp_path / "orchestration"),
    )
    for name in (
        "IPFS_ACCELERATE_AGENT_QUACK_ENDPOINT",
        "IPFS_ACCELERATE_AGENT_CONTROL_DUCKDB",
        "IPFS_ACCELERATE_AGENT_CONTROL_DUCKLAKE",
    ):
        monkeypatch.delenv(name, raising=False)


def _daemon(tmp_path: Path) -> PortalImplementationDaemon:
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / ".git").mkdir()
    worktree_root = tmp_path / "worktrees"
    worktree_root.mkdir()
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    return PortalImplementationDaemon(
        todo_path=tmp_path / "todo.md",
        state_path=state_dir / "task_state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        implementation_log_dir=state_dir / "logs",
        repo_root=repo,
        worktree_root=worktree_root,
        implement=True,
        implementation_command="must-not-run",
    )


def _bind(daemon: PortalImplementationDaemon, events: list[dict]) -> None:
    task = SimpleNamespace(task_id=TASK_ID, status="ready")
    identity = SimpleNamespace(
        canonical_task_key=KEY,
        canonical_task_cid=CID,
        board_namespace=NAMESPACE,
    )
    daemon._load_tasks = lambda: [task]
    daemon._identity_for_task = lambda _task: identity
    daemon._iter_merge_lifecycle_events = lambda: list(events)


def _state() -> PortalTaskState:
    return PortalTaskState(
        last_implementation_task_id=TASK_ID,
        last_implementation_task_cid=CID,
        last_implementation_task_key=KEY,
        last_implementation_started_at="2026-11-07T01:39:15.431469+00:00",
        implementation_attempts={TASK_ID: 1},
        implementation_attempts_by_cid={CID: 1},
    )


def _claim(daemon: PortalImplementationDaemon, *, owner: dict, state_dir: str | None = None) -> dict:
    claim = {
        "kind": IMPLEMENTATION_TASK_CLAIM_LOCK_KIND,
        "lease_role": IMPLEMENTATION_DISPATCH_INTENT_LEASE_ROLE,
        "task_id": TASK_ID,
        "canonical_task_cid": CID,
        "canonical_task_key": KEY,
        "board_namespace": NAMESPACE,
        "attempt": 2,
        "pid": owner["pid"],
        "lease_id": "lease-dead-intent",
        "state_dir": state_dir or str(daemon.state_path.parent.resolve()),
        "state_path": str(daemon.state_path.resolve()),
        "repository_id": daemon.merge_target_repository_id,
        "worktree_root": str(daemon.repo_root.resolve()),
        "repo_root": "",
        "task_shard_count": daemon.task_shard_count,
        "task_shard_index": daemon.task_shard_index,
        "started_at": "2026-11-07T01:44:41.576158+00:00",
        "owner_process_birth": owner,
    }
    claim["dispatch_authority_cid"] = (
        PortalImplementationDaemon._implementation_dispatch_authority_cid(claim)
    )
    return claim


def _write(daemon: PortalImplementationDaemon, claim: dict) -> Path:
    path = daemon._implementation_task_claim_path(TASK_ID, canonical_task_cid=CID)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(claim), encoding="utf-8")
    sibling = path.parent / "canonical-task-ux-sibling.lock"
    sibling.write_text('{"kind":"implementation_task_claim","task_id":"CIMP-UX"}\n', encoding="utf-8")
    return path


def _dead_owner() -> dict:
    birth = read_process_birth(1)
    assert birth is not None
    return {
        "pid": 1,
        "start_time_ticks": birth.start_time_ticks + 1,
        "boot_id": birth.boot_id,
        "parent_pid": 0,
    }


def _history() -> list[dict]:
    return [
        {
            "type": "implementation_started",
            "canonical_task_cid": CID,
            "attempt": 2,
            "timestamp": "2026-11-07T01:19:12.052853+00:00",
        },
        {
            "type": "implementation_finished",
            "canonical_task_cid": CID,
            "attempt": 2,
            "timestamp": "2026-11-07T01:19:17.289981+00:00",
        },
        {
            "type": "implementation_started",
            "canonical_task_cid": CID,
            "attempt": 3,
            "timestamp": "2026-11-07T01:30:42.252884+00:00",
        },
    ]


def test_dead_unstarted_intent_releases_without_refund_or_completion(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    daemon = _daemon(tmp_path)
    _bind(daemon, _history())
    state = _state()
    path = _write(daemon, _claim(daemon, owner=_dead_owner()))
    sibling = path.parent / "canonical-task-ux-sibling.lock"

    result = daemon._reconcile_quiesced_implementation_task_claim(state)

    assert result["reconciled"] is True
    assert result["blocked"] is False
    assert result["reason"] == "quiesced_dead_dispatch_intent_released"
    assert result["attempt_refunded"] is False
    assert result["completion_authority"] is False
    assert result["task_status"] == "todo"
    assert state.implementation_attempts[TASK_ID] == 1
    assert state.implementation_attempts_by_cid[CID] == 1
    assert not path.exists()
    assert sibling.read_text(encoding="utf-8").startswith('{"kind"')


def test_live_owner_and_foreign_state_dir_keep_the_claim(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    daemon = _daemon(tmp_path)
    _bind(daemon, _history())
    live = current_process_birth().to_dict()
    live_path = _write(daemon, _claim(daemon, owner=live))

    live_result = daemon._reconcile_quiesced_implementation_task_claim(_state())

    assert live_result["blocked"] is True
    assert live_result["reason"] == "task_claim_owner_still_active"
    assert live_path.is_file()

    foreign = _claim(daemon, owner=_dead_owner(), state_dir=str(tmp_path / "other-campaign"))
    live_path.write_text(json.dumps(foreign), encoding="utf-8")
    foreign_result = daemon._reconcile_quiesced_implementation_task_claim(_state())

    assert foreign_result["blocked"] is True
    assert foreign_result["reason"] == "task_claim_identity_mismatch"
    assert json.loads(live_path.read_text(encoding="utf-8"))["state_dir"].endswith("other-campaign")


def test_post_publication_start_and_wide_counter_gap_stay_closed(tmp_path: Path, monkeypatch) -> None:
    _isolate(monkeypatch, tmp_path)
    daemon = _daemon(tmp_path)
    started = _history() + [
        {
            "type": "implementation_started",
            "canonical_task_cid": CID,
            "attempt": 2,
            "timestamp": "2026-11-07T01:45:00.000000+00:00",
        }
    ]
    _bind(daemon, started)
    path = _write(daemon, _claim(daemon, owner=_dead_owner()))

    started_result = daemon._reconcile_quiesced_implementation_task_claim(_state())

    assert started_result["blocked"] is True
    assert started_result["reason"] == "task_claim_identity_mismatch"
    assert path.is_file()

    gap = json.loads(path.read_text(encoding="utf-8"))
    gap["attempt"] = 4
    gap["dispatch_authority_cid"] = (
        PortalImplementationDaemon._implementation_dispatch_authority_cid(gap)
    )
    path.write_text(json.dumps(gap), encoding="utf-8")
    _bind(daemon, _history())
    gap_result = daemon._reconcile_quiesced_implementation_task_claim(_state())

    assert gap_result["blocked"] is True
    assert gap_result["reason"] == "task_claim_identity_mismatch"
    assert json.loads(path.read_text(encoding="utf-8"))["attempt"] == 4
