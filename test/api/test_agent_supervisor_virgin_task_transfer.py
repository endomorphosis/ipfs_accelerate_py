from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    build_arg_parser as build_multi_supervisor_arg_parser,
)
from ipfs_accelerate_py.agent_supervisor.runtime.multi_supervisor_runner import (
    common_args_from_parsed_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import (
    implementation_daemon as implementation_daemon_module,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalImplementationDaemon,
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    parse_args as parse_daemon_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon_runner import (
    build_portal_implementation_daemon_from_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    PortalImplementationSupervisor,
    supervisor_config_from_args,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_supervisor import (
    parse_args as parse_supervisor_args,
)


def _task_ids_for_home(home: int, count: int, *, total: int = 2) -> list[str]:
    task_ids: list[str] = []
    candidate = 0
    while len(task_ids) < total:
        task_id = f"ACCEL-{candidate:03d}"
        digest = hashlib.sha256(task_id.encode("utf-8")).hexdigest()
        if int(digest[:8], 16) % count == home:
            task_ids.append(task_id)
        candidate += 1
    return task_ids


def _write_board(path: Path, task_ids: list[str]) -> None:
    sections = ["# Agent Todos", ""]
    for index, task_id in enumerate(task_ids):
        sections.extend(
            [
                f"## {task_id} Ready task {index}",
                "",
                "- Status: todo",
                "- Completion: manual",
                "- Priority: P1",
                "- Track: ops",
                f"- Outputs: src/task_{index}.py",
                "",
            ]
        )
    path.write_text("\n".join(sections), encoding="utf-8")


def _daemon(
    repo: Path,
    board: Path,
    *,
    lane: int,
    count: int,
    task_header_prefix: str = "## ACCEL-",
) -> PortalImplementationDaemon:
    git_dir = repo / ".git"
    git_dir.mkdir(mode=0o700, exist_ok=True)
    state_dir = repo / "state" / f"lane-{lane}"
    return PortalImplementationDaemon(
        todo_path=board,
        state_path=state_dir / "state.json",
        strategy_path=state_dir / "strategy.json",
        events_path=state_dir / "events.jsonl",
        repo_root=repo,
        task_header_prefix=task_header_prefix,
        task_shard_count=count,
        task_shard_index=lane,
        strict_task_sharding=True,
        idle_lane_work_stealing="virgin-transfer",
    )


def _converge_rendezvous(
    lanes: list[PortalImplementationDaemon],
    *,
    max_rounds: int = 4,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for _round in range(max_rounds):
        results = [lane.run_once() for lane in lanes]
        if all(
            result["virgin_task_transfer"]["rendezvous_state"]
            == "active"
            for result in results
        ):
            return results
    raise AssertionError("virgin transfer rendezvous did not converge")


def _expire_rendezvous(daemon: PortalImplementationDaemon) -> None:
    daemon._virgin_transfer_rendezvous_deadline_monotonic = 0.0


def test_non_consuming_exact_revision_transfers_and_binds_task_claim(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)
    target = recipient._load_tasks()[0]
    target_cid = recipient._canonical_ref(target)

    # VGO-021-like dependency preflight evidence is explicitly non-consuming:
    # no attempt count, protected latch, lifecycle, claim, or provider effect.
    recipient._record_event(
        "validation_project_dependency_preflight_failed",
        {
            "task_id": target.task_id,
            "canonical_task_cid": target_cid,
            "attempt_consumed": False,
            "provider_dispatched": False,
        },
    )
    request_result = recipient.run_once()
    assert request_result["virgin_task_transfer"]["request_task_id"] == ""
    assert request_result["virgin_task_transfer"]["rendezvous_state"] == (
        "pending"
    )
    home.run_once()
    request_result = recipient.run_once()
    home_result = home.run_once()
    recipient_result = recipient.run_once()

    assert request_result["active_task_id"] == ""
    assert request_result["virgin_task_transfer"]["request_task_id"] == target.task_id
    assert home_result["virgin_task_transfer"]["granted_away_task_ids"] == [
        target.task_id
    ]
    assert recipient_result["active_task_id"] == target.task_id
    grant = recipient._active_virgin_transfer_grants[target_cid]
    claim = recipient._build_implementation_task_claim_metadata(
        target,
        1,
        "2026-08-12T00:00:00+00:00",
    )
    assert claim["virgin_task_transfer"]["grant_id"] == grant["grant_id"]
    assert claim["virgin_task_transfer"]["home_shard_index"] == 1
    assert claim["virgin_task_transfer"]["recipient_shard_index"] == 0
    assert recipient._virgin_transfer_root() in recipient._runtime_source_paths()[
        "lease"
    ]


def test_display_or_cid_attempt_count_rejects_virgin_transfer(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)
    target = home._load_tasks()[0]
    target_cid = home._canonical_ref(target)

    recipient.run_once()
    PortalTaskState(
        implementation_attempts={target.task_id: 1},
        implementation_attempts_by_cid={target_cid: 1},
    ).save(home.state_path)
    home.run_once()
    recipient.run_once()
    home_result = home.run_once()

    assert target.task_id not in home_result["virgin_task_transfer"][
        "granted_away_task_ids"
    ]
    assert not home._virgin_transfer_grant_path(target).exists()
    eligibility = home._virgin_transfer_eligibility(
        target,
        home._active_virgin_transfer_registrations(),
        current_state=PortalTaskState.load(home.state_path),
    )
    assert "lane_1_display_attempt_count_nonzero" in eligibility["reasons"]
    assert "lane_1_cid_attempt_count_nonzero" in eligibility["reasons"]


def test_nonvirgin_vgo010_does_not_starve_later_virgin_vgo021(
    tmp_path: Path,
) -> None:
    task_ids = ["VGO-010", "VGO-021"]
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    lanes = [
        _daemon(
            tmp_path,
            board,
            lane=lane,
            count=4,
            task_header_prefix="## VGO-",
        )
        for lane in range(4)
    ]
    requester = lanes[0]
    home = lanes[3]
    tasks = home._load_tasks()
    nonvirgin, virgin = tasks
    assert home._task_home_shard_index(nonvirgin.task_id) == 3
    assert home._task_home_shard_index(virgin.task_id) == 3

    # Register the three idle lanes. Partial membership publishes no unproved
    # demand and cannot select ordinary work.
    for lane in lanes[:3]:
        partial = lane.run_once()
        assert partial["active_task_id"] == ""
        assert partial["virgin_task_transfer"]["request_task_id"] == ""
    PortalTaskState(
        implementation_attempts={nonvirgin.task_id: 1},
        implementation_attempts_by_cid={
            home._canonical_ref(nonvirgin): 1,
        },
    ).save(home.state_path)
    first_home_result = home.run_once()
    assert first_home_result["active_task_id"] == ""
    assert nonvirgin.task_id not in first_home_result[
        "virgin_task_transfer"
    ]["granted_away_task_ids"]

    lanes[1].run_once()
    lanes[2].run_once()
    advanced = requester.run_once()
    assert advanced["virgin_task_transfer"]["request_task_id"] == (
        virgin.task_id
    )
    second_home_result = home.run_once()
    assert second_home_result["virgin_task_transfer"][
        "granted_away_task_ids"
    ] == [virgin.task_id]
    assert not home._virgin_transfer_grant_path(nonvirgin).exists()
    assert home._virgin_transfer_grant_path(virgin).exists()


def test_live_shaped_partial_registration_waits_for_exact_demand_round(
    tmp_path: Path,
) -> None:
    task_ids = ["VGO-010", "VGO-021", "VGO-034"]
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    lanes = [
        _daemon(
            tmp_path,
            board,
            lane=lane,
            count=4,
            task_header_prefix="## VGO-",
        )
        for lane in range(4)
    ]
    tasks = {task.task_id: task for task in lanes[3]._load_tasks()}
    assert lanes[3]._task_home_shard_index("VGO-010") == 3
    assert lanes[3]._task_home_shard_index("VGO-021") == 3
    assert lanes[2]._task_home_shard_index("VGO-034") == 2
    for lane, task_id, attempt in (
        (lanes[3], "VGO-010", 1),
        (lanes[2], "VGO-034", 3),
    ):
        state = PortalTaskState()
        lane._record_task_attempt(state, tasks[task_id], attempt)
        state.save(lane.state_path)

    # Reproduce the launch: idle lanes snapshot before lane 3 registers and
    # lane 2/3 reach selection while current-round intents are incomplete.
    for lane in (lanes[1], lanes[2], lanes[0], lanes[3]):
        partial = lane.run_once()
        assert partial["active_task_id"] == ""
        assert partial["selection_idle_reason"] == (
            "virgin_transfer_rendezvous_pending"
        )
        assert partial["virgin_task_transfer"]["request_task_id"] == ""
    assert not list(
        (lanes[0]._virgin_transfer_root() / "requests").glob("*.json")
    )
    assert all(
        entry.attempt_count == 0
        for lane in lanes
        for entry in lane.task_queue.entries.values()
    )
    assert not list(
        (tmp_path / ".git" / "implementation-task-claims").glob(
            "canonical-task-*.lock"
        )
    )

    # Idle lanes prove exact virginity only after full membership. Home lanes
    # publish no-demand acks last, then the home-3 pass grants before selection.
    idle_zero = lanes[0].run_once()
    idle_one = lanes[1].run_once()
    lane_two = lanes[2].run_once()
    assert idle_zero["virgin_task_transfer"]["request_task_id"] == "VGO-021"
    assert idle_one["virgin_task_transfer"]["request_task_id"] == "VGO-021"
    assert lane_two["active_task_id"] == "VGO-034"
    lane_three = lanes[3].run_once()
    assert lane_three["active_task_id"] == "VGO-010"
    assert lane_three["virgin_task_transfer"]["granted_away_task_ids"] == [
        "VGO-021"
    ]
    recipient = lanes[0].run_once()
    assert recipient["active_task_id"] == "VGO-021"
    grant = lanes[3]._valid_virgin_transfer_grant(tasks["VGO-021"])
    assert grant is not None
    assert grant["recipient_shard_index"] == 0


def test_registration_birth_change_invalidates_old_round(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)

    _converge_rendezvous([recipient, home])
    old_registration = recipient._publish_virgin_transfer_registration()
    old_membership = home._virgin_transfer_membership_id(
        home._active_virgin_transfer_registrations()
    )
    replacement_owner = dict(old_registration["owner"])
    replacement_owner["start_time_ticks"] += 1
    original_birth = implementation_daemon_module.current_process_birth
    monkeypatch.setattr(
        implementation_daemon_module,
        "current_process_birth",
        lambda: implementation_daemon_module.ProcessBirthIdentity.from_dict(
            replacement_owner
        ),
    )
    replacement_registration = recipient._publish_virgin_transfer_registration()
    monkeypatch.setattr(
        implementation_daemon_module,
        "current_process_birth",
        original_birth,
    )
    monkeypatch.setattr(
        implementation_daemon_module,
        "owner_liveness",
        lambda _owner: implementation_daemon_module.OwnerLiveness.ALIVE,
    )

    assert replacement_registration["registration_id"] != (
        old_registration["registration_id"]
    )
    result = home.run_once()
    assert result["virgin_task_transfer"]["membership_id"] != old_membership
    assert result["virgin_task_transfer"]["rendezvous_state"] == "pending"
    assert result["active_task_id"] == ""
    assert result["virgin_task_transfer"]["ack_count"] < 2


def test_intent_is_immutable_within_exact_membership(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)
    recipient.run_once()
    home.run_once()
    recipient.run_once()
    registrations = recipient._active_virgin_transfer_registrations()
    membership_id = recipient._virgin_transfer_membership_id(registrations)
    original = recipient._valid_virgin_transfer_intent(
        0,
        registrations,
        membership_id=membership_id,
    )
    assert original is not None
    assert original["request_id"]

    rewritten = recipient._publish_virgin_transfer_intent(
        registrations[0],
        membership_id=membership_id,
        request=None,
        no_request_reason="local_ready",
    )
    assert rewritten == original
    assert len(
        list((recipient._virgin_transfer_root() / "intents").glob("*.json"))
    ) == 2


def test_rendezvous_timeout_resumes_strict_home_selection_without_transfer(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    home = _daemon(tmp_path, board, lane=1, count=2)

    pending = home.run_once()
    assert pending["active_task_id"] == ""
    assert pending["virgin_task_transfer"]["rendezvous_state"] == "pending"
    assert 0.0 <= pending["next_wake_after_seconds"] <= (
        implementation_daemon_module.VIRGIN_TASK_TRANSFER_RENDEZVOUS_TIMEOUT_SECONDS
    )
    _expire_rendezvous(home)
    timed_out = home.run_once()

    assert timed_out["virgin_task_transfer"]["rendezvous_state"] == (
        "timed_out"
    )
    assert timed_out["active_task_id"] in task_ids
    assert timed_out["virgin_task_transfer"]["granted_away_task_ids"] == []
    state = PortalTaskState.load(home.state_path)
    assert state.implementation_attempts == {}
    assert state.implementation_attempts_by_cid == {}


def test_direct_daemon_loop_bounds_wait_by_missing_lane_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    pending = _daemon(tmp_path, board, lane=1, count=2).run_once()
    assert pending["virgin_task_transfer"]["rendezvous_state"] == "pending"
    assert 0.0 <= pending["next_wake_after_seconds"] <= (
        implementation_daemon_module.VIRGIN_TASK_TRANSFER_RENDEZVOUS_TIMEOUT_SECONDS
    )
    observed_waits: list[float] = []
    closed: list[bool] = []

    class DirectLoopDaemon:
        def __init__(self, **_kwargs: object) -> None:
            pass

        def run_once(self) -> dict[str, object]:
            return pending

        def wait_for_wake(self, timeout: float) -> None:
            observed_waits.append(timeout)
            raise SystemExit(0)

        def close_event_runtime(self) -> None:
            closed.append(True)

    monkeypatch.setattr(
        implementation_daemon_module,
        "PortalImplementationDaemon",
        DirectLoopDaemon,
    )
    with pytest.raises(SystemExit) as stopped:
        implementation_daemon_module.main(
            [
                "--interval",
                "300",
                "--todo-path",
                str(board),
                "--state-dir",
                str(tmp_path / "direct-state"),
                "--task-shard-count",
                "2",
                "--task-shard-index",
                "1",
                "--strict-task-sharding",
                "--idle-lane-work-stealing",
                "virgin-transfer",
            ]
        )

    assert stopped.value.code == 0
    assert observed_waits == [pending["next_wake_after_seconds"]]
    assert observed_waits[0] < 300.0
    assert closed == [True]


def test_timed_out_round_cannot_reactivate_grant_after_home_attempt(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)

    recipient.run_once()
    home.run_once()
    recipient.run_once()
    granted = home.run_once()
    target_id = granted["virgin_task_transfer"]["granted_away_task_ids"][0]
    target = next(task for task in home._load_tasks() if task.task_id == target_id)
    grant_path = home._virgin_transfer_grant_path(target)
    assert grant_path.exists()

    # Model the timeout fallback's critical outcome: the current round loses
    # an ack, then strict-home work consumes the exact task revision before
    # that immutable ack is republished.
    recipient._virgin_transfer_ack_path(0).unlink()
    _expire_rendezvous(home)
    timed_out = home.run_once()
    assert timed_out["virgin_task_transfer"]["rendezvous_state"] == "timed_out"
    assert timed_out["virgin_task_transfer"]["granted_away_task_ids"] == []
    home_state = PortalTaskState.load(home.state_path)
    home._record_task_attempt(home_state, target, 1)
    home_state.save(home.state_path)

    recipient_result = recipient.run_once()
    home_result = home.run_once()
    assert recipient_result["active_task_id"] != target_id
    assert target_id not in recipient_result["virgin_task_transfer"][
        "granted_to_lane_task_ids"
    ]
    assert target_id not in home_result["virgin_task_transfer"][
        "granted_away_task_ids"
    ]
    assert grant_path.exists()


@pytest.mark.parametrize(
    "liveness",
    (
        implementation_daemon_module.OwnerLiveness.DEAD,
        implementation_daemon_module.OwnerLiveness.UNKNOWN,
    ),
)
def test_inactive_lane_registration_times_out_to_strict_home(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    liveness: implementation_daemon_module.OwnerLiveness,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)
    recipient.run_once()
    home.run_once()
    monkeypatch.setattr(
        implementation_daemon_module,
        "owner_liveness",
        lambda _owner: liveness,
    )
    home._last_safety_reconciliation_monotonic = 0.0
    pending = home.run_once()
    assert pending["virgin_task_transfer"]["active_registration_count"] == 0
    assert pending["virgin_task_transfer"]["rendezvous_state"] == "pending"
    assert pending["active_task_id"] == ""

    _expire_rendezvous(home)
    home._last_safety_reconciliation_monotonic = 0.0
    timed_out = home.run_once()
    assert timed_out["virgin_task_transfer"]["rendezvous_state"] == (
        "timed_out"
    )
    assert timed_out["active_task_id"] in task_ids
    assert timed_out["virgin_task_transfer"]["granted_away_task_ids"] == []


def test_intent_and_ack_directories_are_runtime_lease_sources(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    lane = _daemon(tmp_path, board, lane=0, count=2)
    transfer_root = lane._virgin_transfer_root()
    lease_sources = set(lane._runtime_source_paths()["lease"])

    assert transfer_root / "intents" in lease_sources
    assert transfer_root / "acks" in lease_sources


@pytest.mark.parametrize(
    ("mutation", "reason"),
    (
        ("completed", "task_completed"),
        ("revised", "task_revision_changed"),
    ),
)
def test_terminal_or_revised_task_retires_effect_free_grant(
    tmp_path: Path,
    mutation: str,
    reason: str,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)
    target = home._load_tasks()[0]

    _converge_rendezvous([recipient, home])
    grant_path = home._virgin_transfer_grant_path(target)
    assert grant_path.exists()
    board_text = board.read_text(encoding="utf-8")
    if mutation == "completed":
        board_text = board_text.replace(
            "- Status: todo",
            "- Status: completed",
            1,
        )
    else:
        board_text = board_text.replace(
            f"## {target.task_id} Ready task 0",
            f"## {target.task_id} Revised task 0",
            1,
        )
    board.write_text(board_text, encoding="utf-8")

    result = home.run_once()

    assert not grant_path.exists()
    revoked = result["virgin_task_transfer"]["revoked_grants"]
    assert len(revoked) == 1
    assert revoked[0]["grant_id"]
    assert revoked[0]["task_id"] == target.task_id
    assert revoked[0]["reason"] == reason


def test_dead_recipient_retires_effect_free_grant(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)
    target = home._load_tasks()[0]

    _converge_rendezvous([recipient, home])
    grant_path = home._virgin_transfer_grant_path(target)
    assert grant_path.exists()
    monkeypatch.setattr(
        implementation_daemon_module,
        "owner_liveness",
        lambda _owner: implementation_daemon_module.OwnerLiveness.DEAD,
    )
    home._last_safety_reconciliation_monotonic = 0.0

    result = home.run_once()

    assert not grant_path.exists()
    assert result["virgin_task_transfer"]["revoked_grants"][0][
        "reason"
    ] == "recipient_dead_before_effect"


def test_stale_grant_is_rejected_at_task_claim_publication(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)
    target = home._load_tasks()[0]

    _converge_rendezvous([recipient, home])
    recipient.run_once()
    metadata = recipient._build_implementation_task_claim_metadata(
        target,
        1,
        "2026-08-12T00:00:00+00:00",
    )
    grant_path = recipient._virgin_transfer_grant_path(target)
    grant_path.unlink()
    claim_path = recipient._implementation_task_claim_path(
        target.task_id,
        canonical_task_cid=recipient._canonical_ref(target),
    )

    claimed, reason, _existing = (
        recipient._try_acquire_implementation_task_claim(
            claim_path,
            metadata,
        )
    )

    assert claimed is False
    assert reason == "virgin_transfer_grant_invalid"
    assert not claim_path.exists()


def test_attempt_cannot_interleave_between_transfer_proof_and_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)
    target = home._load_tasks()[0]

    _converge_rendezvous([recipient, home])
    recipient.run_once()
    metadata = recipient._build_implementation_task_claim_metadata(
        target,
        1,
        "2026-08-12T00:00:00+00:00",
    )
    claim_path = recipient._implementation_task_claim_path(
        target.task_id,
        canonical_task_cid=recipient._canonical_ref(target),
    )
    original_eligibility = recipient._virgin_transfer_eligibility
    proof_reached = threading.Event()
    release_proof = threading.Event()

    def barrier_eligibility(*args: object, **kwargs: object) -> dict[str, object]:
        result = original_eligibility(*args, **kwargs)
        if result.get("eligible") is True:
            proof_reached.set()
            assert release_proof.wait(timeout=5)
        return result

    monkeypatch.setattr(
        recipient,
        "_virgin_transfer_eligibility",
        barrier_eligibility,
    )
    recipient_outcome: list[tuple[bool, str, dict[str, object] | None]] = []
    competitor_outcome: list[tuple[bool, str, dict[str, object] | None]] = []

    def acquire_recipient() -> None:
        recipient_outcome.append(
            recipient._try_acquire_implementation_task_claim(
                claim_path,
                metadata,
            )
        )

    competitor_metadata = home._build_implementation_task_claim_metadata(
        target,
        1,
        "2026-08-12T00:00:01+00:00",
    )
    recipient_lease_id = metadata["lease_id"]
    original_home_owner_active = (
        home._implementation_task_claim_owner_is_active
    )

    def home_owner_active(claim: dict[str, object]) -> bool:
        # Both daemons are threads in this test process. Model the recipient as
        # a genuinely live foreign owner independently of the pytest launcher
        # spelling in /proc/<pid>/cmdline, which varies across test runners.
        if claim.get("lease_id") == recipient_lease_id:
            return True
        return original_home_owner_active(claim)

    monkeypatch.setattr(
        home,
        "_implementation_task_claim_owner_is_active",
        home_owner_active,
    )

    def acquire_competitor() -> None:
        competitor_outcome.append(
            home._try_acquire_implementation_task_claim(
                claim_path,
                competitor_metadata,
            )
        )

    recipient_thread = threading.Thread(target=acquire_recipient)
    competitor_thread = threading.Thread(target=acquire_competitor)
    recipient_thread.start()
    assert proof_reached.wait(timeout=5)
    competitor_thread.start()
    # The competing home lane must be waiting on the exact claim guard while
    # the recipient is paused after proof but before claim publication.
    competitor_thread.join(timeout=0.2)
    assert competitor_thread.is_alive()
    release_proof.set()
    recipient_thread.join(timeout=5)
    competitor_thread.join(timeout=5)

    assert not recipient_thread.is_alive()
    assert not competitor_thread.is_alive()
    assert recipient_outcome[0][0] is True
    assert competitor_outcome[0][0] is False
    assert competitor_outcome[0][1] == "lock_exists"
    assert recipient._release_implementation_task_claim(
        claim_path,
        metadata,
    )

    # Once the competing lane records a consumed exact-revision attempt, a
    # subsequent transfer claimant re-proves under the guard and rejects it.
    home_state = PortalTaskState.load(home.state_path)
    home._record_task_attempt(home_state, target, 1)
    home_state.save(home.state_path)
    claimed, reason, _existing = (
        recipient._try_acquire_implementation_task_claim(
            claim_path,
            metadata,
        )
    )
    assert claimed is False
    assert reason == "virgin_transfer_grant_invalid"
    assert not claim_path.exists()


def test_transfer_authority_requires_private_directories_and_records(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    recipient.run_once()
    root = recipient._virgin_transfer_root()

    for directory in (
        root.parent,
        root,
        root / "registrations",
        root / "requests",
        root / "intents",
        root / "acks",
        root / "grants",
    ):
        assert directory.stat(follow_symlinks=False).st_mode & 0o777 == 0o700
    registration_path = recipient._virgin_transfer_registration_path(0)
    info = registration_path.stat(follow_symlinks=False)
    assert info.st_mode & 0o777 == 0o600
    assert info.st_nlink == 1

    os.chmod(registration_path, 0o644)
    with pytest.raises(RuntimeError, match="owned 0600 regular files"):
        recipient._active_virgin_transfer_registrations()


def test_two_requesters_receive_exactly_one_home_lane_grant(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(2, 3)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    requester_zero = _daemon(tmp_path, board, lane=0, count=3)
    requester_one = _daemon(tmp_path, board, lane=1, count=3)
    home = _daemon(tmp_path, board, lane=2, count=3)

    requester_zero.run_once()
    requester_one.run_once()
    home.run_once()
    zero_result = requester_zero.run_once()
    one_result = requester_one.run_once()
    assert zero_result["virgin_task_transfer"]["request_task_id"] == task_ids[0]
    assert one_result["virgin_task_transfer"]["request_task_id"] == task_ids[0]

    home_result = home.run_once()
    grants = list((home._virgin_transfer_root() / "grants").glob("*.json"))
    assert len(grants) == 1
    assert home_result["virgin_task_transfer"]["granted_away_task_ids"] == [
        task_ids[0]
    ]
    grant = home._valid_virgin_transfer_grant(home._load_tasks()[0])
    assert grant is not None
    assert grant["recipient_shard_index"] == 0


def test_losing_requester_rotates_to_second_surplus_task(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(2, 3, total=3)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    requester_zero = _daemon(tmp_path, board, lane=0, count=3)
    requester_one = _daemon(tmp_path, board, lane=1, count=3)
    home = _daemon(tmp_path, board, lane=2, count=3)

    requester_zero.run_once()
    requester_one.run_once()
    home.run_once()
    requester_zero.run_once()
    requester_one.run_once()
    first_home = home.run_once()
    assert first_home["virgin_task_transfer"]["granted_away_task_ids"] == [
        task_ids[0]
    ]

    winner = requester_zero.run_once()
    loser = requester_one.run_once()
    assert winner["virgin_task_transfer"]["request_task_id"] == ""
    assert loser["virgin_task_transfer"]["request_task_id"] == task_ids[1]
    second_home = home.run_once()
    assert second_home["virgin_task_transfer"]["granted_away_task_ids"] == [
        task_ids[0],
        task_ids[1],
    ]
    assert len(list((home._virgin_transfer_root() / "grants").glob("*.json"))) == 2


def test_completed_transfer_rotates_same_lane_to_next_task(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(1, 2, total=3)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)

    recipient.run_once()
    home.run_once()
    recipient.run_once()
    first_home = home.run_once()
    first_task_id = first_home["virgin_task_transfer"]["granted_away_task_ids"][0]
    recipient.run_once()
    rendered = board.read_text(encoding="utf-8")
    heading = f"## {first_task_id} "
    start = rendered.index(heading)
    status = rendered.index("- Status: todo", start)
    rendered = (
        rendered[:status]
        + "- Status: completed"
        + rendered[status + len("- Status: todo") :]
    )
    board.write_text(rendered, encoding="utf-8")

    home.run_once()
    rotated = recipient.run_once()
    assert rotated["virgin_task_transfer"]["rendezvous_state"] == "active"
    assert rotated["virgin_task_transfer"]["request_task_id"] == task_ids[1]
    second_home = home.run_once()
    assert second_home["virgin_task_transfer"]["granted_away_task_ids"] == [
        task_ids[1]
    ]


def test_startup_certificate_survives_current_request_rotation(
    tmp_path: Path,
) -> None:
    task_ids = _task_ids_for_home(1, 2, total=3)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)
    _converge_rendezvous([recipient, home])
    registrations = home._active_virgin_transfer_registrations()
    membership_id = home._virgin_transfer_membership_id(registrations)
    startup_ack = home._valid_virgin_transfer_ack(
        0,
        registrations,
        membership_id=membership_id,
    )
    assert startup_ack is not None
    startup_request_id = startup_ack["request_id"]

    restarted = _daemon(tmp_path, board, lane=1, count=2)
    restarted_registrations = restarted._active_virgin_transfer_registrations()
    assert restarted._valid_virgin_transfer_ack(
        0,
        restarted_registrations,
        membership_id=membership_id,
    ) == startup_ack

    recipient.run_once()
    current = recipient._load_private_virgin_transfer_record(
        recipient._virgin_transfer_request_path(
            recipient._load_tasks()[0],
            0,
        )
    )
    assert current is None or current["request_id"] != startup_request_id
    assert home._valid_virgin_transfer_ack(
        0,
        registrations,
        membership_id=membership_id,
    ) == startup_ack


def test_grant_publication_serializes_with_current_demand_withdrawal(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    task_ids = _task_ids_for_home(1, 2)
    board = tmp_path / "todo.md"
    _write_board(board, task_ids)
    recipient = _daemon(tmp_path, board, lane=0, count=2)
    home = _daemon(tmp_path, board, lane=1, count=2)
    recipient.run_once()
    home.run_once()
    recipient.run_once()

    registrations = home._active_virgin_transfer_registrations()
    membership_id = home._virgin_transfer_membership_id(registrations)
    acks = home._converged_virgin_transfer_acks(
        home._load_tasks(),
        registrations,
        membership_id=membership_id,
    )
    task = home._load_tasks()[0]
    request = home._valid_virgin_transfer_request(
        task,
        0,
        registrations,
        membership_id=membership_id,
    )
    assert request is not None

    proof_reached = threading.Event()
    release_proof = threading.Event()
    original_eligibility = home._virgin_transfer_eligibility

    def barrier_eligibility(*args: object, **kwargs: object) -> dict[str, object]:
        result = original_eligibility(*args, **kwargs)
        if result.get("eligible") is True:
            proof_reached.set()
            assert release_proof.wait(timeout=5)
        return result

    monkeypatch.setattr(home, "_virgin_transfer_eligibility", barrier_eligibility)
    grant_outcome: list[dict[str, object] | None] = []

    def grant_current() -> None:
        grant_outcome.append(
            home._try_grant_virgin_transfer(
                task,
                request,
                registrations,
                acks,
                current_state=PortalTaskState.load(home.state_path),
            )
        )

    grant_thread = threading.Thread(target=grant_current)
    grant_thread.start()
    assert proof_reached.wait(timeout=5)
    withdrawn: list[bool] = []

    def withdraw_current() -> None:
        recipient._clear_other_virgin_transfer_requests()
        withdrawn.append(True)

    withdraw_thread = threading.Thread(target=withdraw_current)
    withdraw_thread.start()
    withdraw_thread.join(timeout=0.2)
    assert withdraw_thread.is_alive()
    release_proof.set()
    grant_thread.join(timeout=5)
    withdraw_thread.join(timeout=5)
    assert grant_outcome[0] is not None
    assert withdrawn == [True]
    assert not recipient._virgin_transfer_request_path(task, 0).exists()

    grant_path = home._virgin_transfer_grant_path(task)
    grant_path.unlink()
    rejected = home._try_grant_virgin_transfer(
        task,
        request,
        registrations,
        acks,
        current_state=PortalTaskState.load(home.state_path),
    )
    assert rejected is None
    assert not grant_path.exists()


def test_virgin_transfer_cli_propagates_through_all_wrappers(
    tmp_path: Path,
) -> None:
    board = tmp_path / "todo.md"
    _write_board(board, _task_ids_for_home(1, 2))
    daemon_args = parse_daemon_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "daemon-state"),
            "--task-shard-count",
            "2",
            "--task-shard-index",
            "0",
            "--strict-task-sharding",
            "--idle-lane-work-stealing",
            "virgin-transfer",
        ]
    )
    daemon, _context = build_portal_implementation_daemon_from_args(
        daemon_args,
        repo_root=tmp_path,
    )
    assert daemon.idle_lane_work_stealing == "virgin-transfer"

    supervisor_args = parse_supervisor_args(
        [
            "--todo-path",
            str(board),
            "--state-dir",
            str(tmp_path / "supervisor-state"),
            "--task-shard-count",
            "2",
            "--task-shard-index",
            "0",
            "--strict-task-sharding",
            "--idle-lane-work-stealing",
            "virgin-transfer",
        ]
    )
    config = supervisor_config_from_args(supervisor_args, repo_root=tmp_path)
    command = PortalImplementationSupervisor(config)._build_daemon_command()
    assert command.count("--idle-lane-work-stealing") == 1
    assert command[command.index("--idle-lane-work-stealing") + 1] == (
        "virgin-transfer"
    )

    multi_args = build_multi_supervisor_arg_parser().parse_args(
        [
            "--implementation-track",
            "T|worker.py|state|agent",
            "--implementation-supervisor-lanes-per-track",
            "2",
            "--implementation-supervisor-strict-task-sharding",
            "--implementation-supervisor-idle-lane-work-stealing",
            "virgin-transfer",
        ]
    )
    common = common_args_from_parsed_args(multi_args)
    assert "--strict-task-sharding" in common
    assert common[common.index("--idle-lane-work-stealing") + 1] == (
        "virgin-transfer"
    )
