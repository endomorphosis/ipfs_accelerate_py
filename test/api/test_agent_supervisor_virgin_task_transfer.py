from __future__ import annotations

import hashlib
import os
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
    home_result = home.run_once()

    assert target.task_id not in home_result["virgin_task_transfer"][
        "granted_away_task_ids"
    ]
    assert not home._virgin_transfer_grant_path(target).exists()
    events = home.events_path.read_text(encoding="utf-8")
    assert "display_attempt_count_nonzero" in events
    assert "cid_attempt_count_nonzero" in events


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

    # Register the three idle lanes. Until the cohort is complete they safely
    # request the first candidate; the home lane must reject its consumed
    # revision, and the requester must then advance to the later virgin task.
    for lane in lanes[:3]:
        lane.run_once()
    PortalTaskState(
        implementation_attempts={nonvirgin.task_id: 1},
        implementation_attempts_by_cid={
            home._canonical_ref(nonvirgin): 1,
        },
    ).save(home.state_path)
    first_home_result = home.run_once()
    assert nonvirgin.task_id not in first_home_result[
        "virgin_task_transfer"
    ]["granted_away_task_ids"]

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

    recipient.run_once()
    home.run_once()
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

    recipient.run_once()
    home.run_once()
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

    recipient.run_once()
    home.run_once()
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
