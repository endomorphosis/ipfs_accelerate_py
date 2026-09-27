"""Portal task state uses DuckDB/Quack, not a JSON authority file."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon.implementation_daemon import (
    PortalTaskState,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
    ENV_DUCKDB,
    ENV_DUCKLAKE,
    ENV_QUACK,
    load_portal_task_state,
    read_ducklake_rows,
    read_task_state_payload,
)


def test_configured_control_plane_does_not_write_task_state_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    database = tmp_path / "control" / "portal-task-state.duckdb"
    lake = tmp_path / "lake"
    monkeypatch.delenv(ENV_QUACK, raising=False)
    monkeypatch.setenv(ENV_DUCKDB, str(database))
    monkeypatch.setenv(ENV_DUCKLAKE, str(lake.resolve()))
    path = tmp_path / "lane_task_state.json"
    state = PortalTaskState(active_task_id="task-1", active_phase="implement")
    assert state.save(path) is True
    assert state.save(path) is False
    assert not path.exists()
    loaded = PortalTaskState.load(path)
    assert loaded.active_task_id == "task-1"
    assert loaded.active_phase == "implement"
    raw = load_portal_task_state(path.stem)
    assert raw is not None
    assert raw["active_task_id"] == "task-1"
    assert raw["completion_authority"] is False
    assert database.is_file()
    history = read_ducklake_rows(lake.resolve())
    assert history
    assert history[-1]["completion_authority"] is False
    assert history[-1]["authoritative"] is False
    assert "task-1" in history[-1]["body_json"]


def test_merge_handoff_uses_control_plane_instead_of_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.cst_kit_commit import (
        accept_spar_merge_nomination,
        merge_nomination_outstanding,
        nominate_current_authority_merge,
        spar_merge_hold_outstanding,
    )

    database = tmp_path / "control" / "portal-task-state.duckdb"
    monkeypatch.delenv(ENV_QUACK, raising=False)
    monkeypatch.setenv(ENV_DUCKDB, str(database))
    directory = tmp_path / "semantic-world"
    directory.mkdir()
    nominate_current_authority_merge(
        directory,
        directory / "disposable-worktree",
        {"pkg/mod.py": "def moved():\n    return 1\n"},
        mode="guarded",
    )
    assert not (directory / "merge-nomination.json").exists()
    assert merge_nomination_outstanding(coordination_dir=directory) is True
    accept_spar_merge_nomination(coordination_dir=directory)
    assert not (directory / "merge-owner-hold.json").exists()
    assert merge_nomination_outstanding(coordination_dir=directory) is False
    assert spar_merge_hold_outstanding(coordination_dir=directory) is True
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
        load_merge_handoff,
    )

    hold = load_merge_handoff(f"hold:{directory.resolve()}")
    assert hold is not None
    assert hold["merged"] is False
    assert hold["writes_repository"] is False
    assert hold["held_by"] == "current authority gates"
    assert hold["completion_authority"] is False


def test_bound_state_path_uses_its_database_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
        bind_task_state_control_plane,
    )

    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        ORCHESTRATION_DIR_ENV,
    )

    monkeypatch.delenv(ENV_QUACK, raising=False)
    monkeypatch.delenv(ENV_DUCKDB, raising=False)
    monkeypatch.setenv(ORCHESTRATION_DIR_ENV, str(tmp_path / "orchestration"))
    database = tmp_path / "lane.duckdb"
    bound = tmp_path / "lane_task_state.json"
    other = tmp_path / "other_task_state.json"
    bind_task_state_control_plane(bound, str(database))
    assert PortalTaskState(active_task_id="bound-task").save(bound) is True
    assert not bound.exists()
    assert not bound.with_name(bound.stem + ".observation.json").exists()
    from ipfs_accelerate_py.agent_supervisor.rescue.live_board_probe import (
        read_task_state_projection,
    )

    projected = read_task_state_projection(bound.parent, "lane")
    assert projected["active_task_id"] == "bound-task"
    assert projected["completion_authority"] is False
    assert PortalTaskState.load(bound).active_task_id == "bound-task"
    history = read_ducklake_rows(database.parent / "ducklake")
    assert history
    assert history[-1]["completion_authority"] is False
    assert "bound-task" in history[-1]["body_json"]
    assert PortalTaskState(active_task_id="json-task").save(other) is True
    assert not other.exists()
    assert PortalTaskState.load(other).active_task_id == "json-task"
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
        implicit_portal_database,
    )

    other_database = Path(implicit_portal_database(other))
    assert other_database.is_file()
    assert other_database.parent.name == "lanes"
    assert ".git" not in other_database.parts
    assert not (other.parent / "other_task_state.portal.duckdb").exists()


def test_bound_lane_stores_merge_handoff_without_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.cst_kit_commit import (
        accept_spar_merge_nomination,
        merge_nomination_outstanding,
        nominate_current_authority_merge,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
        bind_task_state_control_plane,
    )

    monkeypatch.delenv(ENV_QUACK, raising=False)
    monkeypatch.delenv(ENV_DUCKDB, raising=False)
    state_path = tmp_path / "state" / "lane_task_state.json"
    directory = state_path.parent / "semantic-world"
    directory.mkdir(parents=True)
    bind_task_state_control_plane(state_path, str(tmp_path / "lane.duckdb"))
    nominate_current_authority_merge(
        directory,
        directory / "disposable-worktree",
        {"pkg/mod.py": "x"},
        mode="guarded",
    )
    assert not (directory / "merge-nomination.json").exists()
    assert merge_nomination_outstanding(coordination_dir=directory) is True
    accept_spar_merge_nomination(coordination_dir=directory)
    assert not (directory / "merge-owner-hold.json").exists()
    assert merge_nomination_outstanding(coordination_dir=directory) is False


def test_unconfigured_merge_handoff_is_stored_in_duckdb_and_mirrored(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.semantic_refactoring.cst_kit_commit import (
        merge_nomination_outstanding,
        nominate_current_authority_merge,
    )

    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        ORCHESTRATION_DIR_ENV,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
        resolve_handoff_plane,
    )

    monkeypatch.delenv(ENV_QUACK, raising=False)
    monkeypatch.delenv(ENV_DUCKDB, raising=False)
    monkeypatch.delenv(ENV_DUCKLAKE, raising=False)
    monkeypatch.setenv(ORCHESTRATION_DIR_ENV, str(tmp_path / "orchestration"))
    directory = tmp_path / "semantic-world"
    directory.mkdir()
    nominate_current_authority_merge(
        directory,
        directory / "disposable-worktree",
        {"pkg/mod.py": "x"},
        mode="guarded",
    )
    assert not (directory / "merge-nomination.json").exists()
    handoff_target, handoff_source = resolve_handoff_plane(directory)
    assert handoff_source == "implicit"
    assert handoff_target is not None
    handoff_database = Path(handoff_target)
    assert handoff_database.is_file()
    assert not str(handoff_database).startswith(str(directory))
    assert ".git" not in handoff_database.parts
    assert merge_nomination_outstanding(coordination_dir=directory) is True
    history = read_ducklake_rows(handoff_database.parent / "ducklake", kind="spar_merge_handoff")
    assert history
    assert history[-1]["completion_authority"] is False
    assert history[-1]["authoritative"] is False
    assert "pkg/mod.py" in history[-1]["body_json"]


def test_repo_task_state_uses_the_board_control_plane(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        ORCHESTRATION_DIR_ENV,
        default_control_plane_root,
    )

    monkeypatch.delenv(ENV_QUACK, raising=False)
    monkeypatch.delenv(ENV_DUCKDB, raising=False)
    monkeypatch.setenv(ORCHESTRATION_DIR_ENV, str(tmp_path / "orchestration"))
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    path = repo / "lane_task_state.json"
    assert PortalTaskState(active_task_id="board-task").save(path) is True
    assert not path.exists()
    database = default_control_plane_root(repo) / "control.duckdb"
    assert database.is_file()
    assert ".git" not in database.parts
    assert not str(database).startswith(str(repo))
    assert PortalTaskState.load(path).active_task_id == "board-task"
    history = read_ducklake_rows(database.parent / "ducklake")
    assert history
    assert history[-1]["completion_authority"] is False
    assert "board-task" in history[-1]["body_json"]


def test_unconfigured_control_plane_still_uses_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        ORCHESTRATION_DIR_ENV,
    )
    from ipfs_accelerate_py.agent_supervisor.todo_daemon.portal_task_state_control_plane import (
        implicit_portal_database,
    )

    monkeypatch.delenv(ENV_QUACK, raising=False)
    monkeypatch.delenv(ENV_DUCKDB, raising=False)
    monkeypatch.setenv(ORCHESTRATION_DIR_ENV, str(tmp_path / "orchestration"))
    path = tmp_path / "lane_task_state.json"
    state = PortalTaskState(active_task_id="task-json")
    assert state.save(path) is True
    assert not path.exists()
    assert not path.with_name(path.stem + ".observation.json").exists()
    database = Path(implicit_portal_database(path))
    assert database.is_file()
    assert database.parent.name == "lanes"
    assert ".git" not in database.parts
    assert not (path.parent / "lane_task_state.portal.duckdb").exists()
    assert PortalTaskState.load(path).active_task_id == "task-json"
    assert read_task_state_payload(path)["active_task_id"] == "task-json"
    history = read_ducklake_rows(database.parent / "ducklake")
    assert history
    assert history[-1]["completion_authority"] is False
    assert "task-json" in history[-1]["body_json"]


def test_other_default_supervisor_duckdb_catalogs_stay_out_of_git(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        merge_target_queue_dir,
    )
    from ipfs_accelerate_py.agent_supervisor.merge.merge_queue import MergeQueue
    from ipfs_accelerate_py.agent_supervisor.objectives.bundle_supervisor import (
        DynamicBundleScheduler,
        default_coordination_database,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        ORCHESTRATION_DIR_ENV,
        orchestration_scope_dir,
    )

    monkeypatch.setenv(ORCHESTRATION_DIR_ENV, str(tmp_path / "orchestration"))
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    index = repo / "index.json"
    index.write_text("{}", encoding="utf-8")
    scope = orchestration_scope_dir(repo)

    queue_dir = merge_target_queue_dir(repo, "main")
    queue = MergeQueue(queue_dir)
    assert queue.database_path == queue_dir / "merge_queue.duckdb"
    assert queue.database_path.is_file()
    assert queue_dir.parent == scope / "agent-merge-trains"
    assert ".git" not in queue.database_path.parts
    assert not str(queue.database_path).startswith(str(repo))
    assert not (repo / ".git" / "agent-merge-trains").exists()

    supplied = tmp_path / "caller-queue"
    supplied_queue = MergeQueue(supplied)
    assert supplied_queue.database_path == (supplied / "merge_queue.duckdb").resolve()
    assert supplied_queue.database_path.is_file()

    coordination = default_coordination_database(repo)
    assert coordination == scope / "bundle" / "coordination.duckdb"
    assert ".git" not in coordination.parts
    scheduler = DynamicBundleScheduler(bundle_index_path=index, repo_root=repo)
    assert scheduler.coordination_path == coordination.resolve()
    explicit = repo / "state" / "coordination.duckdb"
    bound = DynamicBundleScheduler(
        bundle_index_path=index,
        repo_root=repo,
        state_root=repo / "state",
        coordination_path=explicit,
    )
    assert bound.coordination_path == explicit.resolve()


def test_unlocked_git_catalogs_move_to_the_orchestration_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
        merge_target_queue_dir,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        ORCHESTRATION_DIR_ENV,
        default_control_plane_root,
        orchestration_scope_dir,
    )

    monkeypatch.setenv(ORCHESTRATION_DIR_ENV, str(tmp_path / "orchestration"))
    repo = tmp_path / "repo"
    legacy_board = repo / ".git" / "agent-board-control-plane"
    legacy_board.mkdir(parents=True)
    (legacy_board / "control.duckdb").write_bytes(b"board")
    legacy_queue = repo / ".git" / "agent-merge-trains" / "kept-queue"
    legacy_queue.mkdir(parents=True)
    (legacy_queue / "merge_queue.duckdb").write_bytes(b"queue")

    board = default_control_plane_root(repo)
    merge_target_queue_dir(repo, "main")
    scope = orchestration_scope_dir(repo)

    assert board == scope / "board-control-plane"
    assert (board / "control.duckdb").read_bytes() == b"board"
    assert not legacy_board.exists()
    moved_queue = scope / "agent-merge-trains" / "kept-queue" / "merge_queue.duckdb"
    assert moved_queue.read_bytes() == b"queue"
    assert not (repo / ".git" / "agent-merge-trains").exists()
    assert ".git" not in board.parts
    assert ".git" not in moved_queue.parts


def test_proof_catalog_inside_a_checkout_uses_the_account_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        ORCHESTRATION_DIR_ENV,
    )
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import (
        resolve_duckdb_path,
    )

    monkeypatch.setenv(ORCHESTRATION_DIR_ENV, str(tmp_path / "orchestration"))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "system-temp"))
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    catalog = repo / "data" / "proof_scheduler.duckdb"
    catalog.parent.mkdir()
    catalog.write_bytes(b"proof")

    target, legacy = resolve_duckdb_path(
        catalog,
        default_filename="proof_scheduler.duckdb",
        temporary_prefix="proof-scheduler-",
    )
    assert legacy is None
    assert "catalogs" in target.parts
    assert target.read_bytes() == b"proof"
    assert not catalog.exists()
    assert ".git" not in target.parts


def test_database_home_uses_the_account_home_on_each_platform(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane as catalogs

    monkeypatch.delenv(catalogs.AGENT_HOME_ENV, raising=False)
    monkeypatch.delenv(catalogs.ORCHESTRATION_DIR_ENV, raising=False)
    monkeypatch.setattr(
        catalogs.Path,
        "home",
        staticmethod(lambda: Path("/Users/ada")),
    )
    assert catalogs.agent_supervisor_home() == Path(
        "/Users/ada/.ipfs_accelerate/agent_supervisor"
    )
    assert catalogs.orchestration_state_root() == Path(
        "/Users/ada/.ipfs_accelerate/agent_supervisor/orchestration"
    )
    monkeypatch.setattr(
        catalogs.Path,
        "home",
        staticmethod(lambda: Path("C:/Users/ada")),
    )
    assert catalogs.agent_supervisor_home() == Path(
        "C:/Users/ada/.ipfs_accelerate/agent_supervisor"
    )
    assert "macOS" in catalogs.DATABASE_HOME_HELP
    assert "Windows" in catalogs.DATABASE_HOME_HELP


def test_checkout_sidecar_duckdb_moves_and_temp_files_stay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import tempfile

    from ipfs_accelerate_py.agent_supervisor.task_sources.board_control_plane import (
        ORCHESTRATION_DIR_ENV,
        repo_resident_duckdb,
    )

    monkeypatch.setenv(ORCHESTRATION_DIR_ENV, str(tmp_path / "orchestration"))
    monkeypatch.setattr(tempfile, "gettempdir", lambda: str(tmp_path / "system-temp"))
    repo = tmp_path / "repo"
    (repo / ".git").mkdir(parents=True)
    sidecar = repo / "data" / "agent_supervisor" / "lane.coordination.duckdb"
    sidecar.parent.mkdir(parents=True)
    sidecar.write_bytes(b"coord")
    wal = Path(str(sidecar) + ".wal")
    wal.write_bytes(b"wal")

    moved = repo_resident_duckdb(sidecar)
    assert "catalogs" in moved.parts
    assert ".git" not in moved.parts
    assert not str(moved).startswith(str(repo))
    assert moved.read_bytes() == b"coord"
    assert Path(str(moved) + ".wal").read_bytes() == b"wal"
    assert not sidecar.exists()
    assert not wal.exists()

    owner = repo / "control.duckdb"
    owner.write_bytes(b"owner")
    assert repo_resident_duckdb(owner) == owner

    temporary = tmp_path / "system-temp"
    temporary.mkdir()
    stayed = temporary / "stay.duckdb"
    stayed.write_bytes(b"temp")
    assert repo_resident_duckdb(stayed) == stayed
    assert stayed.read_bytes() == b"temp"

    from ipfs_accelerate_py.agent_supervisor.runtime.spar_runtime_settlement import (
        lane_paths,
    )

    state = repo / "data" / "agent_supervisor" / "spar" / "state"
    lane = state / "lane-0"
    lane.mkdir(parents=True)
    original = lane / "spar_lane_0_database_coordination.duckdb"
    original.write_bytes(b"spar")
    paths = lane_paths(state, 0)
    assert paths["directory"] == lane
    assert paths["coordination"].read_bytes() == b"spar"
    assert "catalogs" in paths["coordination"].parts
    assert not original.exists()
    assert paths["supervisor_pid"] == lane / "spar_lane_0_supervisor.pid"
    sealed = lane / "quack-lane-control.duckdb"
    sealed.write_bytes(b"vrif")
    assert repo_resident_duckdb(sealed) == sealed
