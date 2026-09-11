"""Real configured-board preflight -> native maintenance -> preflight closure."""

import json
from pathlib import Path
import pytest
from test.api.test_agent_supervisor_configured_board_scheduler import (
    _seed_configured_repo,
    _git,
)
from ipfs_accelerate_py.agent_supervisor.runtime import (
    accepted_submodule_maintenance as maintenance,
)
from ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler import (
    load_configured_board,
    preflight_configured_board,
)
from ipfs_accelerate_py.agent_supervisor.merge import accepted_submodule_sync as sync


@pytest.fixture
def board_fixture(tmp_path):
    repo, config = _seed_configured_repo(tmp_path)
    payload = json.loads(config.read_text())
    source = payload["source_binding"]
    source["ipfs_datasets_submodule_path"] = source.pop("dependency_submodule_path")
    source["ipfs_datasets_planning_revision"] = source.pop(
        "dependency_planning_revision"
    )
    config.write_text(json.dumps(payload, indent=2) + "\n")
    _git(repo, "add", "config/scheduler.json")
    _git(repo, "commit", "-m", "declared datasets role")
    child = repo / "dependency"
    old = _git(child, "rev-parse", "HEAD").stdout.strip()
    _git(child, "config", "user.email", "fixture@example.invalid")
    _git(child, "config", "user.name", "Fixture")
    (child / "dependency.txt").write_text("accepted successor\n")
    _git(child, "add", ".")
    _git(child, "commit", "-m", "accepted successor")
    target = _git(child, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "add", "dependency")
    _git(repo, "commit", "-m", "accept dependency gitlink")
    _git(child, "checkout", "--detach", old)
    return repo, config, child, old, target, tmp_path / "outside-archive"


def run(fixture, guard=lambda: None):
    root, config, child, old, target, archive = fixture
    board = load_configured_board(config, repo_root=root)
    return maintenance.maintain_accepted_configured_submodule(
        board, archive_root=archive, custody_guard=guard
    )


def test_actual_current_configured_preflight_syncs_and_restart_is_noop(board_fixture):
    root, config, child, old, target, archive = board_fixture
    before = preflight_configured_board(load_configured_board(config, repo_root=root))
    assert before["valid"] is False
    assert {x["name"] for x in before["checks"] if not x["passed"]} == {
        "checkout_clean",
        "configured_submodules",
    }
    result = run(board_fixture)
    assert result["changed"] is True and result["preflight"]["valid"] is True
    assert sync.detached_at(child, target)
    journal = json.loads(Path(result["journal"]).read_text())
    assert (
        journal["success"] is True
        and journal["historical_callback_closure_claimed"] is False
    )
    assert (Path(result["journal"]).parent / "submodule-original.index").is_file()
    second = run(board_fixture)
    assert second["changed"] is False
    assert len(list(archive.rglob("journal.json"))) == 1


@pytest.mark.parametrize(
    "change",
    [
        "dirty_parent",
        "dirty_child",
        "attached",
        "wrong_role",
        "runtime_role",
        "validator",
        "provider",
        "lock",
    ],
)
def test_actual_preflight_denies_nonqualifying_repairs(board_fixture, change):
    root, config, child, old, target, archive = board_fixture
    if change == "dirty_parent":
        (root / "docs/plan.md").write_text("unaccepted plan")
    if change == "dirty_child":
        (child / "dependency.txt").write_text("provider edits")
    if change == "attached":
        _git(child, "checkout", "-b", "provider-branch")
    if change in ("wrong_role", "runtime_role"):
        payload = json.loads(config.read_text())
        source = payload["source_binding"]
        source["other_submodule_path"] = source.pop("ipfs_datasets_submodule_path")
        source["other_planning_revision"] = source.pop(
            "ipfs_datasets_planning_revision"
        )
        if change == "runtime_role":
            source["ipfs_accelerate_submodule_path"] = source["other_submodule_path"]
        config.write_text(json.dumps(payload) + "\n")
        _git(root, "add", "config/scheduler.json")
        _git(root, "commit", "-m", "other configured role")
    if change == "validator":
        (root / "scripts/validate_board.py").write_text(
            'import json; print(json.dumps({"valid":False,"errors":["scope denied"]}))'
        )
        _git(root, "add", "scripts/validate_board.py")
        _git(root, "commit", "-m", "validation denial")
    lock = sync.git_index_path(child).with_name("index.lock")
    if change == "lock":
        lock.write_bytes(b"foreign lock")

    def guard():
        if change == "provider":
            raise sync.Refused("exact_provider_present")

    with pytest.raises((sync.Refused, FileExistsError)):
        run(board_fixture, guard)
    assert _git(child, "rev-parse", "HEAD").stdout.strip() == old
    if change == "lock":
        assert lock.read_bytes() == b"foreign lock"
    else:
        assert not archive.exists()


def test_revalidation_failure_preserves_original_index_and_recorded_partial_state(
    board_fixture, monkeypatch
):
    root, config, child, old, target, archive = board_fixture
    original = sync.synchronize_accepted_submodule

    def racing(*args, **kwargs):
        result = original(*args, **kwargs)
        _git(root, "commit", "--allow-empty", "-m", "concurrent accepted parent")
        return result

    monkeypatch.setattr(sync, "synchronize_accepted_submodule", racing)
    with pytest.raises(sync.Refused, match="accepted_parent_changed"):
        run(board_fixture)
    journal = json.loads(next(archive.rglob("journal.json")).read_text())
    assert journal["success"] is False and journal["changed"] is True
    assert journal["error_type"] == "Refused"
    assert sync.detached_at(child, target)


def test_config_drift_after_assessment_is_refused_before_archive(
    board_fixture, monkeypatch
):
    root, config, child, old, target, archive = board_fixture
    native = maintenance.assess_accepted_configured_submodule

    def raced(board):
        result = native(board)
        config.write_text(config.read_text() + "\n")
        return result

    monkeypatch.setattr(maintenance, "assess_accepted_configured_submodule", raced)
    with pytest.raises(sync.Refused, match="configured_maintenance_config_changed"):
        run(board_fixture)
    assert not archive.exists() and sync.detached_at(child, old)


def test_pure_assessment_creates_no_state_or_archive(board_fixture):
    root, config, child, old, target, archive = board_fixture
    before = sync.git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
    value = maintenance.assess_accepted_configured_submodule(
        load_configured_board(config, repo_root=root)
    )
    assert value["needed"] is True and not archive.exists()
    assert (
        sync.git_text(root, "status", "--porcelain=v1", "--untracked-files=all")
        == before
    )


def test_preexisting_archive_leaf_is_never_reused(board_fixture, monkeypatch):
    from types import SimpleNamespace

    root, config, child, old, target, archive = board_fixture
    board = load_configured_board(config, repo_root=root)
    directory = archive / board.board_namespace / "accepted-submodule-collision"
    directory.mkdir(parents=True)
    (directory / "journal.json").write_bytes(b"foreign journal")
    monkeypatch.setattr(
        maintenance.uuid, "uuid4", lambda: SimpleNamespace(hex="collision")
    )
    with pytest.raises(FileExistsError):
        run(board_fixture)
    assert (directory / "journal.json").read_bytes() == b"foreign journal"
    assert sync.detached_at(child, old)


def test_archive_symlink_parent_refused_without_source_mutation(
    board_fixture, tmp_path
):
    root, config, child, old, target, archive = board_fixture
    destination = tmp_path / "foreign"
    destination.mkdir()
    archive.symlink_to(destination, target_is_directory=True)
    with pytest.raises(OSError):
        run(board_fixture)
    assert not list(destination.iterdir()) and sync.detached_at(child, old)
