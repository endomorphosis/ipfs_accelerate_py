"""Legacy relocation must preserve explicitly owned live campaign custody."""
import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import board_control_plane as board


@pytest.fixture
def legacy_home(tmp_path, monkeypatch):
    monkeypatch.setattr(board.Path, "home", staticmethod(lambda: tmp_path))
    monkeypatch.setenv(board.AGENT_HOME_ENV, str(tmp_path / "canonical"))
    monkeypatch.delenv(board.ORCHESTRATION_DIR_ENV, raising=False)
    monkeypatch.setattr(board, "_LEGACY_PLATFORM_ADOPTED", False)
    return tmp_path / ".local/state/ipfs_accelerate_py", tmp_path / "canonical/legacy-platform-state"


def test_managed_campaign_keeps_all_sidecars_while_normal_legacy_is_adopted(legacy_home):
    old, adopted = legacy_home
    campaign = old / "managed-campaign"
    board.register_managed_state_root(campaign, owner_id="campaign:test")
    retained = []
    for relative in ("campaign/execution.duckdb", "campaign/coordination.duckdb", "campaign/merge-queue/merge_queue.duckdb", "campaign/native/attempt/portal-state.duckdb"):
        path = campaign / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
        retained.append(path)
    legacy = old / "ordinary/old.duckdb"
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(b"legacy")
    assert board.adopt_legacy_platform_databases() == 1
    assert all(path.exists() for path in retained)
    assert not (adopted / "managed-campaign").exists()
    assert not legacy.exists()
    assert (adopted / "ordinary/old.duckdb").read_bytes() == b"legacy"


def test_marker_is_exact_root_bound_and_cannot_be_reused_for_another_owner(tmp_path):
    root = tmp_path / "owned"
    marker = board.register_managed_state_root(root, owner_id="owner:a")
    assert board.register_managed_state_root(root, owner_id="owner:a") == marker
    with pytest.raises(board.BoardControlPlaneError, match="another owner"):
        board.register_managed_state_root(root, owner_id="owner:b")
    other = tmp_path / "other"
    other.mkdir()
    (other / board.MANAGED_STATE_ROOT_MARKER).write_bytes(marker.read_bytes())
    assert not board.is_managed_state_root(other)


def test_symlink_and_malformed_markers_do_not_disable_normal_adoption(legacy_home):
    old, adopted = legacy_home
    valid = old / "valid"
    marker = board.register_managed_state_root(valid, owner_id="owner:a")
    for name in ("symlink", "malformed"):
        directory = old / name
        directory.mkdir()
        target = directory / board.MANAGED_STATE_ROOT_MARKER
        if name == "symlink":
            target.symlink_to(marker)
        else:
            target.write_text(json.dumps({"schema": board.MANAGED_STATE_ROOT_SCHEMA}))
        (directory / "legacy.duckdb").write_bytes(name.encode())
    assert board.adopt_legacy_platform_databases() == 2
    assert (adopted / "symlink/legacy.duckdb").exists()
    assert (adopted / "malformed/legacy.duckdb").exists()
