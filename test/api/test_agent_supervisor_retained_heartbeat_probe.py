"""Current configured daemon metadata can withhold recovery without task DB I/O."""
import json

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import live_board_probe as probe


def fixture(tmp_path):
    path = tmp_path / "declared-heartbeat.json"
    birth = {"pid": 100, "start_time_ticks": 1234, "boot_id": "test-boot"}
    payload = {"schema": "ipfs_accelerate_py/agent-supervisor/database-daemon-pass-heartbeat@1",
               "selection_idle_reason": "expired_attempt_settlement_unavailable",
               "completed_at": 1000, "process_birth": birth,
               "completion_authority": True, "task_count": 999, "write_count": 42}
    path.write_text(json.dumps(payload))
    board = {"max_lanes": 1, "daemon_pass_heartbeat_paths": [str(path)]}
    return path, payload, board, birth


def test_configured_current_heartbeat_reports_only_denial_metadata(tmp_path):
    _, _, board, birth = fixture(tmp_path)
    result = probe.read_retained_pass_heartbeat(board, 0, birth, 1001)
    assert result == {"selection_idle_reason": "expired_attempt_settlement_unavailable",
                      "settlement_observation_source": "configured_daemon_pass_heartbeat",
                      "settlement_heartbeat_age_seconds": 1.0}


@pytest.mark.parametrize("change", [
    {"schema": "another-format"}, {"completed_at": 500}, {"completed_at": 1002},
    {"completed_at": None}, {"process_birth": {}},
    {"process_birth": {"pid": 101, "start_time_ticks": 1234, "boot_id": "test-boot"}},
    {"selection_idle_reason": "no_ready_tasks"},
])
def test_stale_or_wrong_daemon_heartbeat_does_not_report_current_wait(tmp_path, change):
    path, payload, board, birth = fixture(tmp_path)
    payload.update(change)
    path.write_text(json.dumps(payload))
    assert probe.read_retained_pass_heartbeat(board, 0, birth, 1001) == {}


@pytest.mark.parametrize("paths", [None, [], ["relative.json"], ["/a", "/b"], "not-a-list"])
def test_only_explicit_exact_lane_inventory_is_read(tmp_path, monkeypatch, paths):
    _, _, board, birth = fixture(tmp_path)
    board["daemon_pass_heartbeat_paths"] = paths
    def forbidden(*a, **kw):
        pytest.fail("invalid inventory reached file access")
    monkeypatch.setattr(probe, "read_json", forbidden)
    assert probe.read_retained_pass_heartbeat(board, 0, birth, 1001) == {}


def test_missing_or_symlinked_heartbeat_cannot_report_current_wait(tmp_path):
    path, _, board, birth = fixture(tmp_path)
    path.rename(tmp_path / "target")
    assert probe.read_retained_pass_heartbeat(board, 0, birth, 1001) == {}
    path.symlink_to(tmp_path / "target")
    assert probe.read_retained_pass_heartbeat(board, 0, birth, 1001) == {}
