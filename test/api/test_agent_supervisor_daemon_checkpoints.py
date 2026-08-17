import pytest
from pathlib import Path
from ipfs_accelerate_py.agent_supervisor.todo_daemon.checkpoints import (
    STALE_REASONS, transition, write_checkpoint, resume_checkpoint, stale_stop, CheckpointError,
)

def test_legal_transitions_and_stale_stop() -> None:
    assert transition("ready", "start") == "running"
    assert transition("running", "checkpoint") == "checkpointed"
    with pytest.raises(CheckpointError):
        transition("completed", "start")
    for reason in STALE_REASONS:
        assert stale_stop(reason)["effect_after"] is False

def test_checkpoint_roundtrip_and_corrupt_resume(tmp_path: Path) -> None:
    path = tmp_path / "cp.txt"
    write_checkpoint({"attempt_id": "a", "packet_cid": "p", "tree_cid": "t", "fence_epoch": 1, "effects": (), "obligations": ()}, path)
    assert path.is_file()
    with pytest.raises(CheckpointError, match="corrupt"):
        resume_checkpoint({"corrupt": True})
    stopped = resume_checkpoint({"stale_reason": "stale-fence"})
    assert stopped["stopped"] is True
