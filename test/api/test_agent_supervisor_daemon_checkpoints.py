import json
import os
import stat
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.todo_daemon import checkpoints
from ipfs_accelerate_py.agent_supervisor.todo_daemon.checkpoints import (
    CHECKPOINT_SCHEMA,
    STALE_REASONS,
    CheckpointError,
    read_checkpoint,
    resume_checkpoint,
    stale_stop,
    transition,
    write_checkpoint,
)


def _record(**overrides):
    return {
        "attempt_id": "a",
        "packet_cid": "p",
        "tree_cid": "t",
        "fence_epoch": 1,
        "effects": (),
        "obligations": (),
        **overrides,
    }


def test_legal_transitions_and_stale_stop() -> None:
    assert transition("ready", "start") == "running"
    assert transition("running", "checkpoint") == "checkpointed"
    with pytest.raises(CheckpointError):
        transition("completed", "start")
    for reason in STALE_REASONS:
        assert stale_stop(reason)["effect_after"] is False


@pytest.mark.parametrize(
    ("state", "action"), [("unknown", "start"), ("ready", "unknown")]
)
def test_unknown_state_or_action_has_typed_error(state, action) -> None:
    with pytest.raises(CheckpointError, match="unknown"):
        transition(state, action)


def test_checkpoint_roundtrip_and_corrupt_resume(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "cp.json"
    result = write_checkpoint(_record(), path)
    envelope = json.loads(path.read_bytes())
    assert envelope["schema"] == CHECKPOINT_SCHEMA
    assert envelope["record_sha256"] == result["record_sha256"]
    assert dict(read_checkpoint(path)) == _record(effects=[], obligations=[])
    assert resume_checkpoint(path)["resumed"] is True
    assert resume_checkpoint(_record())["resumed"] is True
    assert stat.S_IMODE(path.stat().st_mode) == 0o600
    with pytest.raises(CheckpointError, match="corrupt"):
        resume_checkpoint({"corrupt": True})
    assert resume_checkpoint({"stale_reason": "stale-fence"})["stopped"] is True


@pytest.mark.parametrize(
    ("binding", "value", "reason"),
    [
        ("attempt_id", "other", "stale-scope"),
        ("packet_cid", "other", "stale-plan"),
        ("tree_cid", "other", "stale-root"),
        ("fence_epoch", 2, "stale-fence"),
        ("fence_epoch", True, "stale-fence"),
    ],
)
def test_checkpoint_resume_requires_current_bindings(
    tmp_path, binding, value, reason
) -> None:
    path = tmp_path / "cp.json"
    write_checkpoint(_record(), path)
    stopped = resume_checkpoint(path, expected_bindings={binding: value})
    assert dict(stopped) == {"resumed": False, "stopped": True, "reason": reason}
    assert resume_checkpoint(path, expected_bindings={binding: _record()[binding]})[
        "resumed"
    ]


@pytest.mark.parametrize(
    "record",
    [
        {},
        _record(attempt_id=""),
        _record(packet_cid=7),
        _record(tree_cid="\x00"),
        _record(fence_epoch=True),
        _record(fence_epoch=-1),
        _record(effects="effect"),
        _record(obligations=None),
        _record(effects=[float("nan")]),
        _record(as_completion=True),
        _record(stale_reason="invented"),
    ],
)
def test_malformed_checkpoint_is_rejected_before_overwriting(tmp_path, record) -> None:
    path = tmp_path / "cp.json"
    write_checkpoint(_record(), path)
    original = path.read_bytes()
    with pytest.raises(CheckpointError):
        write_checkpoint(record, path)
    assert path.read_bytes() == original
    with pytest.raises(CheckpointError):
        resume_checkpoint(record)


def test_failed_checkpoint_replace_preserves_previous_record(
    tmp_path, monkeypatch
) -> None:
    path = tmp_path / "cp.json"
    write_checkpoint(_record(), path)
    original = path.read_bytes()

    def refuse_replace(*args):
        raise OSError("injected replace failure")

    monkeypatch.setattr(checkpoints.os, "replace", refuse_replace)
    with pytest.raises(CheckpointError, match="persistence failed"):
        write_checkpoint(_record(attempt_id="next"), path)
    assert path.read_bytes() == original
    assert list(tmp_path.iterdir()) == [path]


def test_checkpoint_flushes_file_then_directory(tmp_path, monkeypatch) -> None:
    kinds = []
    real_fsync = os.fsync

    def observe_fsync(descriptor):
        kinds.append(stat.S_IFMT(os.fstat(descriptor).st_mode))
        real_fsync(descriptor)

    monkeypatch.setattr(checkpoints.os, "fsync", observe_fsync)
    write_checkpoint(_record(), tmp_path / "cp.json")
    assert kinds == [stat.S_IFREG, stat.S_IFDIR]


@pytest.mark.parametrize(
    "payload", [b"not JSON", b"[]", b"\xff", b'{"schema":1,"schema":2}']
)
def test_unreadable_or_malformed_envelope_fails_closed(tmp_path, payload) -> None:
    path = tmp_path / "cp.json"
    path.write_bytes(payload)
    with pytest.raises(CheckpointError):
        resume_checkpoint(path)


def test_saved_checkpoint_digest_detects_state_edit(tmp_path) -> None:
    path = tmp_path / "cp.json"
    write_checkpoint(_record(), path)
    envelope = json.loads(path.read_bytes())
    envelope["record"]["fence_epoch"] = 2
    path.write_text(json.dumps(envelope), encoding="utf-8")
    with pytest.raises(CheckpointError, match="digest"):
        resume_checkpoint(path)


def test_unknown_stale_reason_and_binding_fail_closed() -> None:
    with pytest.raises(CheckpointError, match="unknown stale"):
        resume_checkpoint({"stale_reason": "stale-ish"})
    with pytest.raises(CheckpointError, match="unknown checkpoint binding"):
        resume_checkpoint(_record(), expected_bindings={"unknown": "x"})


def test_checkpoint_symlink_is_neither_read_nor_overwritten(tmp_path) -> None:
    target = tmp_path / "target.json"
    write_checkpoint(_record(), target)
    original = target.read_bytes()
    path = tmp_path / "cp.json"
    path.symlink_to(target)
    with pytest.raises(CheckpointError):
        read_checkpoint(path)
    with pytest.raises(CheckpointError, match="symlink"):
        write_checkpoint(_record(attempt_id="next"), path)
    assert target.read_bytes() == original
