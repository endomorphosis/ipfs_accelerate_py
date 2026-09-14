"""Population refusal stays strict; only bounded rejection evidence is added."""

import dataclasses
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import sawm_writer_recovery as r


def foreign_expected():
    binding = dataclasses.asdict(r.process.observe_process(os.getpid()))
    binding.update(pid=99999990, process_group=99999990, session=99999990)
    return {"owner": binding, "controller": binding, "lanes": [], "markers": {}}


def test_real_unknown_child_remains_denied_with_hash_only_evidence(tmp_path):
    child = subprocess.Popen(
        [
            sys.executable,
            "-I",
            "-S",
            "-c",
            "import time;time.sleep(30)",
            "synthetic-private-value",
        ],
        cwd=tmp_path,
        start_new_session=True,
    )
    try:
        for _ in range(100):
            args = (Path("/proc") / str(child.pid) / "cmdline").read_bytes()
            if b"synthetic-private-value" in args:
                break
            time.sleep(0.01)
        with pytest.raises(
            r.NativeScopeProcessObserved, match="additional_native_scope_process"
        ) as exc:
            r.scoped_census(tmp_path, foreign_expected())
        evidence = exc.value.diagnostic
        actual = r.process.observe_process(child.pid)
        assert evidence["pid"] == actual.pid
        assert evidence["birth"] == actual.birth
        assert evidence["parent"] == actual.parent
        assert evidence["process_group"] == actual.process_group
        assert evidence["session"] == actual.session
        assert evidence["cwd"] == str(tmp_path)
        assert evidence["argv_sha256"] == hashlib.sha256(args).hexdigest()
        assert evidence["scope_flags"]["cwd_under_root"] is True
        assert evidence["stable_identity"] is True
        assert evidence["callback_settlement_authority"] is False
        assert "synthetic-private-value" not in json.dumps(evidence)
    finally:
        child.terminate()
        child.wait(timeout=3)


@pytest.mark.parametrize(
    "changed",
    ["birth", "parent", "group", "session", "argv", "cwd", "exited", "unavailable"],
)
def test_recheck_drift_preserves_refusal_and_marks_identity_unverified(
    tmp_path, monkeypatch, changed
):
    pid = os.getpid()
    before = r.process._stat(pid)
    args = r.process._read_proc(Path("/proc") / str(pid) / "cmdline")
    cwd = Path(os.readlink(Path("/proc") / str(pid) / "cwd"))
    after = list(before)
    positions = {"birth": 19, "parent": 1, "group": 2, "session": 3}
    if changed in positions:
        after[positions[changed]] = str(int(after[positions[changed]]) + 1)
    elif changed == "exited":
        after[0] = "Z"

    def observation(_pid):
        if changed == "unavailable":
            raise PermissionError("not emitted")
        return after

    monkeypatch.setattr(r.process, "_stat", observation)
    if changed == "argv":
        monkeypatch.setattr(r.process, "_read_proc", lambda p: b"changed")
    if changed == "cwd":
        monkeypatch.setattr(r.os, "readlink", lambda p: str(tmp_path))
    refusal = r._scope_refusal(pid, before, cwd, args, {"cwd_under_root": True})
    assert isinstance(refusal, r.process.GracefulRecoveryUnverified)
    assert str(refusal) == "additional_native_scope_process"
    assert refusal.diagnostic["stable_identity"] is False
    assert refusal.diagnostic["callback_settlement_authority"] is False
