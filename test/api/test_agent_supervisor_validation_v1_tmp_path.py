"""Portable owning regression for the trusted validation worker's v1 tmp_path route."""

import json
import os
import stat


def test_v1_tmp_path_keeps_regular_private_files(tmp_path):
    from ipfs_accelerate_py.agent_supervisor.rescue import repair_authorization

    assert tmp_path.is_absolute() and not tmp_path.is_symlink()
    assert stat.S_ISDIR(tmp_path.lstat().st_mode)
    assert tmp_path.lstat().st_uid == os.getuid()
    assert stat.S_IMODE(tmp_path.lstat().st_mode) == 0o700
    payload = {"schema": "validation-v1-private-file-regression/v1", "ordinary": True}
    state = tmp_path / "state.json"
    state.write_bytes(repair_authorization.canonical(payload))
    info = state.lstat()
    assert stat.S_ISREG(info.st_mode) and info.st_nlink == 1 and info.st_uid == os.getuid()
    assert not state.is_symlink() and json.loads(state.read_bytes()) == payload
    assert sorted(path.name for path in tmp_path.iterdir()) == ["state.json"]
