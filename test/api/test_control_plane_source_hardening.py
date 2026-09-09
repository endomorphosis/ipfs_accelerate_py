"""Umask repair follows the capsule closure without changing other files."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_implementation_route import (
    _AGENT_CONTROL_PLANE_RELATIVE_FILES,
    _agent_control_plane_source_files,
)
from test.api.test_agent_supervisor_configured_typed_grant_handoff import aseh_operator


def _source(root: Path, relative: str, mode: int = 0o664) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("# fixture\n", encoding="utf-8")
    path.chmod(mode)
    return path


def test_hardening_covers_complete_python_capsule_closure(tmp_path: Path) -> None:
    for relative in _AGENT_CONTROL_PLANE_RELATIVE_FILES:
        _source(tmp_path, relative)
    for relative in (
        "ipfs_accelerate_py/agent_supervisor/runtime/shared_hashing.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/hash_observations.py",
        "ipfs_accelerate_py/agent_supervisor/future/deep/member.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/sql/9999_future.sql",
    ):
        _source(tmp_path, relative)
    excluded = [
        _source(tmp_path, relative)
        for relative in (
            "unrelated.py",
            "ipfs_accelerate_py/unrelated.py",
            "ipfs_accelerate_py/utils/unrelated.py",
            "scripts/ops/agent_supervisor/unrelated.py",
            "ipfs_accelerate_py/agent_supervisor/runtime/notes.txt",
        )
    ]
    closure = _agent_control_plane_source_files(tmp_path, verify_loaded_origins=False)
    original = {path: path.read_bytes() for path in (*closure, *excluded)}

    aseh_operator._strip_control_plane_group_other_write(tmp_path)

    for path in closure:
        expected = 0o644 if path.suffix == ".py" else 0o664
        assert stat.S_IMODE(path.stat().st_mode) == expected, path
    for path in excluded:
        assert stat.S_IMODE(path.stat().st_mode) == 0o664, path
    assert {path: path.read_bytes() for path in original} == original


@pytest.mark.parametrize("mode", [0o644, 0o600, 0o775, 0o666, 0o777])
def test_hardening_only_removes_group_other_write(tmp_path: Path, mode: int) -> None:
    path = _source(tmp_path, "ipfs_accelerate_py/router_deps.py", mode)
    aseh_operator._strip_control_plane_group_other_write(tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == mode & ~0o022


def test_hardening_does_not_change_foreign_owned_files(tmp_path: Path, monkeypatch) -> None:
    path = _source(tmp_path, "ipfs_accelerate_py/router_deps.py")
    other_uid = os.geteuid() + 1
    monkeypatch.setattr(aseh_operator.os, "geteuid", lambda: other_uid)
    aseh_operator._strip_control_plane_group_other_write(tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o664


def test_hardening_does_not_change_hard_linked_files(tmp_path: Path) -> None:
    path = _source(tmp_path, "ipfs_accelerate_py/router_deps.py")
    alias = tmp_path / "unrelated.py"
    os.link(path, alias)
    aseh_operator._strip_control_plane_group_other_write(tmp_path)
    assert stat.S_IMODE(alias.stat().st_mode) == 0o664


def test_hardening_does_not_follow_required_file_symlink(tmp_path: Path) -> None:
    target = _source(tmp_path, "unrelated.py")
    path = tmp_path / "ipfs_accelerate_py" / "router_deps.py"
    path.parent.mkdir()
    path.symlink_to(target)
    aseh_operator._strip_control_plane_group_other_write(tmp_path)
    assert stat.S_IMODE(target.stat().st_mode) == 0o664


def test_hardening_rejects_supervisor_tree_symlink(tmp_path: Path) -> None:
    target = _source(tmp_path, "unrelated.py")
    path = tmp_path / "ipfs_accelerate_py" / "agent_supervisor" / "linked.py"
    path.parent.mkdir(parents=True)
    path.symlink_to(target)
    with pytest.raises(aseh_operator.OperatorError, match="source closure"):
        aseh_operator._strip_control_plane_group_other_write(tmp_path)
    assert stat.S_IMODE(target.stat().st_mode) == 0o664


def test_hardening_rejects_symlinked_ancestor(tmp_path: Path) -> None:
    target = _source(tmp_path, "outside/meta_model_api.py")
    package = tmp_path / "ipfs_accelerate_py"
    package.mkdir()
    (package / "common").symlink_to(target.parent, target_is_directory=True)
    with pytest.raises(aseh_operator.OperatorError, match="hardening failed"):
        aseh_operator._strip_control_plane_group_other_write(tmp_path)
    assert stat.S_IMODE(target.stat().st_mode) == 0o664


def test_hardening_rejects_file_replaced_before_open(tmp_path: Path, monkeypatch) -> None:
    path = _source(tmp_path, "ipfs_accelerate_py/router_deps.py")
    replacement = _source(tmp_path, "replacement.py")
    real_open = os.open

    def replace_then_open(name, flags, *args, **kwargs):
        if name == path.name:
            replacement.replace(path)
        return real_open(name, flags, *args, **kwargs)

    monkeypatch.setattr(aseh_operator.os, "open", replace_then_open)
    with pytest.raises(aseh_operator.OperatorError, match="source file changed"):
        aseh_operator._strip_control_plane_group_other_write(tmp_path)
    assert stat.S_IMODE(path.stat().st_mode) == 0o664


def test_hardening_closes_descriptor_on_chmod_failure(tmp_path: Path, monkeypatch) -> None:
    path = _source(tmp_path, "ipfs_accelerate_py/router_deps.py")
    descriptors = []

    def fail_chmod(descriptor, mode):
        descriptors.append(descriptor)
        raise OSError("fixture fchmod failure")

    monkeypatch.setattr(aseh_operator.os, "fchmod", fail_chmod)
    with pytest.raises(aseh_operator.OperatorError, match="hardening failed"):
        aseh_operator._strip_control_plane_group_other_write(tmp_path)
    assert len(descriptors) == 1
    with pytest.raises(OSError):
        os.fstat(descriptors[0])
    assert stat.S_IMODE(path.stat().st_mode) == 0o664
