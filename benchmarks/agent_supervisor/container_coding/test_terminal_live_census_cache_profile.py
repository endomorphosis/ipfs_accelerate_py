"""Authored 20-GiB cache prerequisites; no credential or container executes."""
import asyncio
import os
from pathlib import Path
from types import SimpleNamespace

import pytest

import benchmarks.agent_supervisor.container_coding.terminal_setup_cache_advice as cache


PROFILE = "source384-5cpu-20gib-planner180@1"
SELECTION = {"schema": "terminal-setup-cache-selection@1",
             "policy": "source384-native-aarch64-dontneed@2", "manifest_sha256": "a" * 64}


def _forbidden(*args, **kwargs):
    pytest.fail("cache prerequisite validation must not read credentials or execute")


def _validate(auth, **changes):
    options = dict(install_codex=True, auth_json=auth, arm="full", resource_profile=PROFILE)
    return cache.validate_setup_cache_prerequisites(dict(SELECTION), **{**options, **changes})


def test_explicit_20gib_profile_accepts_v2_cache_with_auth_metadata_only(tmp_path, monkeypatch):
    auth = tmp_path / "unread-auth.json"
    auth.write_bytes(b"AUTHORED PRIVATE FIXTURE MUST NEVER BE READ")
    auth.chmod(0o600)
    before = auth.lstat()
    monkeypatch.setattr(cache.platform, "machine", lambda: "aarch64")
    with monkeypatch.context() as guard:
        for name in ("open", "read_text", "read_bytes"):
            guard.setattr(Path, name, _forbidden)
        assert _validate(auth) is None
    after = auth.lstat()
    assert (before.st_ino, before.st_size, before.st_mtime_ns, before.st_ctime_ns) == (
        after.st_ino, after.st_size, after.st_mtime_ns, after.st_ctime_ns)
    assert set(cache.RESOURCE_PROFILES) == {
        "source384-5cpu-12gib@1", "source384-5cpu-16gib-extended@1", PROFILE}


@pytest.mark.parametrize("profile", [None, "unknown@1", "source384-5cpu-16gib-planner180@1",
    "source384-5cpu-20gib-coding600@1", "source384-5cpu-20gib-planner180@2"])
def test_new_allowance_does_not_admit_other_resource_profiles_before_auth_stat(monkeypatch, profile):
    monkeypatch.setattr(cache.platform, "machine", lambda: "aarch64")
    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", _forbidden)
        with pytest.raises(ValueError, match="common resources"):
            _validate(Path("/authored/not-opened"), resource_profile=profile)


@pytest.mark.parametrize("change,architecture", [
    ({"arm": "no-index"}, "aarch64"), ({"install_codex": False}, "aarch64"),
    ({"install_codex": 1}, "aarch64"), ({}, "x86_64"),
])
def test_v2_cache_keeps_full_codex_and_aarch64_guards_before_auth_stat(monkeypatch, change, architecture):
    monkeypatch.setattr(cache.platform, "machine", lambda: architecture)
    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", _forbidden)
        with pytest.raises(ValueError, match="aarch64 Codex"):
            _validate(Path("/authored/not-opened"), **change)


def test_20gib_cache_still_requires_explicit_auth_before_stat(monkeypatch):
    monkeypatch.setattr(cache.platform, "machine", lambda: "aarch64")
    with monkeypatch.context() as guard:
        guard.setattr(Path, "lstat", _forbidden)
        with pytest.raises(ValueError, match="explicit supported auth"):
            _validate(None)


@pytest.mark.parametrize("kind", ["missing", "directory", "empty", "oversized", "symlink", "hardlink"])
def test_20gib_cache_auth_guard_rejects_non_singleton_bounded_regular_files(
        tmp_path, monkeypatch, kind):
    auth = tmp_path / "unread-auth.json"
    if kind == "directory":
        auth.mkdir()
    elif kind != "missing":
        auth.write_bytes(b"" if kind == "empty" else b"x" * (65537 if kind == "oversized" else 8))
        if kind == "symlink":
            alias = tmp_path / "linked-auth.json"
            alias.symlink_to(auth)
            auth = alias
        elif kind == "hardlink":
            os.link(auth, tmp_path / "second-auth-link.json")
    monkeypatch.setattr(cache.platform, "machine", lambda: "aarch64")
    with monkeypatch.context() as guard:
        for name in ("open", "read_text", "read_bytes"):
            guard.setattr(Path, name, _forbidden)
        with pytest.raises(ValueError, match="auth file"):
            _validate(auth)


@pytest.mark.parametrize("profile,expected_error", [(PROFILE, "authored-boundary-receipt"),
    ("source384-5cpu-20gib-coding600@1", "unknown setup cache resource profile")])
def test_apply_uses_same_resource_allowance_before_upload_or_execution(
        tmp_path, monkeypatch, profile, expected_error):
    # Reaching a protected public receipt is distinct from applying advice.
    # Existing transport tests qualify its complete pinned asset population.
    monkeypatch.setattr(cache, "validate_setup_cache_selection", lambda *args: (
        {"archive_sha256": "b" * 64}, {"codex_version": "0.160.0"}))
    reads = []

    def boundary_receipt(path, maximum):
        reads.append((path, maximum))
        raise ValueError("authored-boundary-receipt")

    monkeypatch.setattr(cache, "_regular_bytes", boundary_receipt)
    environment = SimpleNamespace(upload_file=_forbidden, exec=_forbidden)
    output = tmp_path / "unused-advice"
    with pytest.raises(ValueError, match=expected_error):
        asyncio.run(cache.apply_setup_cache_advice(environment, archive_dir=tmp_path / "bundle",
            expected=dict(SELECTION), boundary_output=tmp_path / "boundary", output=output,
            resource_profile=profile))
    assert len(reads) == int(profile == PROFILE)
    if reads:
        assert reads[0] == (tmp_path / "boundary/installation/native-codex-binary.log", cache.MAX_RECEIPT)
    assert not output.exists()
