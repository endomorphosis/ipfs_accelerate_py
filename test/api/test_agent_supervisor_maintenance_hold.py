"""Crashed maintenance may resume; manual, live and ambiguous holds may not."""
import errno
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import maintenance_hold as holds


@pytest.fixture
def dead_hold():
    def create(path):
        script = ("from pathlib import Path; import sys; "
                  "from ipfs_accelerate_py.agent_supervisor.rescue.maintenance_hold import create_hold; "
                  "create_hold(Path(sys.argv[1]), reason='bounded source repair', ttl_seconds=10, now=1000)")
        subprocess.run([sys.executable, "-c", script, str(path)], check=True,
                       capture_output=True, text=True, timeout=20)
        return json.loads(path.read_text())
    return create


def test_dead_owner_waits_until_deadline_and_observation_never_releases(tmp_path, dead_hold):
    marker = tmp_path / "HOLD"
    dead_hold(marker)
    assert holds.review_hold(marker, apply=True, now=1009)["reason"] == "maintenance_deadline_pending"
    assert holds.review_hold(marker, now=1010)["status"] == "release_due"
    assert marker.exists()


def test_expired_dead_owner_is_archived_with_exact_bytes_and_registration(tmp_path, dead_hold):
    marker = tmp_path / "HOLD"
    dead_hold(marker)
    before = marker.read_bytes()
    inode = marker.stat().st_ino
    binding = holds.hold_binding(marker, now=1010)
    assert binding and not holds.binding_released(binding)
    result = holds.review_hold(marker, apply=True, now=1010)
    assert result["status"] == "released"
    assert not marker.exists()
    archive = Path(result["archive"])
    assert archive.read_bytes() == before and archive.stat().st_ino == inode
    registration = json.loads(archive.with_name(archive.name + ".lease.json").read_text())
    assert registration["inode"] == inode
    assert result["completion_authority"] is False
    assert holds.binding_released(binding)
    marker.write_text("new operator stop")
    assert not holds.binding_released(binding)
    marker.unlink()
    archive.write_bytes(before + b" ")
    assert not holds.binding_released(binding)
    assert holds.review_hold(marker, apply=True, now=1020)["status"] == "absent"


def test_live_owner_is_preserved_after_expiry_and_can_release_its_own_marker(tmp_path):
    marker = tmp_path / "HOLD"
    lease = holds.create_hold(marker, reason="repair in progress", ttl_seconds=10, now=1000)
    assert holds.review_hold(marker, apply=True, now=1011)["reason"] == "maintenance_owner_alive"
    assert marker.exists()
    result = holds.review_hold(marker, apply=True, now=1005, owner_token=lease["token"])
    assert result["status"] == "released" and not marker.exists()


@pytest.mark.parametrize("body", [b"", b"manual stop", b'{"owner":"codex","at":1}',
                                  b'{"schema":"swissknife.maintenance-hold.v1"}'])
def test_manual_and_legacy_holds_are_never_adopted(tmp_path, body):
    marker = tmp_path / "HOLD"
    marker.write_bytes(body)
    marker.chmod(0o600)
    assert holds.review_hold(marker, apply=True, now=1e10)["status"] == "retained"
    assert marker.read_bytes() == body
    assert list(tmp_path.iterdir()) == [marker]
    with pytest.raises(FileExistsError):
        holds.create_hold(marker, reason="must preserve existing hold", ttl_seconds=10)


@pytest.mark.parametrize("tamper", ["inode", "body", "expiry", "symlink", "hardlink", "permissions", "registration", "archive"])
def test_changed_or_unregistered_custody_remains_held(tmp_path, dead_hold, tamper):
    marker = tmp_path / "HOLD"
    lease = dead_hold(marker)
    reserve, archive = holds._members(marker, lease["token"])
    if tamper == "inode":
        raw = marker.read_bytes(); marker.unlink(); marker.write_bytes(raw); marker.chmod(0o600)
    elif tamper == "body":
        marker.write_bytes(marker.read_bytes() + b" ")
    elif tamper == "expiry":
        lease["expires_at"] = 1001
        marker.write_text(json.dumps(lease))
    elif tamper == "symlink":
        marker.unlink(); marker.symlink_to(reserve)
    elif tamper == "hardlink":
        os.link(marker, tmp_path / "foreign")
    elif tamper == "permissions":
        marker.chmod(0o644)
    elif tamper == "registration":
        holds._registration(reserve).unlink()
    else:
        archive.write_text("conflicting archive"); archive.chmod(0o600)
    assert holds.review_hold(marker, apply=True, now=1010)["status"] == "retained"
    assert os.path.lexists(marker)


def test_different_pid_namespace_cannot_prove_owner_absence(tmp_path, dead_hold, monkeypatch):
    marker = tmp_path / "HOLD"
    lease = dead_hold(marker)
    current = {**lease["owner"], "pid": os.getpid(), "pid_namespace": "pid:[foreign]"}
    monkeypatch.setattr(holds, "_identity", lambda pid: current if pid == os.getpid() else None)
    result = holds.review_hold(marker, apply=True, now=1010)
    assert result["reason"] == "maintenance_owner_namespace_unverified"
    assert marker.exists()


def test_reused_pid_does_not_retain_a_dead_owners_pause(tmp_path, dead_hold, monkeypatch):
    marker = tmp_path / "HOLD"
    lease = dead_hold(marker)
    real = holds._identity
    monkeypatch.setattr(holds, "_identity", lambda pid: {**lease["owner"], "start_ticks": "different-birth"}
                        if pid == lease["owner"]["pid"] else real(pid))
    assert holds.review_hold(marker, apply=True, now=1010)["status"] == "released"


def test_storage_failure_preserves_the_stop_marker(tmp_path, dead_hold, monkeypatch):
    marker = tmp_path / "HOLD"
    dead_hold(marker)
    def full(*args, **kwargs):
        raise OSError(errno.ENOSPC, "full")
    monkeypatch.setattr(holds.os, "link", full)
    assert holds.review_hold(marker, apply=True, now=1010)["status"] == "retained"
    assert marker.exists()


def test_release_resumes_after_crash_following_archive_link(tmp_path, dead_hold):
    marker = tmp_path / "HOLD"
    lease = dead_hold(marker)
    reserve, archive = holds._members(marker, lease["token"])
    os.link(reserve, archive)
    assert holds.review_hold(marker, apply=True, now=1010)["status"] == "released"
    assert archive.exists() and not marker.exists()


def test_context_releases_even_when_maintenance_raises(tmp_path):
    marker = tmp_path / "HOLD"
    with pytest.raises(RuntimeError, match="repair failed"):
        with holds.maintenance_hold(marker, reason="test source repair", ttl_seconds=30):
            assert marker.exists()
            raise RuntimeError("repair failed")
    assert not marker.exists()


def test_fleet_hold_review_releases_only_the_registered_maintenance_marker(tmp_path, dead_hold):
    from ipfs_accelerate_py.agent_supervisor.rescue.fleet_holds import review_one_hold
    marker = tmp_path / "HOLD"
    dead_hold(marker)
    assert review_one_hold(str(marker), {}, {})["status"] == "released"
    marker.write_text("operator pause")
    assert review_one_hold(str(marker), {}, {})["status"] == "retained"
    assert marker.read_text() == "operator pause"


@pytest.mark.parametrize("ttl", [0, -1, 1801, float("nan"), float("inf"), True, 10**1000])
def test_invalid_deadlines_never_publish_a_marker(tmp_path, ttl):
    marker = tmp_path / "HOLD"
    with pytest.raises(ValueError):
        holds.create_hold(marker, reason="repair", ttl_seconds=ttl)
    assert not marker.exists()


@pytest.mark.parametrize("kind", ["pending", "dead_expired", "live_expired", "owner_token"])
def test_readonly_review_never_creates_or_opens_a_writable_lock(tmp_path, dead_hold, monkeypatch, kind):
    marker = tmp_path / "HOLD"
    lease = dead_hold(marker) if kind == "dead_expired" else holds.create_hold(
        marker, reason="readonly maintenance review", ttl_seconds=10, now=1000)
    lock = marker.with_name(".HOLD.maintenance.lock")
    lock.unlink()
    before = {p.name: (p.stat().st_ino, p.read_bytes()) for p in tmp_path.iterdir()}
    original_open = holds.os.open
    reads = []
    def readonly_open(path, flags, *args, **kwargs):
        assert not flags & (os.O_WRONLY | os.O_RDWR | os.O_CREAT | os.O_TRUNC | os.O_APPEND)
        reads.append(str(path))
        return original_open(path, flags, *args, **kwargs)
    def forbidden(*args, **kwargs):
        pytest.fail("readonly review reached a lock or filesystem mutation")
    with monkeypatch.context() as guard:
        guard.setattr(holds.os, "open", readonly_open)
        guard.setattr(holds, "_lock", forbidden)
        guard.setattr(holds, "_write_new", forbidden)
        guard.setattr(holds, "_sync", forbidden)
        guard.setattr(holds.fcntl, "flock", forbidden)
        for name in ("link", "unlink", "remove", "rename", "replace", "mkdir", "rmdir", "chmod", "chown", "utime", "fsync"):
            guard.setattr(holds.os, name, forbidden)
        result = holds.review_hold(marker, apply=False,
            now=1011 if kind in {"dead_expired", "live_expired"} else 1005,
            owner_token=lease["token"] if kind == "owner_token" else None)
    assert reads
    assert result["status"] == ("release_due" if kind in {"dead_expired", "owner_token"} else "retained")
    assert result["completion_authority"] is False
    assert before == {p.name: (p.stat().st_ino, p.read_bytes()) for p in tmp_path.iterdir()}
    assert not lock.exists()
