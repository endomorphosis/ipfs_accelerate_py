"""CLI startup ownership fences never operate on real watchdog processes."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import supervisor_watchdog as watchdog_module
from ipfs_accelerate_py.agent_supervisor.rescue import watchdog_guardian as guardian_module


@pytest.fixture
def startup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    manifest = tmp_path / "manifest.json"
    manifest.write_text('{"lanes": []}')
    calls = []

    class StubWatchdog:
        def __init__(self, **kwargs):
            calls.append(("construct", kwargs))

        def run(self):
            calls.append(("run", None))
            # Ownership must remain locked throughout the live run.
            assert guardian_module.acquire_lock(manifest.parent / "watchdog-owner.lock") is None
            return {"work_complete": False}

    monkeypatch.setattr(watchdog_module, "SupervisorWatchdog", StubWatchdog)
    monkeypatch.setattr(guardian_module, "discover_watchdogs", lambda *args, **kwargs: ([], []))
    argv = ["--manifest-path", str(manifest), "--repo-root", str(tmp_path)]
    return manifest, argv, calls


def test_held_owner_lock_denies_startup_before_constructing_watchdog(startup):
    manifest, argv, calls = startup
    lock = guardian_module.acquire_lock(manifest.parent / "watchdog-owner.lock")
    assert lock is not None
    with lock:
        assert watchdog_module.main(argv) == 0
    assert calls == []


def test_existing_legacy_owner_without_lock_denies_startup(startup, monkeypatch):
    _manifest, argv, calls = startup
    monkeypatch.setattr(guardian_module, "discover_watchdogs", lambda *args: ([{"pid": 1249858}], []))
    assert watchdog_module.main(argv) == 0
    assert calls == []


def test_verified_absence_runs_under_lock_and_releases_it(startup):
    manifest, argv, calls = startup
    assert watchdog_module.main(argv) == 0
    assert [name for name, _ in calls] == ["construct", "run"]
    assert calls[0][1]["manifest_path"] == manifest.resolve()
    released = guardian_module.acquire_lock(manifest.parent / "watchdog-owner.lock")
    assert released is not None
    released.close()


def test_uncertain_discovery_denies_startup(startup, monkeypatch):
    _manifest, argv, calls = startup
    monkeypatch.setattr(guardian_module, "discover_watchdogs", lambda *args: ([], ["identity_changed:123"]))
    assert watchdog_module.main(argv) == 0
    assert calls == []


def test_discovery_ignores_only_the_current_starting_process(startup, monkeypatch):
    _manifest, argv, calls = startup
    monkeypatch.setattr(guardian_module, "discover_watchdogs", lambda *args: ([{"pid": os.getpid()}], []))
    assert watchdog_module.main(argv) == 0
    assert [name for name, _ in calls] == ["construct", "run"]


def test_run_failure_releases_the_owner_lock(startup, monkeypatch):
    manifest, argv, _calls = startup

    class FailingWatchdog:
        def __init__(self, **kwargs):
            pass

        def run(self):
            raise RuntimeError("synthetic failure")

    monkeypatch.setattr(watchdog_module, "SupervisorWatchdog", FailingWatchdog)
    with pytest.raises(RuntimeError, match="synthetic failure"):
        watchdog_module.main(argv)
    released = guardian_module.acquire_lock(manifest.parent / "watchdog-owner.lock")
    assert released is not None
    released.close()
