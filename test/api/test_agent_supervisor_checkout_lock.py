from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import (
    checkout_lock as checkout_lock_module,
)
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    adopt_inactive_checkout_mutation_lease,
    acquire_checkout_mutation_lease,
    checkout_lock_metadata,
    checkout_lock_owner_is_active,
    checkout_mutation_lock_path,
    checkout_repository_id,
    release_checkout_mutation_lease,
    remove_inactive_checkout_mutation_lock,
    serialized_lock_update,
)


def _metadata(
    repo_root: Path,
    *,
    operation: str,
) -> dict[str, object]:
    return checkout_lock_metadata(
        kind="implementation-main-merge",
        repo_root=repo_root,
        owner_script="test-checkout-lock.py",
        extra={"operation": operation},
    )


def _pending_files(lock_path: Path) -> list[Path]:
    return list(
        lock_path.parent.glob(f".{lock_path.name}.*.pending")
    )


def _git(repo: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", *args],
        cwd=repo,
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr or result.stdout
    return result.stdout.strip()


def _seed_git_repository(path: Path) -> Path:
    path.mkdir()
    _git(path, "init")
    _git(path, "checkout", "-b", "main")
    _git(path, "config", "user.name", "Test User")
    _git(path, "config", "user.email", "test@example.invalid")
    (path / "seed.txt").write_text("seed\n", encoding="utf-8")
    _git(path, "add", "seed.txt")
    _git(path, "commit", "-m", "seed")
    return path


def _owner_is_active(metadata: dict[str, object], repo_root: Path) -> bool:
    return checkout_lock_owner_is_active(
        metadata,
        expected_kind="implementation-main-merge",
        expected_repo_root=repo_root,
        process_command_line=lambda _pid: "test-checkout-lock.py",
        process_is_running=lambda pid: pid == os.getpid(),
    )


def _legacy_exact_path_owner_is_active(
    metadata: dict[str, object],
    repo_root: Path,
) -> bool:
    """Model the pre-migration exact-path predicate still running in peers."""

    legacy_repo_root = str(metadata.get("repo_root") or "")
    if (
        legacy_repo_root
        and Path(legacy_repo_root).resolve() != repo_root.resolve()
    ):
        return False
    return int(metadata.get("pid") or 0) == os.getpid()


def test_sibling_worktree_preserves_and_cannot_replace_live_checkout_lease(
    tmp_path: Path,
) -> None:
    primary = _seed_git_repository(tmp_path / "primary")
    sibling = tmp_path / "sibling"
    _git(primary, "worktree", "add", "-b", "sibling", str(sibling))
    foreign = _seed_git_repository(tmp_path / "foreign")

    lock_path = checkout_mutation_lock_path(primary)
    assert lock_path.resolve() == checkout_mutation_lock_path(sibling).resolve()
    assert checkout_repository_id(primary) == checkout_repository_id(sibling)

    metadata = _metadata(primary, operation="original-owner")
    assert metadata["repo_root"] == ""
    assert metadata["worktree_root"] == str(primary.resolve())
    assert metadata["repository_id"] == checkout_repository_id(primary)
    spoofed_identity = checkout_lock_metadata(
        kind="implementation-main-merge",
        repo_root=primary,
        owner_script="test-checkout-lock.py",
        extra={
            "repo_root": str(foreign.resolve()),
            "worktree_root": str(foreign.resolve()),
            "repository_id": checkout_repository_id(foreign),
        },
    )
    assert spoofed_identity["repo_root"] == ""
    assert spoofed_identity["worktree_root"] == str(primary.resolve())
    assert spoofed_identity["repository_id"] == checkout_repository_id(
        primary
    )
    # This is the exact predicate used by pre-migration daemons. An empty
    # legacy path lets them reach the live-PID check during a rolling upgrade.
    assert _legacy_exact_path_owner_is_active(
        metadata,
        sibling,
    )

    lease, reason, incumbent, _waited = acquire_checkout_mutation_lease(
        lock_path,
        metadata,
        owner_active=lambda owner: _owner_is_active(owner, primary),
    )
    assert lease is not None
    assert reason == "acquired"
    assert incumbent is None
    assert _owner_is_active(metadata, sibling)

    sibling_metadata = _metadata(sibling, operation="sibling-contender")
    contender, reason, incumbent, _waited = (
        acquire_checkout_mutation_lease(
            lock_path,
            sibling_metadata,
            owner_active=lambda owner: _owner_is_active(owner, sibling),
        )
    )
    assert contender is None
    assert reason == "lock_exists"
    assert incumbent == metadata
    assert not remove_inactive_checkout_mutation_lock(
        lock_path,
        expected_metadata=metadata,
        owner_active=lambda owner: _owner_is_active(owner, sibling),
    )
    assert (
        adopt_inactive_checkout_mutation_lease(
            lease,
            sibling_metadata,
            owner_active=lambda owner: _owner_is_active(owner, sibling),
        )
        is None
    )
    assert json.loads(lock_path.read_text(encoding="utf-8")) == metadata

    # Legacy records with a concrete sibling path compare by Git common-dir,
    # while a conclusively different physical repository is rejected.
    legacy_metadata = dict(metadata)
    legacy_metadata.pop("repository_id")
    legacy_metadata.pop("worktree_root")
    legacy_metadata["repo_root"] = str(primary.resolve())
    assert _owner_is_active(legacy_metadata, sibling)
    assert not _owner_is_active(metadata, foreign)
    assert not _owner_is_active(legacy_metadata, foreign)

    # Missing repository authority is inconclusive and therefore preserved,
    # even when the process probe cannot establish liveness.
    uncertain_metadata = {
        "kind": "implementation-main-merge",
        "pid": 0,
        "repo_root": "",
        "worktree_root": "",
        "repository_id": "",
    }
    assert checkout_lock_owner_is_active(
        uncertain_metadata,
        expected_kind="implementation-main-merge",
        expected_repo_root=sibling,
        process_command_line=lambda _pid: "",
        process_is_running=lambda _pid: False,
    )

    assert release_checkout_mutation_lease(lease)
    assert not lock_path.exists()


def test_atomic_publication_fsyncs_and_captures_identity_before_link(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    metadata = _metadata(tmp_path, operation="atomic-publication")
    real_fstat = os.fstat
    real_fsync = os.fsync
    real_link = os.link
    observed: dict[str, object] = {
        "fsynced": False,
        "identity_captured": False,
    }

    def fsync(fd: int) -> None:
        real_fsync(fd)
        observed["fsynced"] = True

    def fstat(fd: int):
        stat_result = real_fstat(fd)
        observed["identity_captured"] = True
        observed["source_identity"] = (
            int(stat_result.st_dev),
            int(stat_result.st_ino),
        )
        return stat_result

    def link(source: Path, destination: Path) -> None:
        assert observed["fsynced"] is True
        assert observed["identity_captured"] is True
        assert Path(source).parent == lock_path.parent
        real_link(source, destination)
        # The destination appears in one step with complete JSON.  There is no
        # observable O_EXCL-empty/write interval.
        assert json.loads(Path(destination).read_text(encoding="utf-8")) == metadata

    monkeypatch.setattr(checkout_lock_module.os, "fsync", fsync)
    monkeypatch.setattr(checkout_lock_module.os, "fstat", fstat)
    monkeypatch.setattr(checkout_lock_module.os, "link", link)

    lease, reason, incumbent, _waited = acquire_checkout_mutation_lease(
        lock_path,
        metadata,
        owner_active=lambda _owner: True,
    )

    assert lease is not None
    assert reason == "acquired"
    assert incumbent is None
    assert (lease.device, lease.inode) == observed["source_identity"]
    destination_stat = lock_path.stat(follow_symlinks=False)
    assert (lease.device, lease.inode) == (
        int(destination_stat.st_dev),
        int(destination_stat.st_ino),
    )
    assert _pending_files(lock_path) == []
    assert release_checkout_mutation_lease(lease)


def test_publication_failure_cleans_same_directory_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    metadata = _metadata(tmp_path, operation="publication-failure")

    def fail_link(_source: Path, _destination: Path) -> None:
        raise OSError("simulated hard-link failure")

    monkeypatch.setattr(checkout_lock_module.os, "link", fail_link)

    lease, reason, incumbent, _waited = acquire_checkout_mutation_lease(
        lock_path,
        metadata,
        owner_active=lambda _owner: True,
        timeout_seconds=0.2,
    )

    assert lease is None
    assert reason == "lock_publication_failed"
    assert incumbent is None
    assert not lock_path.exists()
    assert _pending_files(lock_path) == []


@pytest.mark.parametrize(
    "legacy_record",
    [
        "",
        "{not-json\n",
        "[]\n",
        "{}\n",
    ],
)
def test_incomplete_or_legacy_lock_is_never_stolen(
    tmp_path: Path,
    legacy_record: str,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    lock_path.write_text(legacy_record, encoding="utf-8")
    metadata = _metadata(tmp_path, operation="contender")

    lease, reason, _incumbent, waited = acquire_checkout_mutation_lease(
        lock_path,
        metadata,
        owner_active=lambda _owner: pytest.fail(
            "legacy records must not enter owner liveness validation"
        ),
        timeout_seconds=0.025,
        poll_seconds=0.002,
    )

    assert lease is None
    assert reason == "lock_exists"
    assert waited >= 0.015
    assert lock_path.read_text(encoding="utf-8") == legacy_record
    assert _pending_files(lock_path) == []


def test_exact_inactive_parsed_legacy_lock_is_reclaimed(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    legacy = {
        "kind": "merge",
        "pid": 999999,
        "operation": "legacy-owner",
    }
    lock_path.write_text(json.dumps(legacy), encoding="utf-8")
    replacement = _metadata(tmp_path, operation="replacement-owner")

    lease, reason, cleared, _waited = acquire_checkout_mutation_lease(
        lock_path,
        replacement,
        owner_active=lambda owner: owner != legacy,
        timeout_seconds=0.2,
        poll_seconds=0.002,
    )

    assert lease is not None
    assert reason == "acquired"
    assert cleared == legacy
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement
    assert release_checkout_mutation_lease(lease)


def test_exact_inactive_parsed_lease_is_reclaimed(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    stale = _metadata(tmp_path, operation="stale-owner")
    lock_path.write_text(
        json.dumps(stale, sort_keys=True),
        encoding="utf-8",
    )
    replacement = _metadata(tmp_path, operation="replacement-owner")
    checked: list[dict[str, object]] = []

    lease, reason, cleared, _waited = acquire_checkout_mutation_lease(
        lock_path,
        replacement,
        owner_active=lambda owner: checked.append(owner) or False,
        timeout_seconds=0.2,
        poll_seconds=0.002,
    )

    assert lease is not None
    assert reason == "acquired"
    assert cleared == stale
    assert checked == [stale]
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement
    assert release_checkout_mutation_lease(lease)


def test_stale_cleanup_preserves_replacement_installed_during_liveness_probe(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    stale = _metadata(tmp_path, operation="stale-owner")
    replacement = _metadata(tmp_path, operation="replacement-owner")
    lock_path.write_text(json.dumps(stale), encoding="utf-8")
    replacement_path = tmp_path / "replacement.pending"
    replacement_path.write_text(json.dumps(replacement), encoding="utf-8")

    def replace_during_probe(_owner: dict[str, object]) -> bool:
        os.replace(replacement_path, lock_path)
        return False

    assert not remove_inactive_checkout_mutation_lock(
        lock_path,
        expected_metadata=stale,
        owner_active=replace_during_probe,
    )
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement


def test_release_requires_acquired_inode_and_preserves_replacement(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    metadata = _metadata(tmp_path, operation="original-owner")
    lease, _reason, _incumbent, _waited = acquire_checkout_mutation_lease(
        lock_path,
        metadata,
        owner_active=lambda _owner: True,
    )
    assert lease is not None
    replacement_path = tmp_path / "replacement.pending"
    replacement_path.write_text(json.dumps(metadata), encoding="utf-8")
    replacement_stat = replacement_path.stat(follow_symlinks=False)
    assert (
        int(replacement_stat.st_dev),
        int(replacement_stat.st_ino),
    ) != (lease.device, lease.inode)
    os.replace(replacement_path, lock_path)

    assert not release_checkout_mutation_lease(lease)
    assert json.loads(lock_path.read_text(encoding="utf-8")) == metadata


def test_release_is_idempotent_when_exact_lease_is_already_absent(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    metadata = _metadata(tmp_path, operation="already-released-owner")
    lease, _reason, _incumbent, _waited = acquire_checkout_mutation_lease(
        lock_path,
        metadata,
        owner_active=lambda _owner: True,
    )
    assert lease is not None

    lock_path.unlink()

    assert release_checkout_mutation_lease(lease)
    assert not lock_path.exists()


def test_release_requires_acquired_lease_id_on_same_inode(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    metadata = _metadata(tmp_path, operation="original-owner")
    lease, _reason, _incumbent, _waited = acquire_checkout_mutation_lease(
        lock_path,
        metadata,
        owner_active=lambda _owner: True,
    )
    assert lease is not None
    replacement = {
        **metadata,
        "lease_id": "replacement-lease-id",
    }
    lock_path.write_text(json.dumps(replacement), encoding="utf-8")
    current_stat = lock_path.stat(follow_symlinks=False)
    assert (int(current_stat.st_dev), int(current_stat.st_ino)) == (
        lease.device,
        lease.inode,
    )

    assert not release_checkout_mutation_lease(lease)
    assert json.loads(lock_path.read_text(encoding="utf-8")) == replacement


def test_serialized_update_guard_contention_is_bounded(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    completed = threading.Event()
    result: dict[str, object] = {}

    def contend() -> None:
        started = time.monotonic()
        try:
            with serialized_lock_update(
                lock_path,
                timeout_seconds=0.04,
                poll_seconds=0.002,
            ):
                result["entered"] = True
        except Exception as exc:
            result["exception"] = exc
        result["elapsed"] = time.monotonic() - started
        completed.set()

    with serialized_lock_update(lock_path):
        worker = threading.Thread(target=contend)
        worker.start()
        assert completed.wait(timeout=0.5)

    worker.join(timeout=1)
    assert isinstance(result.get("exception"), TimeoutError)
    assert "entered" not in result
    assert 0.02 <= float(result["elapsed"]) < 0.2


def test_acquisition_timeout_budget_includes_stale_cleanup_guard(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    stale = _metadata(tmp_path, operation="stale-owner")
    lock_path.write_text(json.dumps(stale), encoding="utf-8")
    contender = _metadata(tmp_path, operation="contender")
    completed = threading.Event()
    result: dict[str, object] = {}

    def acquire() -> None:
        result["value"] = acquire_checkout_mutation_lease(
            lock_path,
            contender,
            owner_active=lambda _owner: False,
            timeout_seconds=0.04,
            poll_seconds=0.002,
        )
        completed.set()

    with serialized_lock_update(lock_path):
        started = time.monotonic()
        worker = threading.Thread(target=acquire)
        worker.start()
        assert completed.wait(timeout=0.5)
        elapsed = time.monotonic() - started

    worker.join(timeout=1)
    lease, reason, incumbent, waited = result["value"]
    assert lease is None
    assert reason == "lock_exists"
    assert incumbent == stale
    assert 0.02 <= waited < 0.2
    assert elapsed < 0.2
    assert json.loads(lock_path.read_text(encoding="utf-8")) == stale
    assert _pending_files(lock_path) == []
