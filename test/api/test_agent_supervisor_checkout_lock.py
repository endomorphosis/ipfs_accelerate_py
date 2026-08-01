from __future__ import annotations

import json
import os
from pathlib import Path
import threading
import time

import pytest

from ipfs_accelerate_py.agent_supervisor.merge import (
    checkout_lock as checkout_lock_module,
)
from ipfs_accelerate_py.agent_supervisor.merge.checkout_lock import (
    acquire_checkout_mutation_lease,
    checkout_lock_metadata,
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


def test_release_treats_confirmed_absent_lock_as_released(
    tmp_path: Path,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    metadata = _metadata(tmp_path, operation="already-released")
    lease, _reason, _incumbent, _waited = acquire_checkout_mutation_lease(
        lock_path,
        metadata,
        owner_active=lambda _owner: True,
    )
    assert lease is not None
    lock_path.unlink()

    assert release_checkout_mutation_lease(lease)
    assert not lock_path.exists()


@pytest.mark.parametrize(
    "replacement_record",
    [
        "",
        "{not-json\n",
        "[]\n",
        "{}\n",
    ],
)
def test_release_preserves_malformed_or_legacy_replacement(
    tmp_path: Path,
    replacement_record: str,
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
    replacement_path.write_text(replacement_record, encoding="utf-8")
    os.replace(replacement_path, lock_path)

    assert not release_checkout_mutation_lease(lease)
    assert lock_path.read_text(encoding="utf-8") == replacement_record


def test_release_preserves_existing_lock_when_read_is_inconclusive(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lock_path = tmp_path / "implementation-main-merge.lock"
    metadata = _metadata(tmp_path, operation="original-owner")
    lease, _reason, _incumbent, _waited = acquire_checkout_mutation_lease(
        lock_path,
        metadata,
        owner_active=lambda _owner: True,
    )
    assert lease is not None
    replacement_record = b"inconclusive replacement\n"
    replacement_path = tmp_path / "replacement.pending"
    replacement_path.write_bytes(replacement_record)
    os.replace(replacement_path, lock_path)
    monkeypatch.setattr(
        checkout_lock_module,
        "_read_checkout_lock",
        lambda _lock_path: (None, None),
    )

    assert not release_checkout_mutation_lease(lease)
    assert lock_path.read_bytes() == replacement_record


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
