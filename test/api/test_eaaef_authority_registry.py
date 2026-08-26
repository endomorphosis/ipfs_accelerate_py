from __future__ import annotations

import concurrent.futures
import json
import os
import pwd
import stat
import subprocess
import sys
import threading
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.validation import (
    eaaef_authority_registry as registry_module,
)
from ipfs_accelerate_py.agent_supervisor.validation.eaaef_authority_registry import (
    EAAEF_AUTHORITY_PRODUCT_ROOT,
    EAAEF_LOGICAL_AUTHORITY_PREFIX,
    EAAEFAuthorityConflict,
    EAAEFAuthorityNotFound,
    EAAEFAuthorityRegistry,
    EAAEFAuthorityRegistryError,
)


def _logical(name: str) -> str:
    return (EAAEF_LOGICAL_AUTHORITY_PREFIX / name).as_posix()


def _registry(tmp_path: Path) -> EAAEFAuthorityRegistry:
    return EAAEFAuthorityRegistry(authority_root=tmp_path / "state" / "registry")


def test_default_root_is_stable_for_the_effective_account_across_environment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected = (
        Path(pwd.getpwuid(os.geteuid()).pw_dir) / ".local/state" / EAAEF_AUTHORITY_PRODUCT_ROOT
    )
    monkeypatch.setenv("HOME", str(tmp_path / "sealed-duckdb-home"))
    monkeypatch.setenv("XDG_STATE_HOME", str(tmp_path / "xdg"))
    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_STATE_HOME",
        str(tmp_path / "platform-override"),
    )
    first_registry = EAAEFAuthorityRegistry()
    monkeypatch.delenv("HOME", raising=False)
    monkeypatch.delenv("XDG_STATE_HOME", raising=False)
    monkeypatch.delenv("IPFS_ACCELERATE_AGENT_STATE_HOME", raising=False)
    second_registry = EAAEFAuthorityRegistry()

    assert first_registry.root == expected
    assert second_registry.root == expected


def test_isolated_child_environment_resolves_the_same_account_registry(
    tmp_path: Path,
) -> None:
    repository_root = Path(__file__).resolve().parents[2]
    expected = (
        Path(pwd.getpwuid(os.geteuid()).pw_dir) / ".local/state" / EAAEF_AUTHORITY_PRODUCT_ROOT
    )
    code = (
        "import sys; "
        f"sys.path.insert(0, {str(repository_root)!r}); "
        "from ipfs_accelerate_py.agent_supervisor.validation."
        "eaaef_authority_registry import EAAEFAuthorityRegistry; "
        "print(EAAEFAuthorityRegistry().root)"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-B", "-c", code],
        check=False,
        capture_output=True,
        text=True,
        env={
            "HOME": str(tmp_path / "repository-internal-home"),
            "PATH": "/usr/bin:/bin",
            "TZ": "UTC",
        },
        timeout=30,
    )

    assert completed.returncode == 0, completed.stderr
    assert Path(completed.stdout.strip()) == expected


def test_logical_prefix_maps_beneath_registry_without_recreating_repo_prefix(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    logical = _logical("host-admission/final.json")

    assert registry.physical_path(logical) == (
        tmp_path / "state/registry/host-admission/final.json"
    )
    assert registry.physical_path(EAAEF_LOGICAL_AUTHORITY_PREFIX.as_posix()) == (
        tmp_path / "state/registry"
    )
    for unsafe in (
        "authority/final.json",
        f"/{logical}",
        f"{EAAEF_LOGICAL_AUTHORITY_PREFIX}/../outside.json",
        f"{EAAEF_LOGICAL_AUTHORITY_PREFIX}//outside.json",
        f"{EAAEF_LOGICAL_AUTHORITY_PREFIX}\\outside.json",
    ):
        with pytest.raises(EAAEFAuthorityRegistryError):
            registry.physical_path(unsafe)


def test_authority_root_inside_checkout_is_rejected(tmp_path: Path) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()

    with pytest.raises(EAAEFAuthorityRegistryError, match="outside the repository"):
        EAAEFAuthorityRegistry(
            repo_root=checkout,
            authority_root=checkout / "private-authority",
        )


def test_publish_is_private_immutable_and_exactly_idempotent(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    logical = _logical("host-admission/final.json")
    payload = {"z": [2, 1], "a": "bound"}

    physical = registry.publish_json(logical, payload)
    assert registry.publish_json(logical, payload) == physical
    assert registry.read_json(logical) == payload
    assert physical.read_bytes() == b'{"a":"bound","z":[2,1]}\n'
    assert stat.S_IMODE((tmp_path / "state").stat().st_mode) == 0o700
    assert stat.S_IMODE((tmp_path / "state/registry").stat().st_mode) == 0o700
    assert stat.S_IMODE(physical.parent.stat().st_mode) == 0o700
    assert stat.S_IMODE(physical.stat().st_mode) == 0o400
    assert physical.stat().st_nlink == 1
    assert stat.S_IMODE((tmp_path / "state/registry/.registry.lock").stat().st_mode) == 0o600

    with pytest.raises(EAAEFAuthorityConflict):
        registry.publish_json(logical, {"a": "different"})
    assert registry.read_json(logical) == payload


def test_publication_fsyncs_created_directories_and_artifact_parent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path)
    synchronized_directories: list[tuple[int, int]] = []
    real_fsync = os.fsync

    def tracking_fsync(descriptor: int) -> None:
        observed = os.fstat(descriptor)
        if stat.S_ISDIR(observed.st_mode):
            synchronized_directories.append((int(observed.st_dev), int(observed.st_ino)))
        real_fsync(descriptor)

    monkeypatch.setattr(registry_module.os, "fsync", tracking_fsync)
    physical = registry.publish_json(
        _logical("durable/nested/final.json"),
        {"durable": True},
    )

    parent = physical.parent.stat()
    assert (int(parent.st_dev), int(parent.st_ino)) in synchronized_directories


def test_multi_artifact_ceremony_is_reentrant(tmp_path: Path) -> None:
    registry = _registry(tmp_path)

    with registry.ceremony():
        registry.publish_json(_logical("ceremony/first.json"), {"sequence": 1})
        registry.publish_json(_logical("ceremony/second.json"), {"sequence": 2})
        assert registry.read_json(_logical("ceremony/first.json")) == {"sequence": 1}


def test_same_instance_reader_waits_for_atomic_multi_artifact_ceremony(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    first = _logical("same-instance/first.json")
    second = _logical("same-instance/second.json")
    reader_started = threading.Event()
    reader_lock_attempted = threading.Event()
    underlying_mutex = registry._mutex

    class ObservedMutex:
        def acquire(self) -> bool:
            if threading.current_thread().name.startswith("registry-reader"):
                reader_lock_attempted.set()
            return underlying_mutex.acquire()

        def release(self) -> None:
            underlying_mutex.release()

    registry._mutex = ObservedMutex()  # type: ignore[assignment]

    def read_complete_ceremony() -> tuple[dict[str, object], dict[str, object]]:
        reader_started.set()
        return registry.read_json(first), registry.read_json(second)

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=1,
        thread_name_prefix="registry-reader",
    ) as executor:
        with registry.ceremony():
            registry.publish_json(first, {"sequence": 1})
            future = executor.submit(read_complete_ceremony)
            assert reader_started.wait(timeout=5)
            assert reader_lock_attempted.wait(timeout=5)
            assert not future.done()
            registry.publish_json(second, {"sequence": 2})

        assert future.result(timeout=5) == (
            {"sequence": 1},
            {"sequence": 2},
        )


def test_registry_flock_serializes_conflicting_publishers(tmp_path: Path) -> None:
    root = tmp_path / "state/registry"
    logical = _logical("contention/final.json")
    barrier = threading.Barrier(2)

    def publish(value: int) -> tuple[str, int]:
        candidate = EAAEFAuthorityRegistry(authority_root=root)
        barrier.wait()
        try:
            candidate.publish_json(logical, {"value": value})
        except EAAEFAuthorityConflict:
            return ("conflict", value)
        return ("published", value)

    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(publish, (1, 2)))

    assert sorted(result for result, _value in outcomes) == ["conflict", "published"]
    winner = next(value for result, value in outcomes if result == "published")
    assert EAAEFAuthorityRegistry(authority_root=root).read_json(logical) == {"value": winner}


def test_secure_read_rejects_mode_hardlink_and_symlink_tampering(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    logical = _logical("tamper/final.json")
    physical = registry.publish_json(logical, {"admitted": True})

    physical.chmod(0o600)
    with pytest.raises(EAAEFAuthorityRegistryError, match="single-link mode-0400"):
        registry.read_json(logical)
    physical.chmod(0o400)

    alias = physical.with_name("alias.json")
    os.link(physical, alias)
    with pytest.raises(EAAEFAuthorityRegistryError, match="single-link mode-0400"):
        registry.read_json(logical)
    alias.unlink()

    outside = tmp_path / "outside.json"
    outside.write_text(json.dumps({"forged": True}), encoding="utf-8")
    outside.chmod(0o400)
    physical.unlink()
    physical.symlink_to(outside)
    with pytest.raises((EAAEFAuthorityRegistryError, OSError)):
        registry.read_json(logical)


def test_secure_read_translates_a_missing_parent_to_typed_not_found(
    tmp_path: Path,
) -> None:
    registry = _registry(tmp_path)
    registry.publish_json(_logical("seed.json"), {"seed": True})

    with pytest.raises(EAAEFAuthorityNotFound, match="parent does not exist"):
        registry.read_json(_logical("missing/parent/final.json"))


def test_secure_read_rejects_duplicate_json_object_keys(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.publish_json(_logical("seed.json"), {"seed": True})
    duplicate = registry.physical_path(_logical("duplicate.json"))
    duplicate.write_bytes(b'{"decision":"admitted","decision":"no_go"}\n')
    duplicate.chmod(0o400)

    with pytest.raises(EAAEFAuthorityRegistryError, match="duplicate JSON object key"):
        registry.read_json(_logical("duplicate.json"))


def test_parent_walk_never_follows_symlink(tmp_path: Path) -> None:
    registry = _registry(tmp_path)
    registry.publish_json(_logical("seed.json"), {"seed": True})
    outside = tmp_path / "outside"
    outside.mkdir()
    (registry.root / "escape").symlink_to(outside, target_is_directory=True)

    with pytest.raises((EAAEFAuthorityRegistryError, OSError)):
        registry.publish_json(_logical("escape/forged.json"), {"forged": True})
    assert not (outside / "forged.json").exists()


def test_failed_direct_fallback_never_unlinks_a_racing_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    registry = _registry(tmp_path)
    logical = _logical("fallback/final.json")
    physical = registry.physical_path(logical)
    replacement = b'{"replacement":true}\n'
    monkeypatch.setattr(registry_module, "_open_anonymous_file", lambda _fd: None)

    def replace_then_fail(_descriptor: int, _data: bytes) -> None:
        physical.unlink()
        physical.write_bytes(replacement)
        physical.chmod(0o400)
        raise OSError("simulated interrupted direct write")

    monkeypatch.setattr(registry_module, "_write_all", replace_then_fail)
    with pytest.raises(OSError, match="interrupted direct write"):
        registry.publish_json(logical, {"original": True})

    assert physical.read_bytes() == replacement


def test_payload_size_is_bounded_before_filesystem_effects(tmp_path: Path) -> None:
    registry = EAAEFAuthorityRegistry(
        authority_root=tmp_path / "state/registry",
        max_json_bytes=32,
    )

    with pytest.raises(EAAEFAuthorityRegistryError, match="bounded size"):
        registry.publish_json(_logical("too-large.json"), {"value": "x" * 100})
    assert not registry.root.exists()
