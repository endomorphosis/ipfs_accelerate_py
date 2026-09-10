"""Exercise verification reuse against real Linux descriptor and seal changes."""

from __future__ import annotations

import fcntl
import hashlib
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path

import pytest

from ipfs_accelerate_py import agent_implementation_route as route
from test.api.test_agent_supervisor_native_dependency_pin import (
    _inspect,
    _rebound_pin,
    _seal,
    _write_source,
)


pytestmark = pytest.mark.skipif(
    not hasattr(os, "memfd_create"), reason="Linux kernel memfd seals required"
)


@pytest.fixture(autouse=True)
def empty_cache():
    route._agent_clear_immutable_verifications()
    yield
    route._agent_clear_immutable_verifications()


def _memfd(payload: bytes, *, omitted_seal: int = 0) -> int:
    descriptor = os.memfd_create("seal-cache-test", os.MFD_ALLOW_SEALING)
    os.write(descriptor, payload)
    os.fchmod(descriptor, 0o400)
    fcntl.fcntl(
        descriptor, fcntl.F_ADD_SEALS,
        route._agent_native_required_seals() & ~omitted_seal,
    )
    return descriptor


def _digest(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def _verify(descriptor: int, payload: bytes) -> None:
    route._agent_verify_immutable_descriptor(
        descriptor, expected_sha256=_digest(payload), maximum_bytes=1024 * 1024
    )


def _count_reads(monkeypatch):
    actual = os.pread
    counts = {"bytes": 0, "calls": 0}

    def counted(descriptor, size, offset):
        chunk = actual(descriptor, size, offset)
        counts["bytes"] += len(chunk)
        counts["calls"] += 1
        return chunk

    monkeypatch.setattr(os, "pread", counted)
    return counts


def test_repeated_and_parallel_verification_reads_payload_once(monkeypatch):
    payload = b"stable archive" * 4096
    descriptor = _memfd(payload)
    counts = _count_reads(monkeypatch)
    try:
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: _verify(descriptor, payload), range(12)))
        assert counts["bytes"] == len(payload)
        entry = next(iter(route._AGENT_IMMUTABLE_VERIFICATIONS.values()))
        assert entry.descriptor != descriptor
        assert not os.get_inheritable(entry.descriptor)
        with pytest.raises(ValueError, match="digest differs"):
            _verify(descriptor, b"forged expected digest")
    finally:
        os.close(descriptor)


def test_waiting_cold_verification_does_not_block_gate_owner_cache_hit(monkeypatch):
    from ipfs_accelerate_py import _hash_resources

    warm = _memfd(b"warm")
    cold = _memfd(b"cold")
    _verify(warm, b"warm")
    actual_lock = _hash_resources.hashing_lock
    waiter = threading.Event()

    @contextmanager
    def observed_lock(**kwargs):
        waiter.set()
        # Bound failure of the previous inverted resource/cache lock order.
        with actual_lock(**kwargs, timeout=0.5):
            yield

    monkeypatch.setattr(_hash_resources, "hashing_lock", observed_lock)
    try:
        with ThreadPoolExecutor(max_workers=1) as executor:
            with actual_lock(kind="outer-sealing", exclusive=True):
                future = executor.submit(_verify, cold, b"cold")
                assert waiter.wait(2)
                _verify(warm, b"warm")
            future.result(timeout=2)
    finally:
        os.close(warm)
        os.close(cold)


@pytest.mark.parametrize(
    "omitted", [fcntl.F_SEAL_WRITE, fcntl.F_SEAL_GROW,
                fcntl.F_SEAL_SHRINK, fcntl.F_SEAL_SEAL]
)
def test_each_required_kernel_seal_is_checked(omitted):
    descriptor = _memfd(b"payload", omitted_seal=omitted)
    try:
        with pytest.raises(ValueError, match="fully sealed"):
            _verify(descriptor, b"payload")
        assert not route._AGENT_IMMUTABLE_VERIFICATIONS
    finally:
        os.close(descriptor)


def test_read_only_regular_file_never_becomes_cached_authority(tmp_path):
    path = tmp_path / "mutable"
    path.write_bytes(b"payload")
    path.chmod(0o400)
    with path.open("rb") as source:
        with pytest.raises((OSError, ValueError)):
            _verify(source.fileno(), b"payload")
    assert not route._AGENT_IMMUTABLE_VERIFICATIONS


def test_closed_fd_reused_for_different_sealed_payload_is_rejected():
    original = _memfd(b"first payload")
    replacement = _memfd(b"other payload")
    try:
        _verify(original, b"first payload")
        os.dup2(replacement, original)
        with pytest.raises(ValueError, match="digest differs"):
            _verify(original, b"first payload")
    finally:
        os.close(original)
        os.close(replacement)


def test_descriptor_replaced_during_first_read_is_not_cached(monkeypatch):
    original = _memfd(b"first payload")
    replacement = _memfd(b"other payload")
    actual = os.pread

    def replace_during_read(descriptor, size, offset):
        chunk = actual(descriptor, size, offset)
        os.dup2(replacement, original)
        return chunk

    monkeypatch.setattr(os, "pread", replace_during_read)
    try:
        with pytest.raises(ValueError, match="identity changed"):
            _verify(original, b"first payload")
        assert not route._AGENT_IMMUTABLE_VERIFICATIONS
    finally:
        os.close(original)
        os.close(replacement)


def test_lru_closes_evicted_duplicates_and_enforces_byte_budget(monkeypatch):
    monkeypatch.setattr(route, "_AGENT_IMMUTABLE_VERIFICATION_MAX_ENTRIES", 2)
    monkeypatch.setattr(route, "_AGENT_IMMUTABLE_VERIFICATION_MAX_BYTES", 8)
    descriptors = [_memfd(value) for value in (b"1111", b"2222", b"3333")]
    try:
        _verify(descriptors[0], b"1111")
        first = next(iter(route._AGENT_IMMUTABLE_VERIFICATIONS.values()))
        _verify(descriptors[1], b"2222")
        _verify(descriptors[2], b"3333")
        assert len(route._AGENT_IMMUTABLE_VERIFICATIONS) == 2
        assert sum(item.size_bytes for item in route._AGENT_IMMUTABLE_VERIFICATIONS.values()) == 8
        with pytest.raises(OSError):
            os.fstat(first.descriptor)
        route._agent_clear_immutable_verifications()
        assert not route._AGENT_IMMUTABLE_VERIFICATIONS
    finally:
        for descriptor in descriptors:
            os.close(descriptor)


def test_child_discards_parent_verification_and_duplicate_descriptors():
    descriptor = _memfd(b"payload")
    _verify(descriptor, b"payload")
    held = next(iter(route._AGENT_IMMUTABLE_VERIFICATIONS.values())).descriptor
    child = os.fork()
    if child == 0:
        try:
            assert not route._AGENT_IMMUTABLE_VERIFICATIONS
            with pytest.raises(OSError):
                os.fstat(held)
            _verify(descriptor, b"payload")
        except BaseException:
            os._exit(1)
        os._exit(0)
    try:
        _, status = os.waitpid(child, 0)
        assert os.waitstatus_to_exitcode(status) == 0
        os.fstat(held)
        assert len(route._AGENT_IMMUTABLE_VERIFICATIONS) == 1
    finally:
        os.close(descriptor)


def test_control_plane_public_verifier_reuses_digest_and_rejects_substitution(monkeypatch):
    payload = b"sealed control archive" * 1024
    descriptor = _memfd(payload)
    pin = route.AgentImplementationControlPlanePin(
        schema="test", runner_path="test", runner_sha256=_digest(b"runner"),
        capsule_root="test", capsule_id="test", source_head="0" * 40,
        source_tree="1" * 40, archive_sha256=_digest(payload),
    )
    counts = _count_reads(monkeypatch)
    try:
        for _ in range(3):
            assert route.verify_agent_implementation_sealed_control_plane(pin, descriptor) == f"/proc/self/fd/{descriptor}"
        assert counts["bytes"] == len(payload)
        with pytest.raises(ValueError):
            route.verify_agent_implementation_sealed_control_plane(
                replace(pin, archive_sha256=_digest(b"forged")), descriptor
            )
    finally:
        os.close(descriptor)


def test_native_public_verifier_reuses_elf_validation_and_checks_complete_pin(
    tmp_path: Path, monkeypatch,
):
    source = _write_source(tmp_path)
    pin = _inspect(source)
    launch = _seal(source, pin)
    actual = route._agent_parse_native_dependency_elf
    calls = []

    def counted(raw):
        calls.append(len(raw))
        return actual(raw)

    monkeypatch.setattr(route, "_agent_parse_native_dependency_elf", counted)
    route._agent_clear_immutable_verifications()
    try:
        for _ in range(3):
            route.verify_agent_supervisor_native_dependency_sealed_fd(launch)
        assert calls == [source.stat().st_size]
        changed_pin = _rebound_pin(pin, elf_osabi=0 if pin.elf_osabi else 3)
        with pytest.raises(ValueError, match="does not match its pin"):
            route.verify_agent_supervisor_native_dependency_sealed_fd(
                replace(launch, pin=changed_pin)
            )
        os.fchmod(launch.descriptor.descriptor, 0o400)
        with pytest.raises(ValueError, match="identity changed"):
            route.verify_agent_supervisor_native_dependency_sealed_fd(launch)
    finally:
        os.close(launch.descriptor.descriptor)
