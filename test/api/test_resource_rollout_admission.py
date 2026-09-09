"""Resource-only legacy rollout checks; no owner, schema, or live state writes."""

from __future__ import annotations

import os
import select
import subprocess
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import pytest

from ipfs_accelerate_py import _hash_resources as resources
from ipfs_accelerate_py.agent_supervisor.runtime import hash_pressure as facade


@pytest.fixture(autouse=True)
def isolated_admission(tmp_path, monkeypatch):
    path = tmp_path / "test-hashing.lock"
    monkeypatch.setattr(resources, "hash_lock_path", lambda kind="file-hash": path)
    monkeypatch.setattr(resources, "host_hash_pressure", lambda: (4, "test"))
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "2")
    yield path
    assert not resources._open_fds


def test_checkout_uses_its_own_resource_module_and_compatible_facade():
    assert Path(resources.__file__).resolve() == (
        Path(__file__).resolve().parents[2] / "ipfs_accelerate_py" / "_hash_resources.py"
    )
    for name in ("hashing_lock", "hashing_worker_slot", "hash_worker_limit",
                 "HashingResourceTimeout"):
        assert getattr(facade, name) is getattr(resources, name)
    assert facade.host_hash_pressure.__module__ == resources.__name__
    assert facade.MEMORY_PERCENT_LIMIT == 80
    assert facade.SWAP_FLOOR_BYTES == 256 * 1024**2


def test_legacy_kinds_and_exclusive_false_still_participate_in_one_budget():
    # The legacy signature remains accepted, but no call skips admission.
    with facade.hashing_lock(kind="workspace-fingerprint", exclusive=False) as workers:
        assert workers == 2
        with facade.hashing_lock(kind="different-kind", exclusive=True) as nested:
            assert nested == workers
        with facade.hashing_worker_slot() as count:
            assert count == 1


def test_default_two_workers_hard_cap_four_and_pressure_reduction(monkeypatch):
    assert facade.hash_worker_limit() == 2
    monkeypatch.setenv("IPFS_HASH_MAX_WORKERS", "999")
    assert facade.hash_worker_limit(999) == 4
    monkeypatch.setattr(resources, "host_hash_pressure", lambda: (1, "test-pressure"))
    assert facade.hash_worker_limit(999) == 1


def test_lock_namespace_is_shared_across_kinds_and_runtime_directories(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    path = facade.hash_lock_path("workspace-fingerprint")
    assert path == facade.hash_lock_path("parallel-proof-verification")
    assert path == Path("/tmp") / f"ipfs-accelerate-heavy-hash-{os.geteuid()}.lock"


def test_two_threads_overlap_and_third_times_out_without_oversubscription():
    entered = threading.Event()
    release = threading.Event()

    def second():
        with facade.hashing_worker_slot(timeout=1):
            entered.set()
            assert release.wait(2)

    def third():
        with pytest.raises(resources.HashingResourceTimeout):
            with facade.hashing_worker_slot(timeout=0.03):
                pytest.fail("third worker exceeded the shared budget")

    with ThreadPoolExecutor(max_workers=2) as executor:
        with facade.hashing_worker_slot():
            task = executor.submit(second)
            try:
                assert entered.wait(1), "second worker did not overlap first"
                executor.submit(third).result(timeout=1)
            finally:
                release.set()
            task.result(timeout=1)


def test_worker_exception_releases_and_upgrade_fails_explicitly():
    with pytest.raises(RuntimeError, match="test failure"):
        with facade.hashing_worker_slot():
            with pytest.raises(resources.HashingResourceUpgradeError):
                with facade.hashing_lock():
                    pass
            raise RuntimeError("test failure")
    with facade.hashing_lock(timeout=0.1):
        pass


_CHILD = """
import importlib.util, pathlib, sys
spec = importlib.util.spec_from_file_location('isolated_resources', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
module.hash_lock_path = lambda kind='file-hash': pathlib.Path(sys.argv[2])
module.host_hash_pressure = lambda: (4, 'test')
with module.hashing_lock(kind='child-heavy-batch', timeout=2):
    print('entered', flush=True)
    sys.stdin.readline()
"""


@contextmanager
def other_process_heavy_batch(path):
    process = subprocess.Popen(
        [sys.executable, "-B", "-c", _CHILD, resources.__file__, str(path)],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
    )
    try:
        assert select.select([process.stdout], [], [], 3)[0], "child did not acquire admission"
        assert process.stdout.readline().strip() == "entered"
        yield process
    finally:
        if process.poll() is None:
            try:
                process.stdin.write("\n")
                process.stdin.flush()
                process.wait(timeout=3)
            except (BrokenPipeError, subprocess.TimeoutExpired):
                process.kill()
                process.wait(timeout=2)
        for stream in (process.stdin, process.stdout, process.stderr):
            stream.close()


@pytest.mark.skipif(resources.fcntl is None, reason="requires flock")
def test_other_process_heavy_batch_excludes_streaming_workers(isolated_admission):
    with other_process_heavy_batch(isolated_admission):
        with pytest.raises(resources.HashingResourceTimeout):
            with facade.hashing_worker_slot(timeout=0.03):
                pytest.fail("worker entered while another process owns the batch")
    with facade.hashing_worker_slot(timeout=0.1):
        pass


# PCTDD-only integration: hooks and aggregation never hold a streaming slot.
from ipfs_accelerate_py.agent_supervisor.proof.incremental_sealing import (
    parallel_verification as verification,
)


def _unit(name="unit:test", **overrides):
    data = b"bounded hash payload"
    fields = dict(
        unit_id=name, kind=verification.CheckKind.INTEGRITY,
        payload=data, expected_digest=verification.digest_bytes(data),
    )
    fields.update(overrides)
    return verification.VerificationUnit(**fields)


def test_verification_hooks_can_take_exclusive_admission_without_upgrade_deadlock():
    calls = []

    def before(unit):
        with facade.hashing_lock(timeout=0.2):
            calls.append((unit.unit_id, "before"))

    def after(unit, record):
        with facade.hashing_lock(timeout=0.2):
            assert record.accepted
            calls.append((unit.unit_id, "after"))

    result = verification.verify_units_in_parallel(
        [_unit()], hooks=verification.ParallelVerificationHooks(before_unit=before, after_unit=after),
    )
    assert result.accepted and calls == [("unit:test", "before"), ("unit:test", "after")]


def test_only_pure_verification_holds_slot_and_aggregation_is_outside(monkeypatch):
    original_payload = verification._verify_payload
    original_aggregate = verification._aggregate
    observations = []

    def payload(*args, **kwargs):
        assert resources._local.pid == os.getpid() and resources._local.mode == "worker"
        observations.append("payload")
        return original_payload(*args, **kwargs)

    def aggregate(*args, **kwargs):
        assert getattr(resources._local, "pid", None) != os.getpid()
        observations.append("aggregate")
        return original_aggregate(*args, **kwargs)

    monkeypatch.setattr(verification, "_verify_payload", payload)
    monkeypatch.setattr(verification, "_aggregate", aggregate)
    assert verification.verify_units_in_parallel([_unit()]).accepted
    assert observations == ["payload", "aggregate"]


def test_two_pure_verifications_overlap_and_exclude_heavy_batches(monkeypatch):
    original = verification._verify_payload
    release = threading.Event()
    two_entered = threading.Event()
    state_lock = threading.Lock()
    state = {"active": 0, "maximum": 0, "calls": 0}

    def payload(*args, **kwargs):
        with state_lock:
            state["active"] += 1
            state["calls"] += 1
            state["maximum"] = max(state["maximum"], state["active"])
            if state["active"] == 2:
                two_entered.set()
        try:
            assert release.wait(3), "test did not release verification workers"
            return original(*args, **kwargs)
        finally:
            with state_lock:
                state["active"] -= 1

    monkeypatch.setattr(verification, "_verify_payload", payload)
    with ThreadPoolExecutor(max_workers=1) as executor:
        result = executor.submit(
            verification.verify_units_in_parallel, [_unit(f"unit:{index}") for index in range(3)],
        )
        try:
            assert two_entered.wait(2), "verification did not use two admitted workers"
            with pytest.raises(resources.HashingResourceTimeout):
                with facade.hashing_lock(timeout=0.03):
                    pytest.fail("exclusive batch entered during pure verification")
            assert state["calls"] == 2
        finally:
            release.set()
        assert result.result(timeout=3).accepted
    assert state["maximum"] == 2 and state["calls"] == 3 and state["active"] == 0


@pytest.mark.skipif(resources.fcntl is None, reason="requires flock")
def test_busy_global_budget_returns_typed_timeout_without_hashing(isolated_admission, monkeypatch):
    units = [_unit(f"unit:{index}") for index in range(8)]

    def forbidden(*args, **kwargs):
        pytest.fail("verification read payload while another process owns the hash budget")

    monkeypatch.setattr(verification, "_verify_payload", forbidden)
    with other_process_heavy_batch(isolated_admission):
        result = verification.verify_units_in_parallel(
            units, bounds=verification.VerificationBounds(timeout_seconds=0.04),
        )
    assert not result.accepted
    assert all(record.reason is verification.UnitVerificationReason.TIMEOUT for record in result.units)


def test_rejected_payload_keeps_before_hook_but_does_not_run_after_hook():
    calls = []
    result = verification.verify_units_in_parallel(
        [_unit(expected_digest="sha256:" + "0" * 64)],
        hooks=verification.ParallelVerificationHooks(
            before_unit=lambda unit: calls.append("before"),
            after_unit=lambda unit, record: calls.append("after"),
        ),
    )
    assert not result.accepted and calls == ["before"]
    assert result.units[0].reason is verification.UnitVerificationReason.DIGEST_MISMATCH


def test_unavailable_and_oversized_units_never_request_hash_admission(monkeypatch):
    def forbidden(**kwargs):
        pytest.fail("invalid or unavailable unit requested a hashing slot")

    monkeypatch.setattr(verification, "hashing_worker_slot", forbidden)
    unavailable = verification.verify_units_in_parallel([_unit(backend_id="provekit")])
    assert unavailable.units[0].reason is verification.UnitVerificationReason.UNAVAILABLE
    oversized = verification.verify_units_in_parallel(
        [_unit()], bounds=verification.VerificationBounds(max_unit_bytes=1),
    )
    assert oversized.units[0].reason is verification.UnitVerificationReason.BOUND_EXCEEDED


def test_expired_batch_deadline_prevents_queued_hashes(monkeypatch):
    import time

    original = verification._verify_payload
    calls = []

    def slow_payload(*args, **kwargs):
        calls.append(1)
        time.sleep(0.04)
        return original(*args, **kwargs)

    monkeypatch.setattr(verification, "_verify_payload", slow_payload)
    result = verification.verify_units_in_parallel(
        [_unit(f"unit:{index}") for index in range(8)],
        bounds=verification.VerificationBounds(timeout_seconds=0.02),
    )
    assert not result.accepted and 1 <= len(calls) <= 2
    assert all(record.reason is verification.UnitVerificationReason.TIMEOUT for record in result.units)

def test_capsule_required_files_include_root_resource_module(tmp_path):
    from ipfs_accelerate_py import agent_implementation_route as route

    relative = "ipfs_accelerate_py/_hash_resources.py"
    assert route._AGENT_CONTROL_PLANE_RELATIVE_FILES.count(relative) == 1
    (tmp_path / "ipfs_accelerate_py" / "agent_supervisor").mkdir(parents=True)
    selected = route._agent_control_plane_source_files(tmp_path, verify_loaded_origins=False)
    assert tmp_path / relative in selected


def test_capsule_resource_module_origin_must_match_selected_root(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from ipfs_accelerate_py import agent_implementation_route as route

    capsule = tmp_path / "capsule"
    (capsule / "ipfs_accelerate_py" / "agent_supervisor").mkdir(parents=True)
    included = capsule / "ipfs_accelerate_py" / "_hash_resources.py"
    included.write_text("# admitted resource helper\n", encoding="utf-8")
    candidate = tmp_path / "candidate_hash_resources.py"
    candidate.write_text("# untrusted candidate helper\n", encoding="utf-8")
    observed = SimpleNamespace(__file__=str(included))
    monkeypatch.setattr(route, "sys", SimpleNamespace(modules={
        "ipfs_accelerate_py._hash_resources": observed,
    }))
    assert included in route._agent_control_plane_source_files(capsule)
    observed.__file__ = str(candidate)
    with pytest.raises(ValueError, match="crossed capsule roots"):
        route._agent_control_plane_source_files(capsule)
    observed.__file__ = str(tmp_path / "missing.py")
    with pytest.raises(ValueError, match="origin is unavailable"):
        route._agent_control_plane_source_files(capsule)


def test_capsule_copy_imports_resource_facade_without_checkout_fallback(tmp_path):
    import json

    # Exercise the new transitive dependency using an isolated import root,
    # without constructing a full capsule or hashing the repository.
    package = tmp_path / "capsule" / "ipfs_accelerate_py"
    runtime = package / "agent_supervisor" / "runtime"
    runtime.mkdir(parents=True)
    for directory in (package, package / "agent_supervisor", runtime):
        (directory / "__init__.py").write_text("", encoding="utf-8")
    (package / "_hash_resources.py").write_bytes(Path(resources.__file__).read_bytes())
    (runtime / "hash_pressure.py").write_bytes(Path(facade.__file__).read_bytes())
    code = """
import json, pathlib, sys
sys.path.insert(0, sys.argv[1])
from ipfs_accelerate_py import _hash_resources
from ipfs_accelerate_py.agent_supervisor.runtime import hash_pressure
assert hash_pressure.hashing_worker_slot is _hash_resources.hashing_worker_slot
print(json.dumps([str(pathlib.Path(_hash_resources.__file__).resolve()),
                  str(pathlib.Path(hash_pressure.__file__).resolve())]))
"""
    result = subprocess.run(
        [sys.executable, "-I", "-B", "-c", code, str(package.parent)],
        capture_output=True, text=True, timeout=5, check=True,
    )
    assert json.loads(result.stdout) == [
        str(package / "_hash_resources.py"), str(runtime / "hash_pressure.py"),
    ]
