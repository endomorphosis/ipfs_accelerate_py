"""Bounded authored worker handoff tests; no native owner/provider workload.

Filesystem observations retain worker-reported status and never establish the
number of helpers, native source actions, or provider ingestion.
"""
from copy import deepcopy
import errno
import hashlib
import io
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_worker_receipts as api
from ipfs_accelerate_py.agent_supervisor.runtime import repository_source_owner_stdio as stdio


def authored_receipt(target, *, bridge="d" * 32):
    return dict(schema=stdio.FORWARDING_SCHEMA, status="observed", complete=True, reason=None,
        bridge_id=bridge, session_id=target["session_id"], descriptor_sha256=target["descriptor_sha256"],
        returncode=0, owner_exchange_attempts=0, owner_exchange_unconfirmed=0, observed_responses=0,
        events=[], worker_reported=True, independently_verified=False, provider_ingestion_verified=False,
        token_savings_measured=False, proof_authority=False, execution_authority=False,
        completion_authority=False, raw_content_exported=False, observer_provider_calls=0)


@pytest.fixture
def box(tmp_path, monkeypatch):
    scratch, workspace = tmp_path / "scratch", tmp_path / "workspace"
    scratch.mkdir()
    workspace.mkdir()
    monkeypatch.setattr(api.tempfile, "gettempdir", lambda: str(scratch))
    handle = api.create_outbox(workspace=workspace, session_id="a" * 32,
        descriptor_sha256="b" * 64, invocation_id="c" * 32)
    yield SimpleNamespace(handle=handle, target=handle.target, workspace=workspace)
    handle.collect_and_close()
    directory = Path(handle.target["directory"])
    if directory.exists():
        for index in range(api.MAX_SLOTS):
            name = directory / api._slot(index)
            if name.is_symlink() or name.exists():
                name.unlink()
        directory.rmdir()


def test_private_outbox_is_outside_checkout_and_target_copy_is_detached(box):
    directory = Path(box.target["directory"])
    info = directory.stat()
    assert not directory.is_relative_to(box.workspace)
    assert info.st_mode & 0o777 == 0o700
    assert info.st_uid == box.target["uid"] == os.getuid()
    assert (info.st_dev, info.st_ino) == (box.target["device"], box.target["inode"])
    target = box.handle.target
    target["directory"] = "/foreign"
    assert box.handle.target == box.target


def test_empty_scan_is_missing_evidence_not_zero_usage(box):
    capture = box.handle.collect_and_close()
    assert capture["status"] == "unavailable" and capture["reason"] == "capture_unavailable"
    assert capture["receipt_count"] is capture["receipt_bytes"] is None
    assert capture["slot_scan_complete"] and capture["cleanup_complete"]
    assert capture["receipts"] == [] and not Path(box.target["directory"]).exists()
    with pytest.raises(ValueError):
        api.captured_receipt_bytes(capture, box.target)


def test_exact_canonical_receipt_handoff_and_closed_idempotent_collection(box):
    receipt = authored_receipt(box.target)
    raw = api.canonical_json(receipt)
    assert api.publish_receipt(box.target, receipt) is True
    path = Path(box.target["directory"]) / "receipt-00.json"
    assert path.read_bytes() == raw and path.stat().st_mode & 0o777 == 0o600
    capture = box.handle.collect_and_close()
    assert capture["status"] == "captured" and capture["reason"] is None
    assert capture["receipt_count"] == 1 and capture["receipt_bytes"] == len(raw)
    assert capture["target_sha256"] == hashlib.sha256(api.canonical_json(box.target)).hexdigest()
    assert capture["receipts"] == [dict(receipt=receipt, bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest(), slot=0)]
    assert api.captured_receipt_bytes(capture, box.target) == [raw]
    assert capture["worker_reported"] is True and capture["provider_ingestion_verified"] is False
    assert capture["independently_verified"] is capture["token_savings_measured"] is False
    capture["receipts"].clear()
    assert len(box.handle.collect_and_close()["receipts"]) == 1


def test_fixed_slot_capacity_never_overwrites_existing_receipts(box):
    for index in range(api.MAX_SLOTS):
        assert api.publish_receipt(box.target, authored_receipt(box.target, bridge=format(index, "032x")))
    first = (Path(box.target["directory"]) / "receipt-00.json").read_bytes()
    assert api.publish_receipt(box.target, authored_receipt(box.target)) is False
    assert (Path(box.target["directory"]) / "receipt-00.json").read_bytes() == first
    capture = box.handle.collect_and_close()
    assert capture["receipt_count"] == 16
    assert len(api.captured_receipt_bytes(capture, box.target)) == 16


@pytest.mark.parametrize("content", [b"{", b"{}", b'{"schema":"PRIVATE_SOURCE_CONTENT"}',
    b'{"schema":1,"schema":2}', b' {"schema":"unknown"} '])
def test_stable_invalid_content_is_not_exported_but_safe_slot_is_cleaned(box, content):
    path = Path(box.target["directory"]) / "receipt-00.json"
    path.write_bytes(content)
    path.chmod(0o600)
    capture = box.handle.collect_and_close()
    assert capture["status"] == "unavailable" and capture["receipt_count"] is None
    assert capture["receipts"] == [] and capture["slot_scan_complete"] is False
    assert capture["cleanup_complete"] is True and not path.parent.exists()
    assert "PRIVATE_SOURCE_CONTENT" not in api.canonical_json(capture).decode()


def test_noncanonical_valid_json_is_refused_and_cleaned(box):
    path = Path(box.target["directory"]) / "receipt-00.json"
    path.write_bytes(api.canonical_json(authored_receipt(box.target)) + b"\n")
    path.chmod(0o600)
    capture = box.handle.collect_and_close()
    assert capture["status"] == "unavailable" and capture["receipts"] == []
    assert capture["cleanup_complete"] is True


@pytest.mark.parametrize("kind", ["symlink", "hardlink", "fifo", "mode", "oversize", "empty"])
def test_unsafe_slots_are_never_read_as_evidence_or_unlinked(box, tmp_path, kind):
    path = Path(box.target["directory"]) / "receipt-00.json"
    foreign = tmp_path / "foreign"
    raw = api.canonical_json(authored_receipt(box.target))
    foreign.write_bytes(raw)
    foreign.chmod(0o600)
    if kind == "symlink":
        path.symlink_to(foreign)
    elif kind == "hardlink":
        os.link(foreign, path)
    elif kind == "fifo":
        os.mkfifo(path, 0o600)
    else:
        path.write_bytes(raw if kind == "mode" else b"x" * (api.MAX_RECEIPT_BYTES + 1) if kind == "oversize" else b"")
        path.chmod(0o640 if kind == "mode" else 0o600)
    capture = box.handle.collect_and_close()
    assert capture["status"] == "unavailable" and capture["cleanup_complete"] is False
    assert capture["receipts"] == [] and (path.exists() or path.is_symlink())
    assert foreign.read_bytes() == raw


@pytest.mark.parametrize("key,value", [("extra", "PRIVATE"), ("proof_authority", True),
    ("owner_exchange_attempts", True), ("bridge_id", "PRIVATE"), ("reason", "PRIVATE"),
    ("session_id", "e" * 32), ("descriptor_sha256", "e" * 64), ("events", ["PRIVATE"])])
def test_closed_receipt_rejects_content_and_identity_before_publication(box, key, value):
    receipt = authored_receipt(box.target)
    receipt[key] = value
    with pytest.raises((ValueError, TypeError)):
        api.publish_receipt(box.target, receipt)
    assert not (Path(box.target["directory"]) / "receipt-00.json").exists()


@pytest.mark.parametrize("key,value", [("inode", 1), ("uid", os.getuid() + 1), ("device", 0)])
def test_target_identity_changes_refuse_io(box, key, value):
    target = dict(box.target, **{key: value})
    with pytest.raises(ValueError):
        api.publish_receipt(target, authored_receipt(target))


def test_target_validation_is_historical_and_does_not_open_directory(box, monkeypatch):
    target = dict(box.target, directory="/authored/nonexistent/outbox")
    monkeypatch.setattr(api, "_open_directory", lambda _: (_ for _ in ()).throw(AssertionError("filesystem")))
    assert api.validate_target(target) == target
    monkeypatch.undo()


def test_publish_failure_removes_only_own_partial_creation(box, monkeypatch):
    real_write = api.os.write
    calls = []
    def partial(fd, raw):
        calls.append(fd)
        if len(calls) == 1:
            return real_write(fd, raw[:10])
        raise OSError("PRIVATE_EXCEPTION")
    with monkeypatch.context() as patch:
        patch.setattr(api.os, "write", partial)
        with pytest.raises(OSError):
            api.publish_receipt(box.target, authored_receipt(box.target))
    assert not (Path(box.target["directory"]) / "receipt-00.json").exists()
    assert box.handle.collect_and_close()["receipt_count"] is None


@pytest.mark.parametrize("interrupted", [False, True])
def test_first_publication_fstat_failure_closes_fd_and_preserves_unknown_slot(box, monkeypatch, interrupted):
    error = KeyboardInterrupt("PRIVATE_FIRST_FSTAT") if interrupted else OSError("PRIVATE_FIRST_FSTAT")
    real_open, real_fstat, real_close = api.os.open, api.os.fstat, api.os.close
    opened, closed = [], []

    def open_slot(path, flags, *args, **kwargs):
        fd = real_open(path, flags, *args, **kwargs)
        if path == "receipt-00.json" and flags & os.O_CREAT:
            opened.append(fd)
        return fd

    def failed_first_fstat(fd):
        if opened and fd == opened[0]:
            raise error
        return real_fstat(fd)

    def close(fd):
        if opened and fd == opened[0]:
            closed.append(fd)
        return real_close(fd)

    with monkeypatch.context() as patch:
        patch.setattr(api.os, "open", open_slot)
        patch.setattr(api.os, "fstat", failed_first_fstat)
        patch.setattr(api.os, "close", close)
        with pytest.raises(type(error)) as caught:
            api.publish_receipt(box.target, authored_receipt(box.target))
    assert caught.value is error
    assert len(opened) == 1 and closed == opened
    with pytest.raises(OSError) as retired:
        real_fstat(opened[0])
    assert retired.value.errno == errno.EBADF
    path = Path(box.target["directory"]) / "receipt-00.json"
    unknown = path.lstat()
    assert path.read_bytes() == b"" and unknown.st_mode & 0o777 == 0o600

    # A retry chooses the next exclusive slot, without claiming the unknown
    # creation or overwriting it. One valid sibling cannot establish completeness.
    assert api.publish_receipt(box.target, authored_receipt(box.target)) is True
    assert api._snapshot(path.lstat()) == api._snapshot(unknown)
    assert path.read_bytes() == b""
    capture = box.handle.collect_and_close()
    assert capture["status"] == "unavailable" and capture["reason"] == "capture_failed"
    assert capture["receipt_count"] is capture["receipt_bytes"] is None
    assert capture["slot_scan_complete"] is capture["cleanup_complete"] is False
    assert [entry["slot"] for entry in capture["receipts"]] == [1]
    assert box.handle._fd is None and path.exists()
    assert not (path.parent / "receipt-01.json").exists()
    assert "PRIVATE_" not in api.canonical_json(capture).decode()
    with pytest.raises(ValueError):
        api.captured_receipt_bytes(capture, box.target)


def test_zero_write_and_keyboard_interrupt_are_not_success(box, monkeypatch):
    for fault in (0, KeyboardInterrupt()):
        def interrupted(fd, raw):
            if isinstance(fault, BaseException):
                raise fault
            return fault
        with monkeypatch.context() as patch:
            patch.setattr(api.os, "write", interrupted)
            with pytest.raises(KeyboardInterrupt if isinstance(fault, BaseException) else ValueError):
                api.publish_receipt(box.target, authored_receipt(box.target))
        assert not (Path(box.target["directory"]) / "receipt-00.json").exists()


def test_failure_never_unlinks_a_replacement_slot(box, monkeypatch):
    path = Path(box.target["directory"]) / "receipt-00.json"
    def replace_then_fail(fd, raw):
        path.unlink()
        path.write_bytes(b"replacement")
        path.chmod(0o600)
        raise OSError("fail")
    with monkeypatch.context() as patch:
        patch.setattr(api.os, "write", replace_then_fail)
        with pytest.raises(OSError):
            api.publish_receipt(box.target, authored_receipt(box.target))
    assert path.read_bytes() == b"replacement"


def test_scan_interruption_is_sticky_unknown_and_fd_is_closed(box, monkeypatch):
    api.publish_receipt(box.target, authored_receipt(box.target))
    with monkeypatch.context() as patch:
        patch.setattr(api.os, "read", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            box.handle.collect_and_close()
    capture = box.handle.collect_and_close()
    assert capture["status"] == "unavailable" and capture["slot_scan_complete"] is False
    assert capture["receipt_count"] is None and box.handle._fd is None


def test_cleanup_failure_preserves_unknown_and_closed_metadata(box, monkeypatch):
    api.publish_receipt(box.target, authored_receipt(box.target))
    with monkeypatch.context() as patch:
        patch.setattr(api.os, "rmdir", lambda *args, **kwargs: (_ for _ in ()).throw(OSError("PRIVATE")))
        capture = box.handle.collect_and_close()
    assert capture["status"] == "unavailable" and capture["reason"] == "cleanup_failed"
    assert capture["receipt_count"] is capture["receipt_bytes"] is None
    assert capture["cleanup_complete"] is False and "PRIVATE" not in api.canonical_json(capture).decode()


def test_scan_uses_no_directory_listing_or_recursive_cleanup(box, monkeypatch):
    api.publish_receipt(box.target, authored_receipt(box.target))
    with monkeypatch.context() as patch:
        def forbidden(*args, **kwargs):
            raise AssertionError("directory enumeration is forbidden")
        patch.setattr(api.os, "listdir", forbidden)
        patch.setattr(api.os, "scandir", forbidden)
        assert box.handle.collect_and_close()["status"] == "captured"


@pytest.mark.parametrize("mutation", ["digest", "length", "content", "target", "slot", "flags", "duplicate"])
def test_pure_capture_validation_rejects_tampering(box, mutation):
    api.publish_receipt(box.target, authored_receipt(box.target))
    capture = box.handle.collect_and_close()
    entry = capture["receipts"][0]
    if mutation == "digest":
        entry["sha256"] = "0" * 64
    elif mutation == "length":
        entry["bytes"] += 1
    elif mutation == "content":
        entry["receipt"]["PRIVATE"] = "content"
    elif mutation == "target":
        capture["target_sha256"] = "0" * 64
    elif mutation == "slot":
        entry["slot"] = 16
    elif mutation == "flags":
        capture["provider_ingestion_verified"] = True
    else:
        capture["receipts"].append(deepcopy(entry))
    with pytest.raises(ValueError):
        api.captured_receipt_bytes(capture, box.target)


def _cli_args(box):
    return [stdio.protocol.DESCRIPTOR_FLAG, str(box.workspace.parent / "public-descriptor.json"),
        stdio.protocol.DIGEST_FLAG, box.target["descriptor_sha256"], "--workspace", str(box.workspace),
        "--worker-forwarding-target", api.canonical_json(box.target).decode()]


def _authored_serve(box, monkeypatch, *, result=0):
    calls = []
    def serve(*args, **kwargs):
        calls.append(kwargs)
        observation = kwargs.get("observation")
        if observation is not None:
            observation._used = True
            observation._start()
            observation._bind({"session_id": box.target["session_id"]}, box.target["descriptor_sha256"])
            observation._finish(result)
        return result
    monkeypatch.setattr(stdio, "serve", serve)
    monkeypatch.setattr(stdio, "sys", SimpleNamespace(stdin=SimpleNamespace(buffer=io.BytesIO()),
        stdout=SimpleNamespace(buffer=io.BytesIO()), stderr=io.StringIO()))
    return calls


def test_optional_cli_publishes_once_after_serve_and_default_is_unchanged(box, monkeypatch):
    calls = _authored_serve(box, monkeypatch)
    argv = _cli_args(box)
    assert stdio.main(argv[:-2]) == 0 and "observation" not in calls[0]
    assert not (Path(box.target["directory"]) / "receipt-00.json").exists()
    assert stdio.main(argv) == 0 and type(calls[1]["observation"]) is stdio.ForwardingObservation
    capture = box.handle.collect_and_close()
    assert capture["status"] == "captured" and capture["receipt_count"] == 1
    assert stdio.sys.stdout.buffer.getvalue() == b"" and stdio.sys.stderr.getvalue() == ""


@pytest.mark.parametrize("failure", ["target", "publish", "receipt"])
def test_advisory_cli_failures_preserve_original_exit_and_stdout(box, monkeypatch, failure):
    calls = _authored_serve(box, monkeypatch, result=2)
    argv = _cli_args(box)
    if failure == "target":
        argv[-1] = '{"PRIVATE":"bad"}'
    elif failure == "publish":
        monkeypatch.setattr(api, "publish_receipt", lambda *args: (_ for _ in ()).throw(OSError("PRIVATE")))
    else:
        monkeypatch.setattr(stdio.ForwardingObservation, "receipt", lambda _: (_ for _ in ()).throw(ValueError("PRIVATE")))
    assert stdio.main(argv) == 2 and len(calls) == 1
    assert stdio.sys.stdout.buffer.getvalue() == b""
    assert stdio.sys.stderr.getvalue() == stdio.UNAVAILABLE + "\n"


def test_creation_open_failure_cleans_exact_fresh_directory(tmp_path, monkeypatch):
    workspace, scratch = tmp_path / "workspace", tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(api.tempfile, "gettempdir", lambda: str(scratch))
    monkeypatch.setattr(api, "_open_directory", lambda _: (_ for _ in ()).throw(OSError("fail")))
    with pytest.raises(OSError):
        api.create_outbox(workspace=workspace, session_id="a" * 32, descriptor_sha256="b" * 64,
                          invocation_id="c" * 32)
    assert list(scratch.iterdir()) == []


@pytest.mark.parametrize("directory", ["//tmp/workspace/outbox", "/", "/tmp/back\\slash", "/tmp/control\x7f"])
def test_target_path_aliases_and_controls_are_rejected(box, directory):
    with pytest.raises(ValueError):
        api.validate_target(dict(box.target, directory=directory), workspace="/tmp/workspace")


@pytest.mark.parametrize("pending", [False, True])
def test_cli_publication_interruption_preserves_original_control_flow(box, monkeypatch, pending):
    _authored_serve(box, monkeypatch)
    original = KeyboardInterrupt("original")
    if pending:
        monkeypatch.setattr(stdio, "serve", lambda *args, **kwargs: (_ for _ in ()).throw(original))
    publication = SystemExit(7)
    monkeypatch.setattr(api, "publish_receipt", lambda *args: (_ for _ in ()).throw(publication))
    with pytest.raises(KeyboardInterrupt if pending else SystemExit) as found:
        stdio.main(_cli_args(box))
    assert found.value is (original if pending else publication)


def test_final_validation_interruption_cannot_leave_completed_snapshot(box, monkeypatch):
    api.publish_receipt(box.target, authored_receipt(box.target))
    with monkeypatch.context() as patch:
        patch.setattr(api, "captured_receipt_bytes", lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
        with pytest.raises(KeyboardInterrupt):
            box.handle.collect_and_close()
    capture = box.handle.collect_and_close()
    assert capture["status"] == "unavailable" and capture["receipt_count"] is None
    assert capture["cleanup_complete"] is True and box.handle._fd is None


def test_isolated_direct_cli_optional_helper_needs_no_package_or_owner(box):
    # Missing authored descriptor stops before any owner contact/native work.
    # -I forces the real fixed-adjacent standard-library-only import route.
    result = subprocess.run([sys.executable, "-I", "-B", str(Path(stdio.__file__).resolve()), *_cli_args(box)],
        cwd=box.workspace, capture_output=True, check=False, timeout=5)
    assert result.returncode == 2 and result.stdout == b""
    assert result.stderr == (stdio.UNAVAILABLE + "\n").encode()
    capture = box.handle.collect_and_close()
    assert capture["status"] == "captured" and capture["receipt_count"] == 1
    receipt = capture["receipts"][0]["receipt"]
    assert receipt["complete"] is False and receipt["reason"] == "descriptor_unavailable"
    assert receipt["session_id"] is None and receipt["provider_ingestion_verified"] is False


def test_creation_replacement_is_not_adopted_chmodded_or_removed(tmp_path, monkeypatch):
    workspace, scratch = tmp_path / "workspace", tmp_path / "scratch"
    workspace.mkdir()
    scratch.mkdir()
    monkeypatch.setattr(api.tempfile, "gettempdir", lambda: str(scratch))
    original_open = api._open_directory
    changed = []
    def replace_before_open(path):
        path = Path(path)
        moved = path.with_name(path.name + "-moved")
        path.rename(moved)
        path.mkdir(mode=0o755)
        path.chmod(0o755)
        changed.extend((path, moved))
        return original_open(path)
    monkeypatch.setattr(api, "_open_directory", replace_before_open)
    with pytest.raises(ValueError):
        api.create_outbox(workspace=workspace, session_id="a" * 32, descriptor_sha256="b" * 64,
                          invocation_id="c" * 32)
    assert len(changed) == 2 and changed[0].stat().st_mode & 0o777 == 0o755
    for path in changed:
        path.rmdir()
