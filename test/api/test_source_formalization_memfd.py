"""Real Linux sealed stdin when portable Python omits its wrappers/constants."""
import ctypes
import errno
import fcntl
import os
import subprocess
import sys

import pytest

from ipfs_accelerate_py.agent_supervisor.runtime import source_formalization_context as guard
from ipfs_accelerate_py.agent_supervisor.validation import validation_runtime as validation
from test.api.test_source_formalization_context import authored, sealed  # noqa: F401


@pytest.fixture(autouse=True)
def portable_bindings(monkeypatch):
    monkeypatch.delattr(os, "memfd_create", raising=False)
    for name in ("MFD_CLOEXEC", "MFD_ALLOW_SEALING"):
        monkeypatch.delattr(os, name, raising=False)
    for name in ("F_ADD_SEALS", "F_GET_SEALS", "F_SEAL_WRITE", "F_SEAL_GROW", "F_SEAL_SHRINK", "F_SEAL_SEAL"):
        monkeypatch.delattr(fcntl, name, raising=False)


def capture(raw=b"ordinary captured prompt; preserve \xce\xbb\n"):
    input_fd, writer = os.pipe()
    try:
        os.write(writer, raw)
        os.close(writer)
        return guard.prepare_container_source_formalization_dispatch(argv=["--provider", "grok_cli"],
            boundary=None, workspace=None, input_fd=input_fd)
    finally:
        os.close(input_fd)


def assert_sealed(fd, raw):
    assert fcntl.fcntl(fd, 1034) & 15 == 15
    assert fcntl.fcntl(fd, fcntl.F_GETFD) & fcntl.FD_CLOEXEC
    assert os.readlink(f"/proc/self/fd/{fd}").startswith("/memfd:supervisor-owner-prompt")
    assert os.fstat(fd).st_nlink == 0
    assert os.pread(fd, len(raw) + 1, 0) == raw
    for mutate in (lambda: os.pwrite(fd, b"!", 0),
                   lambda: os.ftruncate(fd, len(raw) + 1),
                   lambda: os.ftruncate(fd, max(0, len(raw) - 1))):
        with pytest.raises(OSError) as refused:
            mutate()
        assert refused.value.errno == errno.EPERM


def test_missing_python_bindings_still_dispatch_exact_sealed_stdin():
    raw = b"ordinary captured prompt; preserve \xce\xbb\n"
    prepared = capture(raw)
    try:
        assert prepared["argv"] == ["--provider", "grok_cli"]
        assert_sealed(prepared["stdin_fd"], raw)
        result = subprocess.run([sys.executable, "-I", "-c",
            "import os,sys;sys.stdout.buffer.write(os.read(0,2097153))"],
            stdin=prepared["stdin_fd"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        assert result.stdout == raw
    finally:
        os.close(prepared["stdin_fd"])


def test_missing_bindings_preserve_full_source_advice_capture_and_worker_gate(sealed, monkeypatch):
    case, prepared, request = sealed
    assert len(case["calls"]) == 1
    assert_sealed(prepared["stdin_fd"], request["prompt"].encode())
    def refuse_owner_replay(**kwargs):
        pytest.fail("worker read private owner state")
    monkeypatch.setattr(guard, "load_task_context_selection", refuse_owner_replay)
    observed = guard.validate_source_formalization_context(**request)
    assert observed["owner_validation_inherited"] is True
    assert observed["canonical_source_current"] is observed["workspace_source_current"] is True
    assert observed["proof_authority"] is False


@pytest.mark.parametrize("failure", ["stream_open", "write", "add_seals", "missing_seals", "wrong_bytes"])
def test_capture_failure_closes_its_anonymous_descriptor(monkeypatch, failure):
    created = []
    native = validation._validation_memfd_create
    def create(name, flags):
        fd = native(name, flags)
        if name == "supervisor-owner-prompt":
            created.append(fd)
        return fd
    monkeypatch.setattr(validation, "_validation_memfd_create", create)
    original_fcntl = fcntl.fcntl
    def checked_fcntl(fd, operation, *args):
        if fd in created and operation == 1033 and failure == "add_seals":
            raise OSError(errno.EPERM, "injected seal denial")
        if fd in created and operation == 1034 and failure == "missing_seals":
            return 0
        return original_fcntl(fd, operation, *args)
    monkeypatch.setattr(fcntl, "fcntl", checked_fcntl)
    if failure == "stream_open":
        def fdopen(fd, *args, **kwargs):
            raise OSError(errno.EIO, "injected stream construction failure")
        monkeypatch.setattr(os, "fdopen", fdopen)
    if failure == "write":
        class FailingStream:
            def __enter__(self):
                return self
            def __exit__(self, *args):
                return False
            def write(self, raw):
                raise OSError(errno.EIO, "injected write failure")
        monkeypatch.setattr(os, "fdopen", lambda *args, **kwargs: FailingStream())
    if failure == "wrong_bytes":
        monkeypatch.setattr(os, "pread", lambda *args: b"wrong")
    with pytest.raises((ValueError, OSError)):
        capture()
    assert len(created) == 1
    with pytest.raises(OSError) as closed:
        os.fstat(created[0])
    assert closed.value.errno == errno.EBADF


def test_no_libc_memfd_api_has_no_regular_file_fallback(monkeypatch):
    monkeypatch.setattr(ctypes, "CDLL", lambda *args, **kwargs: object())
    with pytest.raises(validation.ValidationRuntimeError, match="memfd_create is unavailable"):
        capture()
