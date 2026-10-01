"""Real kernel seals remain mandatory when Python omits Linux API wrappers."""
import fcntl
import os
import sys
from types import SimpleNamespace

import pytest

from ipfs_accelerate_py.agent_supervisor.validation import validation_runtime as runtime


pytestmark = pytest.mark.skipif(not sys.platform.startswith('linux'), reason='Linux memfd ABI')


def test_missing_python_memfd_wrapper_uses_real_kernel_seals(monkeypatch):
    monkeypatch.delattr(os, 'memfd_create', raising=False)
    # Deliberately expose only the real syscall wrapper, as this CPython build
    # does; numeric Linux ABI constants do not constitute proof of sealing.
    limited = SimpleNamespace(fcntl=fcntl.fcntl)
    payload = b'#!/bin/sh\nprintf sealed-kernel-check\\n\n'
    descriptor, path = runtime._sealed_executable_memfd(
        name='portable-validation-test', payload=payload, creation_flags=3,
        required_seals=15, fcntl_module=limited)
    try:
        assert fcntl.fcntl(descriptor, 1034) & 15 == 15
        assert os.pread(descriptor, len(payload), 0) == payload
        with pytest.raises(OSError):
            os.write(descriptor, b'changed')
        with pytest.raises(OSError):
            os.ftruncate(descriptor, 0)
        assert path.endswith('/' + str(descriptor))
    finally:
        os.close(descriptor)


def test_missing_macro_fallback_still_rejects_kernel_without_seals(monkeypatch):
    monkeypatch.delattr(os, 'memfd_create', raising=False)
    descriptors = []

    def refusing(fd, command, *args):
        descriptors.append(fd)
        if command == 1034:
            return 0
        return fcntl.fcntl(fd, command, *args)

    with pytest.raises(runtime.ValidationRuntimeError, match='all required seals'):
        runtime._sealed_executable_memfd(name='refused-seals', payload=b'unchanged',
            creation_flags=3, required_seals=15, fcntl_module=SimpleNamespace(fcntl=refusing))
    assert descriptors
    with pytest.raises(OSError):
        os.fstat(descriptors[-1])


def test_validation_memfd_fallback_is_linux_only(monkeypatch):
    monkeypatch.setattr(sys, 'platform', 'darwin')
    with pytest.raises(runtime.ValidationRuntimeError, match='requires Linux'):
        runtime._validation_memfd_create('no-portable-file-fallback', 3)
