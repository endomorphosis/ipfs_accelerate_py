"""Cold-import and exact-object contracts for the accelerator package root."""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest


ACCELERATE_ROOT = Path(__file__).resolve().parents[2]


def _run(script: str, *, skip_core: bool = False) -> None:
    environment = dict(os.environ)
    environment["IPFS_ACCEL_SKIP_CORE"] = "1" if skip_core else "0"
    environment["IPFS_ACCEL_IMPORT_EAGER"] = "0"
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(ACCELERATE_ROOT),
            environment.get("PYTHONPATH", ""),
        )
        if part
    )
    completed = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script)],
        cwd=ACCELERATE_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr


def test_proof_plugin_import_has_no_root_initialization_side_effects() -> None:
    _run(
        """
        import builtins
        import os
        import pathlib
        import socket
        import subprocess
        import sys
        import tempfile
        import threading

        path_before = list(sys.path)
        meta_path_before = list(sys.meta_path)

        def forbidden(operation):
            def fail(*args, **kwargs):
                raise AssertionError(f"{operation} during cold plugin import")
            return fail

        original_open = builtins.open
        def guarded_open(file, mode="r", *args, **kwargs):
            if any(flag in mode for flag in ("w", "a", "x", "+")):
                raise AssertionError(f"write-open during cold import: {file!r}")
            return original_open(file, mode, *args, **kwargs)
        builtins.open = guarded_open

        os.mkdir = forbidden("os.mkdir")
        os.makedirs = forbidden("os.makedirs")
        os.remove = forbidden("os.remove")
        os.unlink = forbidden("os.unlink")
        os.rename = forbidden("os.rename")
        os.replace = forbidden("os.replace")
        pathlib.Path.mkdir = forbidden("Path.mkdir")
        pathlib.Path.touch = forbidden("Path.touch")
        pathlib.Path.unlink = forbidden("Path.unlink")
        pathlib.Path.write_bytes = forbidden("Path.write_bytes")
        pathlib.Path.write_text = forbidden("Path.write_text")
        tempfile.mkstemp = forbidden("tempfile.mkstemp")
        tempfile.mkdtemp = forbidden("tempfile.mkdtemp")
        tempfile.NamedTemporaryFile = forbidden("NamedTemporaryFile")

        socket.create_connection = forbidden("socket.create_connection")
        original_socket = socket.socket
        class GuardedSocket(original_socket):
            connect = forbidden("socket.connect")
            connect_ex = forbidden("socket.connect_ex")
        socket.socket = GuardedSocket

        subprocess.Popen = forbidden("subprocess.Popen")
        subprocess.run = forbidden("subprocess.run")
        subprocess.call = forbidden("subprocess.call")
        subprocess.check_call = forbidden("subprocess.check_call")
        subprocess.check_output = forbidden("subprocess.check_output")
        threading.Thread.start = forbidden("threading.Thread.start")

        import ipfs_accelerate_py.testing.proof_reuse.plugin as plugin
        import ipfs_accelerate_py as package

        assert plugin.PLUGIN_NAME == "ipfs-proof-reuse"
        assert sys.path == path_before
        assert sys.meta_path == meta_path_before
        assert len(package.__all__) == 106
        assert len(package.export) == 100
        availability_only = {
            "inference_backend_manager_available",
            "llm_router_available",
            "embeddings_router_available",
            "multimodal_router_available",
            "tts_router_available",
            "voice_router_available",
        }
        assert set(package.export) == set(package.__all__) - availability_only

        forbidden_prefixes = (
            "ipfs_kit_py",
            "ipfs_datasets_py",
            "test",
            "multiformats",
            "ipfs_accelerate_py.ipfs_accelerate",
            "ipfs_accelerate_py.ipfs_multiformats",
            "ipfs_accelerate_py.container_backends",
            "ipfs_accelerate_py.inference_backend_manager",
            "ipfs_accelerate_py.github_cli",
            "ipfs_accelerate_py.llm_router",
            "ipfs_accelerate_py.embeddings_router",
            "ipfs_accelerate_py.multimodal_router",
            "ipfs_accelerate_py.voice_router",
            "ipfs_accelerate_py.worker",
        )
        loaded = [
            name
            for name in sys.modules
            if any(
                name == prefix or name.startswith(prefix + ".")
                for prefix in forbidden_prefixes
            )
        ]
        assert loaded == [], loaded

        raw_export = dict(package.export)
        assert raw_export["ipfs_multiformats_py"] is None
        assert raw_export["ModelManager"] is None
        """
    )


def test_scoped_support_module_imports_do_not_acquire_storage_or_tests() -> None:
    _run(
        """
        import importlib
        import sys

        storage_name = "ipfs_accelerate_py.common.storage_wrapper"
        assert storage_name not in sys.modules
        for module_name in (
            "ipfs_accelerate_py.config.config",
            "ipfs_accelerate_py.install_depends.install_depends",
            "ipfs_accelerate_py.github_cli.cache",
        ):
            importlib.import_module(module_name)
            assert storage_name not in sys.modules, module_name
            assert not any(
                name == "test" or name.startswith("test.")
                for name in sys.modules
            ), module_name
        """
    )


def test_lazy_root_access_returns_and_caches_exact_objects() -> None:
    _run(
        """
        import inspect
        import ipfs_accelerate_py as package

        assert "ipfs_accelerate_py.ipfs_multiformats" not in __import__(
            "sys"
        ).modules
        via_getattr = package.ipfs_multiformats_py
        from ipfs_accelerate_py import ipfs_multiformats_py as via_from
        from ipfs_accelerate_py.ipfs_multiformats import (
            ipfs_multiformats_py as real_multiformats,
        )
        assert via_getattr is real_multiformats
        assert via_from is real_multiformats
        assert package.export["ipfs_multiformats_py"] is real_multiformats
        assert package.ipfs_multiformats_py is real_multiformats
        assert via_getattr.__module__ == real_multiformats.__module__
        assert via_getattr.__name__ == real_multiformats.__name__
        assert via_getattr.__qualname__ == real_multiformats.__qualname__
        assert inspect.signature(via_getattr) == inspect.signature(
            real_multiformats
        )
        assert issubclass(via_getattr, real_multiformats)

        from ipfs_accelerate_py import config as via_config_from
        from ipfs_accelerate_py.config.config import config as real_config
        assert via_config_from is real_config
        assert package.config is real_config
        assert package.export["config"] is real_config

        from ipfs_accelerate_py import install_depends as via_install_from
        real_install = __import__("importlib").import_module(
            "ipfs_accelerate_py.install_depends.install_depends"
        )
        assert via_install_from is real_install
        assert package.install_depends is real_install
        assert package.export["install_depends"] is real_install

        from ipfs_accelerate_py import backends as via_backends_from
        from ipfs_accelerate_py.container_backends import backends as real_backends
        assert via_backends_from is real_backends
        assert package.backends is real_backends
        assert package.export["backends"] is real_backends
        """
    )


def test_missing_multiformats_is_bounded_then_recovers_after_install() -> None:
    _run(
        """
        import importlib.abc
        import importlib
        import sys

        class BlockMultiformats(importlib.abc.MetaPathFinder):
            enabled = True
            attempts = 0

            def find_spec(self, fullname, path=None, target=None):
                if self.enabled and (
                    fullname == "multiformats"
                    or fullname.startswith("multiformats.")
                    or fullname == "ipfs_accelerate_py.ipfs_multiformats"
                ):
                    self.attempts += 1
                    raise ModuleNotFoundError(
                        f"blocked optional provider: {fullname}",
                        name=fullname,
                    )
                return None

        blocker = BlockMultiformats()
        sys.meta_path.insert(0, blocker)
        import ipfs_accelerate_py as package
        from ipfs_accelerate_py import ipfs_multiformats_py

        assert ipfs_multiformats_py is None
        assert package.ipfs_multiformats_py is None
        assert package.export["ipfs_multiformats_py"] is None
        assert dict(package.export)["ipfs_multiformats_py"] is None
        first_attempts = blocker.attempts
        assert first_attempts > 0

        # Immediate repeated reads use the bounded negative result.
        assert package.ipfs_multiformats_py is None
        assert blocker.attempts == first_attempts

        # A controlled installer/import changes the in-process capability
        # state. The next explicit read retries immediately and caches the
        # exact real class.
        blocker.enabled = False
        importlib.import_module("multiformats")
        recovered = package.ipfs_multiformats_py
        real = importlib.import_module(
            "ipfs_accelerate_py.ipfs_multiformats"
        ).ipfs_multiformats_py
        assert recovered is real
        assert package.export["ipfs_multiformats_py"] is real
        """
    )


def test_install_depends_exact_module_and_injected_first_use() -> None:
    _run(
        """
        import importlib
        import sys
        import ipfs_accelerate_py as package

        implementation_name = (
            "ipfs_accelerate_py.install_depends.install_depends"
        )
        storage_name = "ipfs_accelerate_py.common.storage_wrapper"
        assert implementation_name not in sys.modules
        assert storage_name not in sys.modules

        from ipfs_accelerate_py import install_depends as root_value
        implementation = importlib.import_module(implementation_name)
        package_surface = importlib.import_module(
            "ipfs_accelerate_py.install_depends"
        )
        assert root_value is implementation
        assert package.install_depends is implementation
        assert package.export["install_depends"] is implementation
        assert (
            package_surface.install_depends_py
            is implementation.install_depends_py
        )
        assert storage_name not in sys.modules

        injected_storage = object()
        instance = package_surface.install_depends_py(
            {"storage": injected_storage},
            {},
        )
        assert instance.storage is injected_storage
        assert storage_name not in sys.modules
        """
    )


def test_skip_core_preserves_root_manifests_without_core_imports() -> None:
    _run(
        """
        import sys
        import ipfs_accelerate_py as package

        expected_export = {
            "backends",
            "config",
            "install_depends",
            "ipfs_accelerate_py",
            "worker",
            "ipfs_multiformats_py",
            "get_instance",
            "accelerate_with_browser",
            "WebNNWebGPUAccelerator",
            "get_accelerator",
            "webnn_webgpu_available",
            "ModelManager",
            "get_default_model_manager",
            "model_manager_available",
            "EndpointContract",
            "SpaceRuntimeInfo",
            "OutputBackend",
            "LocalFileSystemBackend",
            "HFBucketBackend",
            "HFBucketBackendError",
            "HFSpaceClient",
            "RefreshableGradioFile",
            "BatchState",
            "BatchProcessor",
            "is_hf_space_transport_error",
            "is_retryable_hf_space_error",
            "is_stale_gradio_file_error",
            "normalize_api_name",
        }
        assert set(package.export) == expected_export
        assert len(package.__all__) == 106
        assert len(package.__all__) == len(set(package.__all__))
        assert package.worker is None
        assert package.config is None
        assert package.install_depends is None
        assert package.ipfs_multiformats_py is None
        assert package.accelerate_with_browser is None
        assert package.WebNNWebGPUAccelerator is None
        assert package.get_accelerator is None
        assert package.webnn_webgpu_available is False
        assert not any(
            name == "ipfs_accelerate_py.ipfs_multiformats"
            or name.startswith("ipfs_accelerate_py.ipfs_multiformats.")
            for name in sys.modules
        )
        """,
        skip_core=True,
    )


@pytest.mark.parametrize(
    "availability_name",
    [
        "inference_backend_manager_available",
        "llm_router_available",
        "embeddings_router_available",
        "multimodal_router_available",
        "tts_router_available",
        "voice_router_available",
    ],
)
def test_non_exported_availability_names_remain_in_all(
    availability_name: str,
) -> None:
    import ipfs_accelerate_py as package

    assert availability_name in package.__all__
    assert availability_name not in package.export
