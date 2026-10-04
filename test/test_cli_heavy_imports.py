"""Regression tests for the main CLI's deferred runtime imports."""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from ipfs_accelerate_py import cli


def test_load_heavy_imports_constructs_operations_from_primary_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The primary import path must bind every constructor before using it."""

    shared_module = ModuleType("ipfs_accelerate_py.shared")

    class SharedCore:
        pass

    def operation_type(name: str) -> type[object]:
        class Operation:
            def __init__(self, core: SharedCore) -> None:
                self.core = core

        Operation.__name__ = name
        return Operation

    operation_names = (
        "InferenceOperations",
        "FileOperations",
        "ModelOperations",
        "NetworkOperations",
        "QueueOperations",
        "TestOperations",
    )
    shared_module.SharedCore = SharedCore
    for name in operation_names:
        setattr(shared_module, name, operation_type(name))

    server_module = ModuleType("ipfs_accelerate_py.mcp_server.server")

    class IPFSAccelerateMCPServer:
        pass

    server_module.IPFSAccelerateMCPServer = IPFSAccelerateMCPServer
    monkeypatch.setitem(sys.modules, "ipfs_accelerate_py.shared", shared_module)
    monkeypatch.setitem(
        sys.modules,
        "ipfs_accelerate_py.mcp_server.server",
        server_module,
    )

    for name in (
        "HAVE_CORE",
        "shared_core",
        "inference_ops",
        "file_ops",
        "model_ops",
        "network_ops",
        "queue_ops",
        "test_ops",
        "IPFSAccelerateMCPServer",
    ):
        monkeypatch.setattr(cli, name, None)

    cli._load_heavy_imports()

    assert cli.HAVE_CORE is True
    assert type(cli.shared_core) is SharedCore
    assert cli.IPFSAccelerateMCPServer is IPFSAccelerateMCPServer
    for operation in (
        cli.inference_ops,
        cli.file_ops,
        cli.model_ops,
        cli.network_ops,
        cli.queue_ops,
        cli.test_ops,
    ):
        assert operation.core is cli.shared_core
