"""Tests for the cold proof-reuse pytest plugin shell."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
from ipfs_accelerate_py.testing.proof_reuse.config import (
    PROOF_REUSE_MODE_ENV,
    PROOF_REUSE_REQUIRED_AUDIT_ENV,
    ProofReuseConfig,
    ProofReuseMode,
)
from ipfs_accelerate_py.testing.proof_reuse.plugin import (
    CONFIG_ATTRIBUTE,
    ITEM_METADATA_ATTRIBUTE,
    ProofReuseItemMetadata,
    collect_item_metadata,
    pytest_collection_modifyitems,
)


class _Marker:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs


class _Item:
    def __init__(self, nodeid, markers=None):
        self.nodeid = nodeid
        self._markers = markers or {}

    def get_closest_marker(self, name):
        values = self._markers.get(name, ())
        return values[0] if values else None

    def iter_markers(self, name):
        return iter(self._markers.get(name, ()))


class _Config:
    pass


@pytest.mark.parametrize(
    ("mode", "reads", "skips", "writes"),
    [
        ("off", False, False, False),
        ("shadow", True, False, False),
        ("read", True, True, False),
        ("write", False, False, True),
        ("readwrite", True, True, True),
    ],
)
def test_config_modes_have_explicit_capabilities(mode, reads, skips, writes):
    config = ProofReuseConfig.resolve(
        environ={PROOF_REUSE_MODE_ENV: mode},
    )

    assert config.mode is ProofReuseMode(mode)
    assert config.reads_candidates is reads
    assert config.may_skip is skips
    assert config.writes_receipts is writes


def test_invalid_environment_configuration_degrades_to_off():
    config = ProofReuseConfig.resolve(
        environ={
            PROOF_REUSE_MODE_ENV: "required-audit",
            PROOF_REUSE_REQUIRED_AUDIT_ENV: "not-a-boolean",
        },
    )

    assert config.mode is ProofReuseMode.OFF
    assert config.required_audit is False
    assert config.source == "invalid"
    assert config.configuration_error


def test_required_audit_is_separate_from_operational_mode():
    config = ProofReuseConfig.resolve(
        environ={
            PROOF_REUSE_MODE_ENV: "read",
            PROOF_REUSE_REQUIRED_AUDIT_ENV: "true",
        },
    )

    assert config.mode is ProofReuseMode.READ
    assert config.required_audit is True


def test_off_collection_hook_is_behaviorally_inert():
    config = _Config()
    setattr(config, CONFIG_ATTRIBUTE, ProofReuseConfig())
    item = _Item("test_direct.py::test_one")

    pytest_collection_modifyitems(config, [item])

    assert not hasattr(item, ITEM_METADATA_ATTRIBUTE)


def test_enabled_collection_uses_direct_nodes_and_marker_metadata():
    config = _Config()
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.SHADOW),
    )
    item = _Item(
        "test_direct.py::test_one[param]",
        {
            "proof_reuse_disabled": (_Marker(reason="external state"),),
            "proof_reuse_effects": (
                _Marker("env", "filesystem"),
                _Marker(adapters=("env", "subprocess")),
            ),
        },
    )

    pytest_collection_modifyitems(config, [item])

    assert getattr(item, ITEM_METADATA_ATTRIBUTE) == ProofReuseItemMetadata(
        nodeid="test_direct.py::test_one[param]",
        disabled=True,
        disabled_reason="external state",
        effect_adapters=("env", "filesystem", "subprocess"),
    )
    assert collect_item_metadata(item).nodeid == item.nodeid


def test_cold_import_does_not_import_optional_providers(tmp_path):
    script = tmp_path / "cold_import.py"
    script.write_text(
        """
import builtins
import json
import sys

blocked = (
    "ipfs_datasets_py",
    "ipfs_kit_py",
    "multiformats",
    "pytest",
    "torch",
)
real_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name in blocked or name.startswith(tuple(part + "." for part in blocked)):
        raise AssertionError("optional import attempted: " + name)
    return real_import(name, *args, **kwargs)
builtins.__import__ = guarded_import

import ipfs_accelerate_py.testing.proof_reuse.plugin
print(json.dumps(sorted(name for name in sys.modules if name.startswith(blocked))))
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (package_root, environment.get("PYTHONPATH", ""))
        if part
    )

    completed = subprocess.run(
        [sys.executable, str(script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout) == []


def test_enabled_plugin_collects_a_direct_node_without_a_registry(tmp_path):
    test_file = tmp_path / "test_direct_node.py"
    test_file.write_text(
        """
import pytest

from ipfs_accelerate_py.testing.proof_reuse.plugin import get_item_metadata

@pytest.mark.proof_reuse_disabled(reason="direct-node-check")
@pytest.mark.proof_reuse_effects("environment")
def test_direct(request):
    metadata = get_item_metadata(request.node)
    assert metadata is not None
    assert metadata.nodeid.endswith("test_direct_node.py::test_direct")
    assert metadata.disabled is True
    assert metadata.disabled_reason == "direct-node-check"
    assert metadata.effect_adapters == ("environment",)
""",
        encoding="utf-8",
    )
    environment = dict(os.environ)
    package_root = str(Path(__file__).resolve().parents[2])
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (package_root, environment.get("PYTHONPATH", ""))
        if part
    )
    environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-p",
            "ipfs_accelerate_py.testing.proof_reuse.plugin",
            "--proof-reuse-mode=shadow",
            f"{test_file}::test_direct",
            "-q",
        ],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "1 passed" in completed.stdout


def test_module_reload_is_pure(monkeypatch):
    monkeypatch.setenv(PROOF_REUSE_MODE_ENV, "readwrite")
    module = importlib.import_module(
        "ipfs_accelerate_py.testing.proof_reuse.plugin"
    )

    reloaded = importlib.reload(module)

    assert not hasattr(reloaded, "PROOF_REUSE_CONFIG")
