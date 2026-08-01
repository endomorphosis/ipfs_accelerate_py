"""Tests for the cold proof-reuse pytest plugin shell."""

from __future__ import annotations

import importlib
import builtins
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
    IDENTITY_SERVICES_ATTRIBUTE,
    ITEM_METADATA_ATTRIBUTE,
    ProofReuseItemMetadata,
    collect_item_metadata,
    pytest_collection_modifyitems,
    set_proof_reuse_identity_services,
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


def test_identity_services_setter_validates_without_calling_providers():
    from ipfs_accelerate_py.testing.proof_reuse.item_identity import (
        ItemIdentityAssemblyServices,
    )

    calls = []
    services = ItemIdentityAssemblyServices(
        repository_forest_provider=lambda item: calls.append(item)
    )
    config = _Config()

    set_proof_reuse_identity_services(config, services)

    assert getattr(config, IDENTITY_SERVICES_ATTRIBUTE) is services
    assert calls == []
    with pytest.raises(TypeError, match="ItemIdentityAssemblyServices"):
        set_proof_reuse_identity_services(config, object())


def test_enabled_collection_calls_automatic_identity_for_direct_node(
    monkeypatch,
):
    from ipfs_accelerate_py.testing.proof_reuse import item_identity
    from ipfs_accelerate_py.testing.proof_reuse.item_identity import (
        ItemIdentityAssemblyServices,
    )

    config = _Config()
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.SHADOW),
    )
    services = ItemIdentityAssemblyServices()
    set_proof_reuse_identity_services(config, services)
    item = _Item("test_direct.py::test_one")
    calls = []

    def record(direct_item, direct_services):
        calls.append((direct_item, direct_services))

    monkeypatch.setattr(
        item_identity,
        "assemble_and_attach_item_identity",
        record,
    )

    pytest_collection_modifyitems(config, [item])

    assert calls == [(item, services)]
    assert getattr(item, ITEM_METADATA_ATTRIBUTE).nodeid == item.nodeid


def test_enabled_collection_without_identity_di_attaches_typed_run(
    tmp_path,
):
    from ipfs_accelerate_py.testing.proof_reuse.item_identity import (
        ITEM_IDENTITY_RESULT_ATTRIBUTE,
        ItemIdentityAssemblyReason,
    )

    source = tmp_path / "test_direct.py"
    source.write_text("def test_one():\n    assert True\n", encoding="utf-8")
    config = _Config()
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.SHADOW),
    )
    item = _Item("test_direct.py::test_one")
    item.path = source
    item.originalname = "test_one"
    item.name = "test_one"
    item.fixturenames = ()
    item.cls = None
    item.own_markers = []

    pytest_collection_modifyitems(config, [item])

    result = getattr(item, ITEM_IDENTITY_RESULT_ATTRIBUTE)
    assert result.reason is ItemIdentityAssemblyReason.PROVIDER_UNAVAILABLE
    assert result.stage == "repository_forest"
    assert result.action == "RUN"
    assert result.authorizes_skip is False
    assert not hasattr(item, "_ipfs_proof_reuse_lookup_request")
    assert not any(
        getattr(marker, "name", "") == "skip" for marker in item.own_markers
    )
    assert item._ipfs_proof_reuse_decision.action.value == "RUN"


def test_off_collection_does_not_import_or_call_identity_assembler(
    monkeypatch,
):
    config = _Config()
    setattr(config, CONFIG_ATTRIBUTE, ProofReuseConfig())
    item = _Item("test_direct.py::test_one")
    imported = []
    real_import = builtins.__import__

    def guarded(name, *args, **kwargs):
        if name.endswith("item_identity"):
            imported.append(name)
            raise AssertionError("off mode imported item identity")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)

    pytest_collection_modifyitems(config, [item])

    assert imported == []
    assert not hasattr(item, ITEM_METADATA_ATTRIBUTE)


def test_manual_exact_identity_is_untouched_by_automatic_boundary(
    tmp_path,
):
    from ipfs_accelerate_py.testing.proof_reuse.item_identity import (
        ITEM_IDENTITY_RESULT_ATTRIBUTE,
        ItemIdentityAssemblyReason,
    )

    source = tmp_path / "test_direct.py"
    source.write_text("def test_one():\n    assert True\n", encoding="utf-8")
    config = _Config()
    setattr(
        config,
        CONFIG_ATTRIBUTE,
        ProofReuseConfig(mode=ProofReuseMode.SHADOW),
    )
    item = _Item("test_direct.py::test_one")
    item.path = source
    item.originalname = "test_one"
    item.name = "test_one"
    item.fixturenames = ()
    item.cls = None
    locator = object()
    execution_key = object()
    item._ipfs_proof_reuse_locator = locator
    item._ipfs_proof_reuse_execution_key = execution_key

    pytest_collection_modifyitems(config, [item])

    result = getattr(item, ITEM_IDENTITY_RESULT_ATTRIBUTE)
    assert result.reason is ItemIdentityAssemblyReason.EXISTING_IDENTITY_CONFLICT
    assert item._ipfs_proof_reuse_locator is locator
    assert item._ipfs_proof_reuse_execution_key is execution_key
    assert not hasattr(item, "_ipfs_proof_reuse_lookup_request")


def test_plugin_source_has_no_per_file_identity_registry():
    from ipfs_accelerate_py.testing.proof_reuse import plugin

    source = Path(plugin.__file__).read_text(encoding="utf-8")
    for forbidden in (
        "PROOF_REUSE_TEST_LIST",
        "proof_reuse_test_paths",
        "TEST_PATH_REGISTRY",
        "allowed_test_files",
    ):
        assert forbidden not in source
