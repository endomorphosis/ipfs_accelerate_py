"""Dependency qualification binds local files and rejects source/import drift."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import py_compile
import sys
import tempfile
from types import SimpleNamespace

import pytest


HELPER = Path(__file__).resolve().parents[2] / "scripts/legal_autoformalization_runtime_dependencies.py"
spec = importlib.util.spec_from_file_location("runtime_dependency_contract_test", HELPER)
dependencies = importlib.util.module_from_spec(spec)
spec.loader.exec_module(dependencies)


@pytest.fixture
def local_runtime(tmp_path, monkeypatch):
    root = tmp_path / "site-packages"
    package = root / "fixture_native_plugin"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__version__ = '1.0'\n")
    metadata = root / "fixture_native_plugin-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text("Metadata-Version: 2.1\nName: fixture-native-plugin\nVersion: 1.0\n")
    (metadata / "RECORD").write_text(
        "fixture_native_plugin/__init__.py,,\n"
        "fixture_native_plugin-1.0.dist-info/METADATA,,\n"
        "fixture_native_plugin-1.0.dist-info/RECORD,,\n")
    source = tmp_path / "datasets"
    source.mkdir()
    (source / "ipfs_datasets_py").mkdir()
    (source / "ipfs_datasets_py/__init__.py").write_text("")
    bytecode = tmp_path / "empty-bytecode"
    bytecode.mkdir()
    monkeypatch.setattr(sys, "pycache_prefix", str(bytecode))
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.setattr(dependencies.importlib.util, "find_spec",
        lambda value: SimpleNamespace(origin=str(source / "ipfs_datasets_py/__init__.py")))
    monkeypatch.delitem(__import__("sys").modules, "fixture_native_plugin", raising=False)
    observed = dependencies.observe([str(root)], ["fixture-native-plugin"],
                                    ["fixture_native_plugin"], str(source))
    manifest = {"schema": dependencies.SCHEMA, "roots": [str(root)], "observation": observed}
    yield source, root, package, manifest
    __import__("sys").modules.pop("fixture_native_plugin", None)


def test_curated_import_closure_records_actual_origins_versions_and_bytes(local_runtime):
    source, root, package, manifest = local_runtime
    result = dependencies.verify(manifest, str(source))
    assert result == manifest["observation"]
    assert result["no_user_site"] == 1 and result["optimize"] == 0
    assert result["distributions"][0]["version"] == "1.0"
    assert result["distributions"][0]["files"] == 3
    assert result["modules"][0]["path"] == str(package / "__init__.py")
    assert result["modules"][0]["module_version"] == "1.0"


def test_changed_dependency_is_refused_before_its_code_runs(local_runtime):
    source, _, package, manifest = local_runtime
    (package / "__init__.py").write_text("raise AssertionError('changed code must not execute')\n")
    with pytest.raises(ValueError, match="hash closure differ"):
        dependencies.verify(manifest, str(source))


def test_wrong_dependency_version_or_module_origin_is_refused(local_runtime):
    source, _, _, manifest = local_runtime
    for section, field, value in (("distributions", "version", "99.0"),
                                  ("modules", "path", "/unrelated/fixture_native_plugin.py")):
        altered = copy.deepcopy(manifest)
        altered["observation"][section][0][field] = value
        with pytest.raises(ValueError, match="origins, versions or hash closure differ"):
            dependencies.verify(altered, str(source))


def test_shadow_datasets_source_is_refused(local_runtime, monkeypatch):
    source, _, _, manifest = local_runtime
    monkeypatch.setattr(dependencies.importlib.util, "find_spec",
                        lambda value: SimpleNamespace(origin="/unrelated/ipfs_datasets_py/__init__.py"))
    with pytest.raises(ValueError, match="source origin differs"):
        dependencies.verify(manifest, str(source))


def test_missing_dependency_and_unbounded_root_are_refused(local_runtime):
    source, root, _, _ = local_runtime
    with pytest.raises(ValueError, match="missing"):
        dependencies.observe([str(root)], ["missing-native-plugin"],
                             ["fixture_native_plugin"], str(source))
    for roots in ([str(root)] * 2, [str(root)] * 5, [str(root.parent)]):
        with pytest.raises(ValueError, match="roots"):
            dependencies.roots_from_manifest({"schema": dependencies.SCHEMA, "roots": roots})


def test_alias_or_hacc_dependency_root_is_refused(local_runtime):
    _, root, _, _ = local_runtime
    alias = root.parent / "alias"
    alias.symlink_to(root, target_is_directory=True)
    hacc = root.parent / "HACC/site-packages"
    hacc.mkdir(parents=True)
    for path in (alias, hacc):
        with pytest.raises(ValueError):
            dependencies.roots_from_manifest({"schema": dependencies.SCHEMA, "roots": [str(path)]})


def test_record_file_escape_and_unrecorded_package_inventory_are_refused(local_runtime):
    source, root, _, manifest = local_runtime
    metadata = root / "fixture_native_plugin-1.0.dist-info"
    (metadata / "RECORD").unlink()
    with pytest.raises(ValueError, match="no recorded file inventory"):
        dependencies.verify(manifest, str(source))


def test_undeclared_metadata_change_is_refused(local_runtime):
    source, root, _, manifest = local_runtime
    (root / "fixture_native_plugin-1.0.dist-info/UNREVIEWED").write_text("unreviewed metadata")
    with pytest.raises(ValueError, match="hash closure differ"):
        dependencies.verify(manifest, str(source))


def test_fresh_probe_ignores_unpinned_preexisting_bytecode(local_runtime, monkeypatch):
    source, root, package, manifest = local_runtime
    source_file = package / "__init__.py"
    original = source_file.read_bytes()
    before = source_file.stat()
    source_file.write_bytes(original.replace(b"1.0", b"9.0"))
    os.utime(source_file, ns=(before.st_atime_ns, before.st_mtime_ns))
    cache = package / "__pycache__" / ("__init__." + sys.implementation.cache_tag + ".pyc")
    py_compile.compile(str(source_file), cfile=str(cache), doraise=True)
    source_file.write_bytes(original)
    os.utime(source_file, ns=(before.st_atime_ns, before.st_mtime_ns))
    poisoned = cache.read_bytes()
    driver_path = HELPER.with_name("legal_autoformalization_source_views.py")
    spec = importlib.util.spec_from_file_location("bytecode_source_views_test", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    monkeypatch.setenv("PYTHONPYCACHEPREFIX", "/unrelated/ambient/cache")
    observed = driver.validate_runtime_dependencies(source, {"runtime_dependencies": manifest})
    assert observed["modules"][0]["module_version"] == "1.0"
    assert observed["bytecode_cache_policy"] == "empty temporary prefix with writes disabled"
    assert cache.read_bytes() == poisoned
