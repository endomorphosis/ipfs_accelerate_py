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
    full_manifest = source / "dependency-manifest.json"
    full_manifest.write_bytes(driver.wire(manifest))
    binding = driver.validate_runtime_dependencies(source, {"runtime_dependencies": manifest,
        "runtime_dependencies_reference": driver.identity(full_manifest)})
    assert binding["canonical_observation_sha256"] == driver.sha(driver.wire(manifest["observation"]))
    assert binding["bytecode_cache_policy"] == "empty temporary prefix with writes disabled"
    assert cache.read_bytes() == poisoned


def test_large_complete_manifest_has_compact_content_address_binding(tmp_path):
    driver_path = HELPER.with_name("legal_autoformalization_source_views.py")
    spec = importlib.util.spec_from_file_location("large_dependency_binding_test", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    observed = {"distributions": [{"metadata": "complete retained observation" * 4000}],
        "modules": [{"module": "fixture"}], "file_count": 3130, "file_bytes": 265827038,
        "no_user_site": 1, "bytecode_cache_policy": "empty temporary prefix with writes disabled"}
    manifest = {"schema": dependencies.SCHEMA, "roots": [str(tmp_path)], "observation": observed}
    path = tmp_path / "complete-runtime-manifest.json"
    path.write_bytes(driver.wire(manifest))
    job = {"runtime_dependencies": manifest, "runtime_dependencies_reference": driver.identity(path)}
    binding = driver.runtime_dependency_binding(job)
    assert len(driver.wire(manifest)) > 100000
    assert len(driver.wire(binding)) < 2048
    # Preserve the prior large full-report result budget without lifting caps
    # or removing a byte from the independently retained observation.
    result = {"complete_real_report": "x" * 855973, "bindings": binding, "execution": binding}
    assert len(driver.wire(result)) + 1 < driver.MAX_RESULT_BYTES
    assert driver.read_json(path) == manifest
    assert binding["canonical_manifest_sha256"] == driver.sha(driver.wire(manifest))


def test_tampered_full_manifest_reference_cannot_be_resealed(tmp_path):
    driver_path = HELPER.with_name("legal_autoformalization_source_views.py")
    spec = importlib.util.spec_from_file_location("tampered_dependency_binding_test", driver_path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    manifest = {"observation": {"distributions": [], "modules": [], "file_count": 0,
        "file_bytes": 0, "no_user_site": 1, "bytecode_cache_policy": "fixture"}, "roots": []}
    path = tmp_path / "complete-runtime-manifest.json"
    path.write_bytes(driver.wire(manifest))
    job = {"runtime_dependencies": manifest, "runtime_dependencies_reference": driver.identity(path)}
    altered = copy.deepcopy(job)
    altered["runtime_dependencies_reference"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="file identity differs"):
        driver.runtime_dependency_binding(altered)
    unrelated = copy.deepcopy(manifest)
    unrelated["observation"]["file_count"] = 99
    path.write_bytes(driver.wire(unrelated))
    altered["runtime_dependencies_reference"] = driver.identity(path)
    with pytest.raises(ValueError, match="content differs"):
        driver.runtime_dependency_binding(altered)
