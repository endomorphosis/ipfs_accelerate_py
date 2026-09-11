"""Explicit DuckDB runtime binding for an isolated retained capture process.

The manifest selects code, never native process/database authority. Only the
named DuckDB package and extension are loaded; user-site hooks and unrelated
packages are not enabled. No dependency is installed or fetched.
"""
from __future__ import annotations

import fcntl
import hashlib
import importlib.abc
import importlib.machinery
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

SCHEMA = "spar/capture-python-runtime@1"
MAX_FILE_BYTES = 256 * 1024 * 1024
_ACTIVE_RUNTIME = None


class CaptureRuntimeDenied(RuntimeError):
    pass


def _require(value, reason):
    if not value:
        raise CaptureRuntimeDenied(reason)


def _bytes(path):
    _require(path.is_absolute() and path.resolve() == path, "runtime_path_not_canonical")
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        _require(stat.S_ISREG(before.st_mode) and before.st_uid in {0, os.geteuid()}
                 and before.st_size <= MAX_FILE_BYTES, "runtime_file_not_admitted")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            data = stream.read(MAX_FILE_BYTES + 1)
        after = os.fstat(descriptor)
        fields = lambda value: (value.st_dev, value.st_ino, value.st_size,
                                value.st_mtime_ns, value.st_ctime_ns)
        _require(fields(before) == fields(after) and len(data) == before.st_size,
                 "runtime_file_changed")
        return data
    finally:
        os.close(descriptor)


def observed_runtime():
    """Export the already selected native interpreter's DuckDB code for review."""
    import duckdb
    import _duckdb

    root = Path(duckdb.__file__).resolve().parent.parent
    extension = Path(_duckdb.__file__).resolve()
    _require(extension.parent == root, "runtime_extension_package_roots_differ")
    metadata = root / ("duckdb-" + duckdb.__version__ + ".dist-info/METADATA")
    paths = sorted((root / "duckdb").rglob("*.py")) + [extension, metadata]
    _require(1 <= len(paths) <= 128, "runtime_file_population_invalid")
    return {
        "schema": SCHEMA,
        "interpreter": str(Path(sys.executable).resolve()),
        "interpreter_sha256": hashlib.sha256(_bytes(Path("/proc/self/exe").resolve())).hexdigest(),
        "python_cache_tag": sys.implementation.cache_tag,
        "duckdb_version": duckdb.__version__,
        "package_root": str(root),
        "extension": extension.name,
        "distribution_metadata": str(metadata.relative_to(root)),
        "files": {str(path.relative_to(root)): hashlib.sha256(_bytes(path)).hexdigest()
                  for path in paths},
    }


class _SourceLoader(importlib.abc.Loader):
    def __init__(self, path, data):
        self.path, self.data = path, data

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        module.__file__ = str(self.path)
        exec(compile(self.data, str(self.path), "exec"), module.__dict__)


class BoundDuckDBRuntime(importlib.abc.MetaPathFinder):
    """Keep selected code and extension bytes bound in this process."""
    def __init__(self, manifest):
        _require(type(manifest) is dict and set(manifest) == {
            "schema", "interpreter", "interpreter_sha256", "python_cache_tag", "duckdb_version",
            "package_root", "extension", "distribution_metadata", "files"}, "runtime_manifest_fields_invalid")
        _require(manifest["schema"] == SCHEMA
                 and manifest["interpreter"] == str(Path(sys.executable).resolve())
                 and manifest["python_cache_tag"] == sys.implementation.cache_tag,
                 "runtime_interpreter_binding_differs")
        from ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema import (
            PINNED_DUCKDB_VERSION_PREFIX,
        )
        _require(type(manifest["duckdb_version"]) is str
                 and re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+[A-Za-z0-9.+-]*", manifest["duckdb_version"])
                 and manifest["duckdb_version"].startswith(PINNED_DUCKDB_VERSION_PREFIX),
                 "runtime_duckdb_version_outside_native_profile")
        self.manifest = json.loads(json.dumps(manifest))
        self.root = Path(manifest["package_root"])
        files = manifest["files"]
        _require(type(files) is dict and 2 <= len(files) <= 128, "runtime_files_invalid")
        extension = manifest["extension"]
        _require(type(extension) is str and any(
            extension == "_duckdb" + suffix for suffix in importlib.machinery.EXTENSION_SUFFIXES),
            "runtime_extension_name_invalid")
        metadata = manifest["distribution_metadata"]
        _require(metadata == "duckdb-" + manifest["duckdb_version"] + ".dist-info/METADATA"
                 and metadata in files and extension in files and "duckdb/__init__.py" in files,
                 "runtime_required_files_missing")
        for name, digest in files.items():
            _require(type(name) is str and (name in {extension, metadata} or re.fullmatch(
                r"duckdb/(?:[A-Za-z_][A-Za-z_0-9]*/)*[A-Za-z_][A-Za-z_0-9]*\.py", name))
                and type(digest) is str and re.fullmatch(r"[0-9a-f]{64}", digest),
                "runtime_file_binding_invalid")
        self.extension_fd = None
        self.require_current()

    def require_current(self):
        interpreter = Path("/proc/self/exe").resolve()
        _require(str(interpreter) == self.manifest["interpreter"]
                 and hashlib.sha256(_bytes(interpreter)).hexdigest() == self.manifest["interpreter_sha256"],
                 "runtime_interpreter_hash_changed")
        for name, expected in self.manifest["files"].items():
            _require(hashlib.sha256(_bytes(self.root / name)).hexdigest() == expected,
                     "runtime_dependency_hash_changed")

    def _source(self, relative):
        expected = self.manifest["files"].get(relative)
        _require(expected is not None, "runtime_unbound_dependency_import")
        path = self.root / relative
        data = _bytes(path)
        _require(hashlib.sha256(data).hexdigest() == expected, "runtime_dependency_hash_changed")
        return path, data

    def find_distributions(self, context=importlib.metadata.DistributionFinder.Context()):
        if context.name not in (None, "duckdb"):
            return iter(())
        _path, raw = self._source(self.manifest["distribution_metadata"])
        class Distribution(importlib.metadata.Distribution):
            def read_text(self, filename):
                return raw.decode("utf-8") if filename == "METADATA" else None
            def locate_file(self, path):
                raise CaptureRuntimeDenied("runtime_distribution_file_lookup_not_admitted")
        return iter((Distribution(),))

    def find_spec(self, fullname, path=None, target=None):
        if fullname == "_duckdb":
            source, data = self._source(self.manifest["extension"])
            descriptor = os.memfd_create("spar-bound-duckdb", os.MFD_CLOEXEC | os.MFD_ALLOW_SEALING)
            try:
                with os.fdopen(descriptor, "wb", closefd=False) as stream:
                    stream.write(data)
                fcntl.fcntl(descriptor, fcntl.F_ADD_SEALS, fcntl.F_SEAL_WRITE
                            | fcntl.F_SEAL_GROW | fcntl.F_SEAL_SHRINK | fcntl.F_SEAL_SEAL)
                loader = importlib.machinery.ExtensionFileLoader(fullname, f"/proc/self/fd/{descriptor}")
                self.extension_fd = descriptor
                return importlib.util.spec_from_file_location(fullname, loader.path, loader=loader)
            except BaseException:
                os.close(descriptor)
                raise
        if fullname == "duckdb" or fullname.startswith("duckdb."):
            relative = fullname.replace(".", "/")
            package = relative + "/__init__.py" in self.manifest["files"]
            relative += "/__init__.py" if package else ".py"
            source, data = self._source(relative)
            return importlib.util.spec_from_file_location(
                fullname, source, loader=_SourceLoader(source, data),
                submodule_search_locations=[str(source.parent)] if package else None,
            )
        return None

    def load(self):
        for name, module in tuple(sys.modules.items()):
            if name == "_duckdb" or name == "duckdb" or name.startswith("duckdb."):
                path = Path(getattr(module, "__file__", "")).resolve()
                _require(path.is_relative_to(self.root)
                         and str(path.relative_to(self.root)) in self.manifest["files"],
                         "runtime_preloaded_dependency_differs")
        sys.meta_path.insert(0, self)
        try:
            import duckdb
            _require(duckdb.__version__ == self.manifest["duckdb_version"], "runtime_duckdb_version_differs")
            self.require_current()
            return duckdb
        except BaseException:
            sys.meta_path.remove(self)
            raise


def admit_runtime(manifest_path=None, manifest_sha256=None):
    """Eagerly qualify the exact current interpreter before native inspection."""
    global _ACTIVE_RUNTIME
    if manifest_path is None:
        _require(manifest_sha256 is None, "runtime_manifest_binding_incomplete")
        runtime = _ACTIVE_RUNTIME or BoundDuckDBRuntime(observed_runtime())
    else:
        data = _bytes(Path(manifest_path))
        _require(type(manifest_sha256) is str and re.fullmatch(r"[0-9a-f]{64}", manifest_sha256)
                 and hashlib.sha256(data).hexdigest() == manifest_sha256,
                 "runtime_manifest_digest_differs")
        runtime = BoundDuckDBRuntime(json.loads(data))
    if _ACTIVE_RUNTIME is not None:
        _require(runtime.manifest == _ACTIVE_RUNTIME.manifest, "runtime_selection_changed")
        runtime = _ACTIVE_RUNTIME
        runtime.require_current()
        import duckdb
    else:
        duckdb = runtime.load()
        _ACTIVE_RUNTIME = runtime
    from . import spar_legacy_capture, spar_legacy_import_plan, spar_legacy_origin, spar_merge_owner
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import connect_duckdb_with_policy
    _require(all(callable(value) for value in (
        spar_legacy_capture.RetainedNativeLegacySession,
        spar_legacy_import_plan.produce_offline_import_plan,
        spar_legacy_origin.install_captured_queue, spar_merge_owner.prepare_offline_clone,
    )), "runtime_capture_prepare_imports_incomplete")
    connection = connect_duckdb_with_policy(duckdb, ":memory:")
    try:
        _require(spar_merge_owner.inventory(connection) == {}, "runtime_memory_inventory_unexpected")
        _require(spar_merge_owner.load_control_plane_catalog().latest_version > 0,
                 "runtime_migration_catalog_unavailable")
    finally:
        connection.close()
    runtime.require_current()
    return runtime
