"""Bind existing local Python dependencies without enabling ambient user-site code.

This module observes imports and distribution files only. It installs nothing,
opens no database, and grants no source, semantic, model or Lean qualification.
"""
from __future__ import annotations

import hashlib
import importlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import stat
import sys

SCHEMA = "legal-autoformalization-runtime-dependencies/v1"
MAX_DISTRIBUTIONS = 32
MAX_FILES = 20000
MAX_FILE_BYTES = 1024 * 1024 * 1024
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
SYSTEM_TREE_PACKAGES = {"pygments": "pygments", "chardet": "chardet",
                        "html5lib": "html5lib", "webencodings": "webencodings"}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def wire(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode()


def name(value):
    return re.sub(r"[-_.]+", "-", value).lower()


def exact_path(value):
    path = Path(value)
    need(path.is_absolute() and ".." not in path.parts
         and str(path) == value and path.resolve(strict=True) == path,
         "Dependency path must be canonical and absolute")
    return path


def file_identity(path):
    path = exact_path(str(path))
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "rb") as stream:
        before = os.fstat(stream.fileno())
        need(stat.S_ISREG(before.st_mode) and before.st_size <= MAX_FILE_BYTES,
             "Dependency must be a bounded regular file")
        digest, count = hashlib.sha256(), 0
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            count += len(block)
            need(count <= MAX_FILE_BYTES, "Dependency file grew beyond byte cap")
            digest.update(block)
        marker = lambda entry: (entry.st_dev, entry.st_ino, entry.st_size,
                                entry.st_mtime_ns, entry.st_ctime_ns)
        need(marker(before) == marker(os.fstat(stream.fileno())) == marker(path.stat()),
             "Dependency changed while hashing")
    return {"path": str(path), "bytes": count, "sha256": digest.hexdigest()}


def roots_from_manifest(manifest):
    need(type(manifest) is dict and manifest.get("schema") == SCHEMA,
         "Explicit runtime dependency manifest required")
    roots = manifest.get("roots")
    need(type(roots) is list and 0 < len(roots) <= 4
         and all(type(value) is str for value in roots)
         and len(roots) == len(set(roots)), "Distinct bounded dependency roots required")
    paths = [exact_path(value) for value in roots]
    need(all(path.is_dir() and path.name in ("site-packages", "dist-packages")
             and "HACC" not in path.parts for path in paths),
         "Approved local dependency roots required")
    return paths


def metadata_identity(dist):
    root = exact_path(str(dist._path))
    paths = sorted(path for path in root.rglob("*") if path.is_file())
    need(0 < len(paths) <= 512, "Dependency metadata inventory exceeds finite bound")
    return {"origin": str(root), "version": dist.version,
            "files": [file_identity(path) for path in paths]}


def observe(roots, distributions, modules, datasets, *, expected=None):
    """Hash a curated importable distribution closure and record actual origins."""
    startup_modules = frozenset(sys.modules)
    need(sys.flags.no_user_site == 1 and sys.flags.optimize == 0,
         "Runtime dependency probe requires disabled user site and assertions")
    need(sys.dont_write_bytecode and sys.pycache_prefix
         and Path(sys.pycache_prefix).is_dir() and not any(Path(sys.pycache_prefix).iterdir()),
         "Runtime dependency probe requires an empty temporary bytecode prefix and -B")
    manifest = {"schema": SCHEMA, "roots": roots}
    roots = roots_from_manifest(manifest)
    datasets = exact_path(datasets)
    need(type(distributions) is list and 0 < len(distributions) <= MAX_DISTRIBUTIONS
         and all(type(value) is str and value for value in distributions)
         and len({name(value) for value in distributions}) == len(distributions),
         "Distinct bounded dependency distribution names required")
    need(type(modules) is list and 0 < len(modules) <= 32
         and len(modules) == len(set(modules))
         and all(type(value) is str and re.fullmatch(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*", value)
                 for value in modules), "Distinct bounded dependency modules required")
    # Metadata discovery is restricted to the declared roots. No weights or
    # unrelated user-site packages are walked or hashed.
    available = {}
    for dist in metadata.distributions(path=[str(path) for path in roots]):
        key = name(dist.metadata["Name"])
        if key in {name(value) for value in distributions}:
            available.setdefault(key, []).append(dist)
    rows, owned_files, count, total = [], set(), 0, 0
    for value in sorted(distributions, key=name):
        need(name(value) in available, "Required local dependency is missing: " + value)
        candidates = available[name(value)]
        need(len(candidates) <= 4, "Too many duplicate dependency metadata candidates")
        # Some pre-existing roots retain old dist-info directories. Bind both
        # the active resolver choice and every same-name metadata candidate;
        # adding/reordering stale metadata cannot silently change provenance.
        dist = metadata.distribution(value)
        metadata_path = exact_path(str(dist._path))
        need(str(metadata_path) in {str(candidate._path) for candidate in candidates},
             "Active dependency metadata is outside approved roots")
        need(any(metadata_path.is_relative_to(root) for root in roots),
             "Dependency metadata escaped approved roots")
        inventory = dist.files
        inventory_source = "distribution-record"
        if inventory is None:
            # These pre-existing Ubuntu packages have no wheel RECORD.
            # Bind only their explicit pure-Python package trees and metadata;
            # do not infer or scan an arbitrary missing distribution closure.
            need(name(value) in SYSTEM_TREE_PACKAGES
                 and str(metadata_path.parent) == "/usr/lib/python3/dist-packages",
                 "Dependency has no recorded file inventory")
            package = metadata_path.parent / SYSTEM_TREE_PACKAGES[name(value)]
            need(package.is_dir(), "Existing system dependency package is missing")
            inventory = [path.relative_to(metadata_path.parent)
                         for path in package.rglob("*") if path.is_file()]
            inventory_source = "explicit-existing-system-package-tree"
        files = []
        for relative in sorted(inventory, key=str):
            if "__pycache__" in relative.parts or str(relative).endswith((".pyc", ".pyo")):
                continue
            # Wheel entry-point executables outside site-packages are not in
            # the import closure; their RECORD declarations remain hashed.
            if ".." in relative.parts:
                continue
            path = exact_path(str(Path(dist.locate_file(relative))))
            need(any(path.is_relative_to(root) for root in roots),
                 "Dependency file escaped approved roots")
            identity = file_identity(path)
            count += 1
            total += identity["bytes"]
            need(count <= MAX_FILES and total <= MAX_TOTAL_BYTES,
                 "Dependency import closure exceeds finite bounds")
            files.append({"relative_path": str(relative), **identity})
            owned_files.add(str(path))
        need(files, "Dependency import closure is empty")
        rows.append({"distribution": dist.metadata["Name"], "version": dist.version,
            "metadata_origin": str(metadata_path), "files": len(files),
            "inventory_source": inventory_source,
            "bytes": sum(row["bytes"] for row in files),
            "metadata_candidates": [metadata_identity(candidate)
                for candidate in sorted(candidates, key=lambda item: str(item._path))],
            "importable_files_sha256": hashlib.sha256(wire(files)).hexdigest()})
    interpreter = file_identity(Path(sys.executable).resolve(strict=True))
    if expected is not None:
        # Verify files before importing package code; changed dependencies must
        # not get an opportunity to run and then report their own provenance.
        need(rows == expected.get("distributions")
             and interpreter == expected.get("interpreter"),
             "Runtime dependency origins, versions or hash closure differ")
    module_rows = []
    for value in modules:
        loaded = importlib.import_module(value)
        path = getattr(loaded, "__file__", None)
        need(path in owned_files, "Dependency module escaped pinned file closure: " + value)
        module_rows.append({"module": value, "module_version": getattr(loaded, "__version__", None),
                            **file_identity(path)})
    # Catch optional dependency imports selected by the installed packages,
    # including native modules, rather than relying only on Requires-Dist.
    for value, loaded in tuple(sys.modules.items()):
        path = getattr(loaded, "__file__", None)
        if value not in startup_modules and path and any(
                part in Path(path).parts for part in ("site-packages", "dist-packages")):
            need(str(Path(path).resolve(strict=True)) in owned_files,
                 "Imported dependency is outside the declared closure: " + value)
    source = importlib.util.find_spec("ipfs_datasets_py")
    need(source is not None and source.origin == str(datasets / "ipfs_datasets_py/__init__.py"),
         "Datasets source origin differs from the pinned tree")
    return {"schema": SCHEMA, "roots": [str(path) for path in roots],
        "interpreter": interpreter,
        "python_version": list(sys.version_info[:3]), "no_user_site": sys.flags.no_user_site,
        "bytecode_cache_policy": "empty temporary prefix with writes disabled",
        "optimize": sys.flags.optimize, "datasets_origin": source.origin,
        "distributions": rows, "modules": module_rows,
        "file_count": count, "file_bytes": total}


def verify(manifest, datasets):
    need(type(manifest) is dict and set(manifest) == {"schema", "roots", "observation"},
         "Closed runtime dependency manifest required")
    expected = manifest["observation"]
    need(type(expected) is dict and expected.get("schema") == SCHEMA,
         "Runtime dependency observation required")
    actual = observe([str(path) for path in roots_from_manifest(manifest)],
        [row["distribution"] for row in expected["distributions"]],
        [row["module"] for row in expected["modules"]], datasets, expected=expected)
    need(actual == expected, "Runtime dependency origins, versions or hash closure differ")
    return actual


if __name__ == "__main__":
    spec = json.loads(sys.argv[1])
    if spec.get("action") == "observe":
        result = observe(spec["roots"], spec["distributions"], spec["modules"], spec["datasets"])
    else:
        result = verify(spec["manifest"], spec["datasets"])
    print(wire(result).decode("utf-8"))
