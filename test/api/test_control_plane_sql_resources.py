"""Pure filesystem tests for explicit bundled control-plane SQL resources."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.task_sources import control_plane_migrations as migrations


@pytest.fixture
def bound_sql_module(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(migrations, "_BOUND_SQL_RESOURCES", None)
    return migrations


def _sql_files(directory: Path, entries: dict[str, bytes]) -> dict[str, str]:
    directory.mkdir()
    for name, raw in entries.items():
        (directory / name).write_bytes(raw)
    return {name: hashlib.sha256(raw).hexdigest() for name, raw in entries.items()}


def test_bound_sql_frozen_text_and_catalog_inventory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound_sql_module
) -> None:
    module = bound_sql_module
    directory = tmp_path / "sql"
    entries = {name: b"SELECT 1;\r\n" for name in module.REQUIRED_PACKAGE_SQL_FILENAMES}
    hashes = _sql_files(directory, entries)
    module.bind_bundled_sql_resources(directory, hashes)
    assert module._package_sql_resources() == directory
    monkeypatch.setattr(module.resources, "files", lambda _: pytest.fail("ambient package fallback"))
    for name in entries:
        (directory / name).unlink()
    hashes.clear()
    assert module.read_bundled_sql_text("0001_control_plane.sql") == "SELECT 1;\r\n"
    assert module.load_default_catalog().latest_version == 5
    with pytest.raises(module.MigrationCatalogError, match="missing"):
        module.read_bundled_sql_text("0006_unbound.sql")
    with pytest.raises(module.MigrationCatalogError, match="filename"):
        module.read_bundled_sql_text("../0001_control_plane.sql")


def test_bound_sql_idempotent_and_cannot_rebind(tmp_path: Path, bound_sql_module) -> None:
    module = bound_sql_module
    directory = tmp_path / "sql"
    hashes = _sql_files(directory, {"0001_one.sql": b"SELECT 1;", "0002_two.sql": b"SELECT 2;"})
    module.bind_bundled_sql_resources(directory, hashes)
    (directory / "0001_one.sql").write_bytes(b"changed after binding")
    module.bind_bundled_sql_resources(directory, dict(reversed(tuple(hashes.items()))))
    assert module.read_bundled_sql_text("0001_one.sql") == "SELECT 1;"
    other = tmp_path / "other"
    _sql_files(other, {"0001_one.sql": b"SELECT 1;", "0002_two.sql": b"SELECT 2;"})
    with pytest.raises(module.MigrationCatalogError, match="already bound"):
        module.bind_bundled_sql_resources(other, hashes)
    with pytest.raises(module.MigrationCatalogError, match="already bound"):
        module.bind_bundled_sql_resources(directory, {"0001_one.sql": "0" * 64})


@pytest.mark.parametrize("name", ["../0001_one.sql", "/0001_one.sql", "one.sql", "0001_one.sql\n"])
def test_bound_sql_rejects_filename_traversal(tmp_path: Path, bound_sql_module, name: str) -> None:
    with pytest.raises(bound_sql_module.MigrationCatalogError, match="filename"):
        bound_sql_module.bind_bundled_sql_resources(tmp_path, {name: "0" * 64})


@pytest.mark.parametrize("digest", ["0" * 63, "A" * 64, "sha256:" + "0" * 64, None])
def test_bound_sql_rejects_invalid_hash(tmp_path: Path, bound_sql_module, digest) -> None:
    with pytest.raises(bound_sql_module.MigrationCatalogError, match="SHA256"):
        bound_sql_module.bind_bundled_sql_resources(tmp_path, {"0001_one.sql": digest})


def test_bound_sql_hash_failure_is_atomic(tmp_path: Path, bound_sql_module) -> None:
    module = bound_sql_module
    directory = tmp_path / "sql"
    hashes = _sql_files(directory, {"0001_one.sql": b"SELECT 1;", "0002_two.sql": b"SELECT 2;"})
    bad = dict(hashes, **{"0002_two.sql": "0" * 64})
    with pytest.raises(module.MigrationCatalogError, match="checksum"):
        module.bind_bundled_sql_resources(directory, bad)
    assert module._BOUND_SQL_RESOURCES is None
    module.bind_bundled_sql_resources(directory, hashes)
    assert module.read_bundled_sql_text("0002_two.sql") == "SELECT 2;"


def test_bound_sql_missing_or_non_utf8_file(tmp_path: Path, bound_sql_module) -> None:
    module = bound_sql_module
    directory = tmp_path / "sql"
    hashes = _sql_files(directory, {"0001_one.sql": b"\xff"})
    with pytest.raises(module.MigrationCatalogError, match="unreadable"):
        module.bind_bundled_sql_resources(directory, hashes)
    assert module._BOUND_SQL_RESOURCES is None
    (directory / "0001_one.sql").unlink()
    with pytest.raises(module.MigrationCatalogError, match="unreadable"):
        module.bind_bundled_sql_resources(directory, hashes)
    assert module._BOUND_SQL_RESOURCES is None


@pytest.mark.parametrize("kind", ["file_symlink", "directory_symlink", "hardlink", "not_directory", "relative"])
def test_bound_sql_requires_absolute_regular_unaliased_resources(
    tmp_path: Path, bound_sql_module, kind: str
) -> None:
    module = bound_sql_module
    directory = tmp_path / "sql"
    hashes = _sql_files(directory, {"0001_one.sql": b"SELECT 1;"})
    target = directory / "0001_one.sql"
    if kind == "file_symlink":
        raw_file = tmp_path / "raw.sql"
        target.rename(raw_file)
        target.symlink_to(raw_file)
    elif kind == "directory_symlink":
        alias = tmp_path / "alias"
        alias.symlink_to(directory, target_is_directory=True)
        directory = alias
    elif kind == "hardlink":
        os.link(target, tmp_path / "hardlinked.sql")
    elif kind == "not_directory":
        directory = target
    else:
        directory = Path("relative/sql")
    with pytest.raises(module.MigrationCatalogError):
        module.bind_bundled_sql_resources(directory, hashes)
    assert module._BOUND_SQL_RESOURCES is None


@pytest.mark.parametrize("count", [0, 33])
def test_bound_sql_file_count_bounds(tmp_path: Path, bound_sql_module, count: int) -> None:
    hashes = {f"{i:04d}_resource.sql": "0" * 64 for i in range(1, count + 1)}
    with pytest.raises(bound_sql_module.MigrationCatalogError, match="1..32"):
        bound_sql_module.bind_bundled_sql_resources(tmp_path, hashes)


@pytest.mark.parametrize("count,size,error", [(1, 1024 * 1024 + 1, "1 MiB"), (5, 1024 * 1024, "4 MiB")])
def test_bound_sql_byte_bounds(tmp_path: Path, bound_sql_module, count: int, size: int, error: str) -> None:
    directory = tmp_path / "sql"
    hashes = _sql_files(directory, {f"{i:04d}_resource.sql": b"x" * size for i in range(1, count + 1)})
    with pytest.raises(bound_sql_module.MigrationCatalogError, match=error):
        bound_sql_module.bind_bundled_sql_resources(directory, hashes)
    assert bound_sql_module._BOUND_SQL_RESOURCES is None


def test_bound_sql_byte_boundaries_are_inclusive(tmp_path: Path, bound_sql_module) -> None:
    module = bound_sql_module
    directory = tmp_path / "sql"
    hashes = _sql_files(directory, {f"{i:04d}_resource.sql": b"x" * (1024 * 1024) for i in range(1, 5)})
    module.bind_bundled_sql_resources(directory, hashes)
    assert len(module.read_bundled_sql_text("0004_resource.sql")) == 1024 * 1024


def test_bound_sql_default_package_and_explicit_directory_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, bound_sql_module
) -> None:
    module = bound_sql_module
    package = tmp_path / "package"
    package.mkdir()
    hashes = _sql_files(package / "sql", {name: b"SELECT 1;" for name in module.REQUIRED_PACKAGE_SQL_FILENAMES})
    calls = []
    monkeypatch.setattr(module.resources, "files", lambda name: calls.append(name) or package)
    assert module.read_bundled_sql_text("0001_control_plane.sql") == "SELECT 1;"
    assert module.load_default_catalog().latest_version == 5
    assert calls
    module.bind_bundled_sql_resources(package / "sql", hashes)
    explicit = tmp_path / "explicit"
    _sql_files(explicit, {"0001_explicit.sql": b"SELECT 2;"})
    catalog = module.load_default_catalog(explicit)
    assert catalog.latest_version == 1
    assert catalog.get(1).sql_text == "SELECT 2;"
