from __future__ import annotations

import asyncio
import tarfile
from pathlib import Path

import pytest
from benchmarks.agent_supervisor.container_coding.terminal_deployment import (
    build_runtime_archive,
    deploy_supervisor,
    runtime_environment,
)


def _inputs(tmp_path):
    roots = {}
    for key, package in [
        ("source", "ipfs_accelerate_py"),
        ("datasets", "ipfs_datasets_py"),
        ("kit", "ipfs_kit_py"),
    ]:
        root = tmp_path / key
        (root / package).mkdir(parents=True)
        (root / package / "__init__.py").write_text("VALUE = 1\n")
        (root / package / "auth.json").write_text('{"sensitive":"must not enter archive"}')
        (root / package / ".env").write_text("TOKEN=must-not-enter-archive")
        roots[key] = root
    bench = roots["source"] / "benchmarks/agent_supervisor/container_coding"
    bench.mkdir(parents=True)
    (bench / "worker.py").write_text("VALUE = 2\n")
    extensions = tmp_path / "extensions"
    extensions.mkdir()
    for name in ("ducklake", "httpfs", "quack"):
        (extensions / (name + ".duckdb_extension")).write_bytes(b"packaging-test-only")
    return {**roots, "extension_dir": extensions}


def test_archive_contains_only_explicit_runtime_sources_and_native_assets(tmp_path):
    result = build_runtime_archive(output=tmp_path / "bundle", **_inputs(tmp_path))
    with tarfile.open(tmp_path / "bundle/runtime.tar.gz") as archive:
        names = archive.getnames()
        assert len(names) == 7
        assert "source/ipfs_accelerate_py/__init__.py" in names
        assert all(not name.startswith("/") and ".." not in Path(name).parts for name in names)
        assert not any("auth.json" in name or ".env" in name for name in names)
        assert all(x.uid == 0 and x.gid == 0 and x.isfile() for x in archive.getmembers())
    assert result["credentials_in_archive"] is False
    assert result["task_inputs_in_archive"] is False
    assert "OPENAI_API_KEY" not in runtime_environment()


def test_extension_symlink_cannot_enter_runtime_archive(tmp_path):
    inputs = _inputs(tmp_path)
    target = inputs["extension_dir"] / "quack.duckdb_extension"
    target.unlink()
    target.symlink_to(inputs["extension_dir"] / "httpfs.duckdb_extension")
    with pytest.raises(ValueError, match="non-symlink"):
        build_runtime_archive(output=tmp_path / "bundle", **inputs)


def test_changed_runtime_archive_refused_before_container_calls(tmp_path):
    build_runtime_archive(output=tmp_path / "bundle", **_inputs(tmp_path))
    (tmp_path / "bundle/runtime.tar.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="archive changed"):
        asyncio.run(
            deploy_supervisor(None, archive_dir=tmp_path / "bundle", output=tmp_path / "deployment")
        )
    assert not (tmp_path / "deployment/deployment.json").exists()


def test_model_revision_survives_asset_packaging(tmp_path):
    revision = "1" * 40
    snapshot = tmp_path / revision
    snapshot.mkdir()
    (snapshot / "config.json").write_text('{"packaging_fixture": true}')
    result = build_runtime_archive(
        output=tmp_path / "bundle", model_snapshot=snapshot, **_inputs(tmp_path)
    )
    assert result["model_snapshot_revision"] == revision
    assert result["runtime_python_version"] == "3.12.12"
    with tarfile.open(tmp_path / "bundle/runtime.tar.gz") as archive:
        assert "models/embedding/config.json" in archive.getnames()


def test_model_revision_must_be_an_explicit_snapshot_directory(tmp_path):
    snapshot = tmp_path / "unversioned-model"
    snapshot.mkdir()
    with pytest.raises(ValueError, match="snapshot revision directory"):
        build_runtime_archive(
            output=tmp_path / "bundle", model_snapshot=snapshot, **_inputs(tmp_path)
        )
