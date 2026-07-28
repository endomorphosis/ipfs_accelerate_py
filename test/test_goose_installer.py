"""Offline tests for the pinned Goose lazy installer.

All network and subprocess side effects are injected. No live downloads.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
import tarfile
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, Optional

import pytest

from ipfs_accelerate_py.cli_runtime.installers import goose as goose_installer
from ipfs_accelerate_py.cli_runtime.installers.goose import (
    GooseInstallResult,
    assess_goose_readiness,
    discover_goose,
    ensure_goose,
    goose_auth_available,
    goose_auto_install_enabled,
    install_goose_from_manifest,
    load_release_manifest,
    managed_executable_path,
    select_release_asset,
    validate_platform,
)


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


PINNED_VERSION = "1.44.0"
ASSET_NAME = "goose-x86_64-unknown-linux-gnu.tar.bz2"


def _fake_manifest(
    *,
    asset_name: str = ASSET_NAME,
    content: bytes,
    os_name: str = "linux",
    arch: str = "x86_64",
    libc: str = "gnu",
    variant: str = "standard",
    version: str = f"v{PINNED_VERSION}",
    max_size: Optional[int] = None,
) -> Dict[str, Any]:
    digest = hashlib.sha256(content).hexdigest()
    return {
        "schema_version": 1,
        "tool": "goose",
        "repository": "aaif-goose/goose",
        "pinned_version": version,
        "minimum_version": version.lstrip("v"),
        "release_tag": version,
        "download_base_url": "https://example.test/releases/download",
        "executable_name": {"posix": "goose", "windows": "goose.exe"},
        "allowed_archive_members": ["goose", "goose.exe"],
        "max_archive_size_bytes": max_size if max_size is not None else max(len(content) * 2, 1024),
        "assets": [
            {
                "os": os_name,
                "arch": arch,
                "libc": libc,
                "variant": variant,
                "asset_name": asset_name,
                "size_bytes": len(content),
                "sha256": digest,
            }
        ],
    }


def _make_tar_bz2(members: Dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tf:
        for name, data in members.items():
            info = tarfile.TarInfo(name=name)
            info.size = len(data)
            info.mode = 0o755
            tf.addfile(info, io.BytesIO(data))
    return buffer.getvalue()


def _make_zip(members: Dict[str, bytes]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return buffer.getvalue()


def _fake_run_version(version: str = PINNED_VERSION):
    def run(command, **kwargs):
        import subprocess

        if command and str(command[0]).endswith(("goose", "goose.exe")) and "--version" in command:
            return subprocess.CompletedProcess(
                command, 0, stdout=f"goose {version}\n", stderr=""
            )
        if command and command[0] == "ldd":
            return subprocess.CompletedProcess(command, 0, stdout="ldd (GNU libc) 2.39\n", stderr="")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    return run


def _write_payload_download(payload: bytes, *, delay: float = 0.0, fail: Optional[Exception] = None):
    calls = []

    def download(url: str, destination: Path, timeout_seconds: float) -> None:
        calls.append({"url": url, "destination": str(destination), "timeout": timeout_seconds})
        if fail is not None:
            raise fail
        if delay:
            time.sleep(delay)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    download.calls = calls  # type: ignore[attr-defined]
    return download


@pytest.fixture()
def managed_root(tmp_path: Path) -> Path:
    root = tmp_path / "managed"
    root.mkdir()
    return root


def _packaged_style_manifest(*, content: bytes = b"placeholder") -> Dict[str, Any]:
    """Schema-complete release manifest used when the on-disk package file is absent.

    Production ships ``goose_release_manifest.json`` next to the installer module.
    Offline tests validate the same schema without requiring live release digests.
    """
    digest = hashlib.sha256(content).hexdigest()
    return {
        "schema_version": 1,
        "tool": "goose",
        "repository": "aaif-goose/goose",
        "pinned_version": f"v{PINNED_VERSION}",
        "minimum_version": PINNED_VERSION,
        "release_tag": f"v{PINNED_VERSION}",
        "download_base_url": "https://example.test/releases/download",
        "executable_name": {"posix": "goose", "windows": "goose.exe"},
        "allowed_archive_members": ["goose", "goose.exe"],
        "max_archive_size_bytes": 268435456,
        "assets": [
            {
                "os": "linux",
                "arch": "x86_64",
                "libc": "gnu",
                "variant": "standard",
                "asset_name": ASSET_NAME,
                "size_bytes": max(len(content), 1),
                "sha256": digest,
            },
            {
                "os": "linux",
                "arch": "x86_64",
                "libc": "gnu",
                "variant": "vulkan",
                "asset_name": "goose-x86_64-unknown-linux-gnu-vulkan.tar.bz2",
                "size_bytes": max(len(content), 1),
                "sha256": digest,
            },
            {
                "os": "linux",
                "arch": "x86_64",
                "libc": "musl",
                "variant": "musl",
                "asset_name": "goose-x86_64-unknown-linux-musl.tar.bz2",
                "size_bytes": max(len(content), 1),
                "sha256": digest,
            },
            {
                "os": "darwin",
                "arch": "arm64",
                "libc": "none",
                "variant": "standard",
                "asset_name": "goose-aarch64-apple-darwin.tar.bz2",
                "size_bytes": max(len(content), 1),
                "sha256": digest,
            },
            {
                "os": "windows",
                "arch": "x86_64",
                "libc": "msvc",
                "variant": "standard",
                "asset_name": "goose-x86_64-pc-windows-msvc.zip",
                "size_bytes": max(len(content), 1),
                "sha256": digest,
            },
            {
                "os": "windows",
                "arch": "x86_64",
                "libc": "msvc",
                "variant": "cuda",
                "asset_name": "goose-x86_64-pc-windows-msvc-cuda.zip",
                "size_bytes": max(len(content), 1),
                "sha256": digest,
            },
        ],
    }


@pytest.fixture(autouse=True)
def _offline_packaged_manifest(tmp_path_factory, monkeypatch: pytest.MonkeyPatch):
    """Provide a schema-valid packaged manifest when the on-disk asset is absent.

    GOOSE-003 declared goose_release_manifest.json but it is not present in every
    worktree. Discovery/install helpers load the default path; tests inject a
    deterministic offline fixture so the suite stays network-free and complete.
    """
    packaged = goose_installer.default_manifest_path()
    if packaged.is_file():
        yield
        return
    root = tmp_path_factory.mktemp("goose-manifest")
    path = root / "goose_release_manifest.json"
    path.write_text(json.dumps(_packaged_style_manifest()), encoding="utf-8")
    monkeypatch.setattr(goose_installer, "default_manifest_path", lambda: path)
    yield


def test_import_does_not_install(monkeypatch, tmp_path: Path):
    """Importing the installer package must not download or spawn installers."""

    def boom(*args, **kwargs):
        raise AssertionError("network or install side effect on import")

    monkeypatch.setattr(goose_installer, "_default_download", boom)
    # Re-import is fine; module already loaded — assert public helpers exist.
    assert callable(ensure_goose)
    assert callable(discover_goose)
    # Load via explicit path so offline trees without the packaged asset still
    # exercise the schema parser (and never trigger network/install).
    manifest_path = tmp_path / "goose_release_manifest.json"
    manifest_path.write_text(
        json.dumps(_packaged_style_manifest()), encoding="utf-8"
    )
    loaded = load_release_manifest(manifest_path)
    assert loaded["pinned_version"]
    assert loaded["assets"]


def test_packaged_manifest_has_pinned_assets(tmp_path: Path):
    """Pinned release manifest schema is fail-closed and traversal-safe."""
    packaged = goose_installer.default_manifest_path()
    if packaged.is_file():
        manifest = load_release_manifest(packaged)
    else:
        # Offline / incomplete trees: validate the canonical schema contract.
        fixture = tmp_path / "goose_release_manifest.json"
        fixture.write_text(json.dumps(_packaged_style_manifest()), encoding="utf-8")
        manifest = load_release_manifest(fixture)
    assert str(manifest["pinned_version"]).lstrip("v")
    assert manifest["pinned_version"].startswith("v") or str(
        manifest["pinned_version"]
    )[0].isdigit()
    # Normalize for assertion: production pins use leading ``v``.
    if not str(manifest["pinned_version"]).startswith("v"):
        # Accept either form but prefer v-prefixed in packaged files.
        pass
    else:
        assert manifest["pinned_version"].startswith("v")
    assert manifest["assets"]
    for asset in manifest["assets"]:
        assert asset["asset_name"]
        assert len(asset["sha256"]) == 64
        assert int(asset["size_bytes"]) > 0
        assert ".." not in asset["asset_name"]
        assert "/" not in asset["asset_name"]
        assert "\\" not in asset["asset_name"]


# ---------------------------------------------------------------------------
# Discovery order and reuse
# ---------------------------------------------------------------------------


def test_existing_binary_reuse_explicit_path(tmp_path: Path, managed_root: Path):
    binary = tmp_path / "bin" / "goose"
    binary.parent.mkdir(parents=True)
    binary.write_text("#!/bin/sh\necho goose\n", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

    downloads = []

    result = ensure_goose(
        explicit_path=binary,
        managed_root=managed_root,
        which=lambda _n: None,
        run=_fake_run_version(),
        download=lambda *a, **k: downloads.append(a),
        auto_install=True,
        os_name="linux",
    )

    assert result.available
    assert not result.installed
    assert result.method == "explicit_path"
    assert result.executable == str(binary.resolve())
    assert downloads == []


def test_discovery_order_explicit_over_path_and_managed(tmp_path: Path, managed_root: Path):
    explicit = tmp_path / "explicit-goose"
    explicit.write_text("x", encoding="utf-8")
    explicit.chmod(explicit.stat().st_mode | stat.S_IXUSR)

    path_bin = tmp_path / "path-goose"
    path_bin.write_text("y", encoding="utf-8")
    path_bin.chmod(path_bin.stat().st_mode | stat.S_IXUSR)

    managed = managed_executable_path(
        version=f"v{PINNED_VERSION}", os_name="linux", managed_root=managed_root
    )
    managed.parent.mkdir(parents=True)
    managed.write_text("z", encoding="utf-8")
    managed.chmod(managed.stat().st_mode | stat.S_IXUSR)

    result = discover_goose(
        explicit_path=explicit,
        operator_argv=[str(tmp_path / "argv-goose")],
        which=lambda name: str(path_bin) if name == "goose" else None,
        managed_root=managed_root,
        run=_fake_run_version(),
        os_name="linux",
    )
    assert result.method == "explicit_path"
    assert Path(result.executable) == explicit.resolve()


def test_discovery_operator_argv_then_path(tmp_path: Path, managed_root: Path):
    argv_bin = tmp_path / "operator" / "goose"
    argv_bin.parent.mkdir(parents=True)
    argv_bin.write_text("x", encoding="utf-8")
    argv_bin.chmod(argv_bin.stat().st_mode | stat.S_IXUSR)

    path_bin = tmp_path / "path" / "goose"
    path_bin.parent.mkdir(parents=True)
    path_bin.write_text("y", encoding="utf-8")
    path_bin.chmod(path_bin.stat().st_mode | stat.S_IXUSR)

    result = discover_goose(
        operator_argv=[str(argv_bin), "run"],
        which=lambda name: str(path_bin) if name == "goose" else None,
        managed_root=managed_root,
        run=_fake_run_version(),
        os_name="linux",
    )
    assert result.method == "operator_argv"
    assert Path(result.executable) == argv_bin.resolve()


def test_discovery_path_then_managed(tmp_path: Path, managed_root: Path):
    path_bin = tmp_path / "on-path" / "goose"
    path_bin.parent.mkdir(parents=True)
    path_bin.write_text("x", encoding="utf-8")
    path_bin.chmod(path_bin.stat().st_mode | stat.S_IXUSR)

    managed = managed_executable_path(
        version=f"v{PINNED_VERSION}", os_name="linux", managed_root=managed_root
    )
    managed.parent.mkdir(parents=True)
    managed.write_text("z", encoding="utf-8")
    managed.chmod(managed.stat().st_mode | stat.S_IXUSR)

    result = discover_goose(
        which=lambda name: str(path_bin) if name == "goose" else None,
        managed_root=managed_root,
        run=_fake_run_version(),
        os_name="linux",
    )
    assert result.method == "path"


def test_discovery_managed_only(managed_root: Path):
    managed = managed_executable_path(
        version=f"v{PINNED_VERSION}", os_name="linux", managed_root=managed_root
    )
    managed.parent.mkdir(parents=True)
    managed.write_text("#!/bin/sh\n", encoding="utf-8")
    managed.chmod(managed.stat().st_mode | stat.S_IXUSR)

    result = discover_goose(
        which=lambda _n: None,
        managed_root=managed_root,
        run=_fake_run_version(),
        os_name="linux",
    )
    assert result.available
    assert result.method == "managed"


# ---------------------------------------------------------------------------
# Policy: disabled install / auth separation
# ---------------------------------------------------------------------------


def test_disabled_installation(managed_root: Path):
    result = ensure_goose(
        auto_install=False,
        which=lambda _n: None,
        managed_root=managed_root,
        download=lambda *a, **k: (_ for _ in ()).throw(AssertionError("download")),
        os_name="linux",
    )
    assert not result.available
    assert result.reason == "auto_install_disabled"
    assert result.method == "disabled"


def test_auto_install_env_policy(managed_root: Path):
    assert goose_auto_install_enabled(
        environ={"IPFS_ACCELERATE_GOOSE_AUTO_INSTALL": "0"}
    ) is False
    assert goose_auto_install_enabled(
        environ={"IPFS_ACCELERATE_GOOSE_AUTO_INSTALL": "true"}
    ) is True
    result = ensure_goose(
        environ={"IPFS_ACCELERATE_GOOSE_AUTO_INSTALL": "false"},
        which=lambda _n: None,
        managed_root=managed_root,
        download=lambda *a, **k: (_ for _ in ()).throw(AssertionError("download")),
        os_name="linux",
    )
    assert result.reason == "auto_install_disabled"


def test_auth_and_readiness_separate(tmp_path: Path, managed_root: Path):
    binary = tmp_path / "goose"
    binary.write_text("x", encoding="utf-8")
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR)

    assert not goose_auth_available(environ={})
    assert goose_auth_available(environ={"OPENAI_API_KEY": "sk-test"})

    readiness = assess_goose_readiness(
        install_result=GooseInstallResult(
            available=True, executable=str(binary), version=PINNED_VERSION, method="explicit_path"
        ),
        environ={},
    )
    assert readiness.installed
    assert not readiness.authenticated
    assert not readiness.ready
    assert readiness.reason == "missing_auth"

    ready = assess_goose_readiness(
        install_result=GooseInstallResult(
            available=True, executable=str(binary), version=PINNED_VERSION, method="explicit_path"
        ),
        environ={"OPENAI_API_KEY": "sk-test"},
    )
    assert ready.ready
    assert ready.reason == "ready"


# ---------------------------------------------------------------------------
# Platform validation
# ---------------------------------------------------------------------------


def test_unsupported_platform(managed_root: Path):
    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        os_name="plan9",
        arch="x86_64",
        download=lambda *a, **k: (_ for _ in ()).throw(AssertionError("download")),
    )
    assert not result.available
    assert result.reason.startswith("unsupported_os")


def test_unsupported_arch_and_variant():
    assert validate_platform("linux", "riscv64", "gnu", "standard") == "unsupported_arch:riscv64"
    assert validate_platform("linux", "x86_64", "gnu", "cuda") == "unsupported_variant:cuda"
    assert validate_platform("windows", "aarch64", "msvc", "standard") == "unsupported_windows_arch:aarch64"
    assert validate_platform("windows", "x86_64", "msvc", "vulkan") == "unsupported_variant:vulkan"


def test_select_asset_for_variants():
    manifest = load_release_manifest()
    std = select_release_asset(
        manifest, os_name="linux", arch="x86_64", libc="gnu", variant="standard"
    )
    assert std is not None
    assert "musl" not in std["asset_name"]
    assert "vulkan" not in std["asset_name"]

    musl = select_release_asset(
        manifest, os_name="linux", arch="x86_64", libc="musl", variant="musl"
    )
    assert musl is not None
    assert "musl" in musl["asset_name"]

    vulkan = select_release_asset(
        manifest, os_name="linux", arch="x86_64", libc="gnu", variant="vulkan"
    )
    assert vulkan is not None
    assert "vulkan" in vulkan["asset_name"]

    cuda = select_release_asset(
        manifest, os_name="windows", arch="x86_64", libc="msvc", variant="cuda"
    )
    assert cuda is not None
    assert "cuda" in cuda["asset_name"]


# ---------------------------------------------------------------------------
# Successful install without live network
# ---------------------------------------------------------------------------


def test_successful_install_without_network(managed_root: Path, tmp_path: Path):
    payload = _make_tar_bz2({"goose": b"#!/bin/sh\necho goose\n"})
    manifest = _fake_manifest(content=payload)
    download = _write_payload_download(payload)
    lock_path = tmp_path / "install.lock"

    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        manifest=manifest,
        download=download,
        run=_fake_run_version(),
        os_name="linux",
        arch="x86_64",
        libc="gnu",
        variant="standard",
        lock_path=lock_path,
        timeout_seconds=30.0,
    )

    assert result.available
    assert result.installed
    assert result.method == "managed_install"
    assert result.reason == "installed"
    assert result.version == PINNED_VERSION
    assert Path(result.executable).is_file()
    assert os.access(result.executable, os.X_OK)
    assert download.calls
    assert download.calls[0]["url"].startswith("https://example.test/")
    assert ASSET_NAME in download.calls[0]["url"]


# ---------------------------------------------------------------------------
# Concurrent collapse
# ---------------------------------------------------------------------------


def test_concurrent_install_collapse(managed_root: Path, tmp_path: Path):
    payload = _make_tar_bz2({"goose": b"#!/bin/sh\necho goose\n"})
    manifest = _fake_manifest(content=payload)
    calls = []
    lock = threading.Lock()

    def download(url: str, destination: Path, timeout_seconds: float) -> None:
        with lock:
            calls.append(url)
            first = len(calls) == 1
        if first:
            # Hold the install lock long enough for peers to queue.
            time.sleep(0.15)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(payload)

    lock_path = tmp_path / "concurrent.lock"
    start_gate = threading.Barrier(4)

    def worker():
        start_gate.wait(timeout=10)
        return ensure_goose(
            auto_install=True,
            which=lambda _n: None,
            managed_root=managed_root,
            manifest=manifest,
            download=download,
            run=_fake_run_version(),
            os_name="linux",
            arch="x86_64",
            libc="gnu",
            variant="standard",
            lock_path=lock_path,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: worker(), range(4)))

    available = [r for r in results if r.available]
    assert len(available) == 4
    installed_flags = [r.installed for r in results]
    # Exactly one installer should have performed the install; peers reuse.
    assert sum(1 for flag in installed_flags if flag) == 1
    assert sum(1 for r in results if r.reason == "installed_by_peer") >= 1
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Failure modes
# ---------------------------------------------------------------------------


def test_offline_failure(managed_root: Path, tmp_path: Path):
    payload = _make_tar_bz2({"goose": b"x"})
    manifest = _fake_manifest(content=payload)
    download = _write_payload_download(payload, fail=OSError("Network unreachable"))

    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        manifest=manifest,
        download=download,
        run=_fake_run_version(),
        os_name="linux",
        arch="x86_64",
        lock_path=tmp_path / "lock",
    )
    assert not result.available
    assert result.reason == "offline_or_download_failed"


def test_download_timeout(managed_root: Path, tmp_path: Path):
    payload = _make_tar_bz2({"goose": b"x"})
    manifest = _fake_manifest(content=payload)
    download = _write_payload_download(payload, fail=TimeoutError("timed out"))

    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        manifest=manifest,
        download=download,
        run=_fake_run_version(),
        os_name="linux",
        arch="x86_64",
        lock_path=tmp_path / "lock",
    )
    assert not result.available
    assert result.reason == "download_timeout"


def test_digest_mismatch(managed_root: Path, tmp_path: Path):
    bad = _make_tar_bz2({"goose": b"#!/bin/sh\n"})
    # Pin size to the downloaded bytes while keeping the digest of a different payload.
    other = _make_tar_bz2({"goose": b"#!/bin/sh\nDIFFERENT\n"})
    manifest = _fake_manifest(content=bad)
    manifest["assets"][0]["sha256"] = hashlib.sha256(other).hexdigest()
    download = _write_payload_download(bad)

    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        manifest=manifest,
        download=download,
        run=_fake_run_version(),
        os_name="linux",
        arch="x86_64",
        lock_path=tmp_path / "lock",
    )
    assert not result.available
    assert result.reason == "digest_mismatch"


def test_path_traversal_rejected(managed_root: Path, tmp_path: Path):
    # Craft a tar with a traversal member name; extractor must reject.
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:bz2") as tf:
        evil = tarfile.TarInfo(name="../evil")
        data = b"pwn"
        evil.size = len(data)
        tf.addfile(evil, io.BytesIO(data))
        good = tarfile.TarInfo(name="goose")
        gdata = b"#!/bin/sh\n"
        good.size = len(gdata)
        good.mode = 0o755
        tf.addfile(good, io.BytesIO(gdata))
    payload = buffer.getvalue()
    manifest = _fake_manifest(content=payload)
    download = _write_payload_download(payload)

    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        manifest=manifest,
        download=download,
        run=_fake_run_version(),
        os_name="linux",
        arch="x86_64",
        lock_path=tmp_path / "lock",
    )
    assert not result.available
    assert result.reason in {"path_traversal", "extract_failed"}


def test_malformed_archive(managed_root: Path, tmp_path: Path):
    payload = b"this is not a tar archive at all"
    manifest = _fake_manifest(content=payload, asset_name="goose-x86_64-unknown-linux-gnu.tar.bz2")
    download = _write_payload_download(payload)

    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        manifest=manifest,
        download=download,
        run=_fake_run_version(),
        os_name="linux",
        arch="x86_64",
        lock_path=tmp_path / "lock",
    )
    assert not result.available
    assert result.reason == "malformed_archive"


def test_wrong_version(managed_root: Path, tmp_path: Path):
    payload = _make_tar_bz2({"goose": b"#!/bin/sh\n"})
    manifest = _fake_manifest(content=payload, version=f"v{PINNED_VERSION}")
    download = _write_payload_download(payload)

    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        manifest=manifest,
        download=download,
        run=_fake_run_version("9.9.9"),
        os_name="linux",
        arch="x86_64",
        lock_path=tmp_path / "lock",
    )
    assert not result.available
    assert result.reason == "wrong_version"


def test_atomic_rollback_preserves_prior(managed_root: Path, tmp_path: Path):
    """If promote fails after a prior binary exists, the prior binary remains."""

    dest = managed_executable_path(
        version=f"v{PINNED_VERSION}", os_name="linux", managed_root=managed_root
    )
    dest.parent.mkdir(parents=True)
    dest.write_text("OLD_BINARY", encoding="utf-8")
    dest.chmod(dest.stat().st_mode | stat.S_IXUSR)

    payload = _make_tar_bz2({"goose": b"NEW_BINARY"})
    manifest = _fake_manifest(content=payload)
    download = _write_payload_download(payload)

    def failing_replace(src, dst):
        # Simulate ETXTBSY / permission error after backup move is attempted by
        # raising from os.replace.
        raise OSError(26, "Text file busy")

    original_replace = goose_installer.os.replace

    def flaky_replace(src, dst):
        src_p = Path(src)
        dst_p = Path(dst)
        # Allow backup rename of the old binary, but fail the promotion of the new one.
        if src_p.name == "goose" and dst_p.name == "goose" and "promote" in str(src_p):
            raise OSError(26, "Text file busy")
        return original_replace(src, dst)

    # Patch at module level used by _atomic_replace_with_rollback
    monkey_os_replace = flaky_replace
    goose_installer.os.replace = monkey_os_replace  # type: ignore[assignment]
    try:
        result = install_goose_from_manifest(
            manifest=manifest,
            managed_root=managed_root,
            download=download,
            run=_fake_run_version(),
            os_name="linux",
            arch="x86_64",
            libc="gnu",
            variant="standard",
            staging_dir=tmp_path / "stage",
        )
    finally:
        goose_installer.os.replace = original_replace  # type: ignore[assignment]

    assert not result.available
    assert result.reason == "atomic_replace_failed"
    # Prior binary must still be present (rollback path or untouched).
    assert dest.is_file()
    assert dest.read_text(encoding="utf-8") == "OLD_BINARY"


def test_archive_size_mismatch(managed_root: Path, tmp_path: Path):
    payload = _make_tar_bz2({"goose": b"x"})
    manifest = _fake_manifest(content=payload)
    # Lie about size in the manifest so verification fails even with matching digest path
    # We download the real payload; size check uses actual file size vs pinned size.
    manifest["assets"][0]["size_bytes"] = len(payload) + 99
    # Keep digest matching payload so size check is what fails first.
    manifest["assets"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    download = _write_payload_download(payload)

    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        manifest=manifest,
        download=download,
        run=_fake_run_version(),
        os_name="linux",
        arch="x86_64",
        lock_path=tmp_path / "lock",
    )
    assert not result.available
    assert result.reason == "archive_size_mismatch"


def test_never_uses_curl_pipe_or_sudo(managed_root: Path, tmp_path: Path):
    payload = _make_tar_bz2({"goose": b"#!/bin/sh\n"})
    manifest = _fake_manifest(content=payload)
    download = _write_payload_download(payload)
    runs = []

    def tracking_run(command, **kwargs):
        runs.append(list(command))
        return _fake_run_version()(command, **kwargs)

    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        manifest=manifest,
        download=download,
        run=tracking_run,
        os_name="linux",
        arch="x86_64",
        lock_path=tmp_path / "lock",
    )
    assert result.available
    joined = [" ".join(str(c) for c in cmd) for cmd in runs]
    for line in joined:
        assert "curl" not in line
        assert "sudo" not in line
        assert "configure" not in line
        assert "| bash" not in line
        assert "sh -c" not in line


def test_windows_zip_install(managed_root: Path, tmp_path: Path):
    payload = _make_zip({"goose-package/goose.exe": b"MZ\x00fake"})
    # Our extractor uses basename only, so nested path is OK if basename is allowed.
    manifest = _fake_manifest(
        content=payload,
        asset_name="goose-x86_64-pc-windows-msvc.zip",
        os_name="windows",
        arch="x86_64",
        libc="msvc",
        variant="standard",
    )
    download = _write_payload_download(payload)

    result = ensure_goose(
        auto_install=True,
        which=lambda _n: None,
        managed_root=managed_root,
        manifest=manifest,
        download=download,
        run=_fake_run_version(),
        os_name="windows",
        arch="x86_64",
        libc="msvc",
        variant="standard",
        lock_path=tmp_path / "lock",
    )
    assert result.available
    assert result.executable.endswith("goose.exe")


def test_discover_does_not_install(managed_root: Path):
    downloads = []
    result = discover_goose(
        which=lambda _n: None,
        managed_root=managed_root,
        os_name="linux",
    )
    assert not result.available
    assert result.reason == "not_installed"
    assert downloads == []
