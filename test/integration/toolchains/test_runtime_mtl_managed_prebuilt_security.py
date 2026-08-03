"""Security boundary for read-only managed Runtime MTL TypeScript parity."""

from __future__ import annotations

import importlib.util
import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CENTRAL_PATH = REPO_ROOT / "tools/logic/certify_formal_verification_toolchains.py"
RUNTIME_PATH = REPO_ROOT / "tools/logic/certification/runtime_mtl.py"
DEFAULT_SEALED_ROOT = Path(
    "/opt/ipfs-accelerate/formal-toolchains/fvt083-20260801-01/provers"
)
SEALED_ROOT = Path(
    os.environ.get(
        "IPFS_DATASETS_PY_EXTERNAL_PROVER_ROOT",
        str(DEFAULT_SEALED_ROOT),
    )
)
VERSION = "1.0.0-reviewed"
VENDOR_RELATIVE = (
    Path("runtime-mtl-vendor")
    / "runtime-mtl-external"
    / VERSION
)


def _load(name: str, path: Path):
    for candidate in (REPO_ROOT, REPO_ROOT / "ipfs_datasets_py"):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def central():
    return _load("runtime_mtl_prebuilt_security_central", CENTRAL_PATH)


@pytest.fixture(scope="module")
def runtime():
    return _load("runtime_mtl_prebuilt_security_focused", RUNTIME_PATH)


def _require_sealed_root() -> Path:
    required = (
        SEALED_ROOT / VENDOR_RELATIVE / "identity.json",
        SEALED_ROOT / VENDOR_RELATIVE / "package" / "dist" / "src" / "index.js",
        SEALED_ROOT / "bin" / "runtime-mtl",
    )
    if not all(path.is_file() for path in required):
        pytest.skip(f"sealed Runtime MTL deployment unavailable: {SEALED_ROOT}")
    return SEALED_ROOT


def _offline_env(central: Any, *, path_value: str) -> dict[str, str]:
    return central.offline_env(
        {
            central.RUNTIME_MTL_SEALED_ROOT_ENV: str(_require_sealed_root()),
            "PATH": path_value,
        }
    )


def _copy_sparse_vendor_tree(destination: Path) -> Path:
    source_root = _require_sealed_root()
    source_version = source_root / VENDOR_RELATIVE
    target_version = destination / VENDOR_RELATIVE
    target_package = target_version / "package"
    for relative in (
        Path("src"),
        Path("dist/src"),
        Path("bin"),
    ):
        (target_package / relative).mkdir(parents=True, exist_ok=True)
    (target_version / "bin").mkdir(parents=True, exist_ok=True)
    (destination / "bin").mkdir(parents=True, exist_ok=True)

    for relative in (
        Path("package.json"),
        Path("package-lock.json"),
        Path("src/index.ts"),
        Path("src/cli.ts"),
        Path("dist/src/index.js"),
        Path("dist/src/cli.js"),
    ):
        shutil.copy2(
            source_version / "package" / relative,
            target_package / relative,
        )
    shutil.copy2(source_version / "identity.json", target_version / "identity.json")
    shutil.copy2(
        source_version / "bin/runtime-mtl-external",
        target_version / "bin/runtime-mtl-external",
    )
    public_text = (source_root / "bin/runtime-mtl").read_text(encoding="utf-8")
    (destination / "bin/runtime-mtl").write_text(
        public_text.replace(str(source_root), str(destination)),
        encoding="utf-8",
    )

    for path in sorted(destination.rglob("*"), reverse=True):
        path.chmod(0o555 if path.is_dir() else 0o444)
    for path in (
        target_version / "bin/runtime-mtl-external",
        target_package / "dist/src/cli.js",
        destination / "bin/runtime-mtl",
    ):
        path.chmod(0o555)
    destination.chmod(0o555)
    return destination


def test_sealed_prebuilt_positive_without_checkout_mutation(
    central: Any,
    runtime: Any,
    tmp_path: Path,
) -> None:
    package_dist = REPO_ROOT / runtime.TS_PACKAGE_RELATIVE / "dist"
    before = (package_dist.exists(), package_dist.is_symlink())
    malicious_bin = tmp_path / "malicious-bin"
    malicious_bin.mkdir()
    (malicious_bin / "runtime-mtl").write_text("#!/bin/sh\nexit 99\n")
    (malicious_bin / "node").write_text("#!/bin/sh\nexit 99\n")

    binding = central._runtime_mtl_managed_prebuilt_binding(
        REPO_ROOT,
        env=_offline_env(central, path_value=str(malicious_bin)),
    )
    assert binding["public"]["authenticated"] is True
    assert binding["public"]["ambient_path_used"] is False
    assert binding["public"]["checkout_mutated"] is False
    assert binding["invocation"]["sealed_root"] == str(_require_sealed_root())

    receipt = runtime.certify_runtime_mtl_semantics(
        repo_root=REPO_ROOT,
        typescript_prebuilt_root=_require_sealed_root(),
    )
    assert receipt["certified"] is True
    assert receipt["parity"]["authenticated_external_prebuilt"] is True
    assert receipt["parity"]["ambient_path_used"] is False
    assert before == (package_dist.exists(), package_dist.is_symlink())


def test_checked_receipt_tamper_fails_closed(
    central: Any,
    tmp_path: Path,
) -> None:
    source = REPO_ROOT / central.RUNTIME_MTL_VENDOR_RECEIPT_RELATIVE
    receipt = json.loads(source.read_text(encoding="utf-8"))
    receipt["runtime_mtl_external"]["package_digest_sha256"] = "0" * 64
    receipt["receipt_digest_sha256"] = central.content_digest(
        {
            key: value
            for key, value in receipt.items()
            if key != "receipt_digest_sha256"
        }
    )
    tampered = tmp_path / "tampered-receipt.json"
    tampered.write_text(json.dumps(receipt), encoding="utf-8")

    binding = central._runtime_mtl_managed_prebuilt_binding(
        REPO_ROOT,
        env=_offline_env(central, path_value="/definitely/not/used"),
        receipt_path=tampered,
    )
    assert binding["public"]["authenticated"] is False
    assert {
        "vendor_identity_package_digest_sha256_mismatch",
        "sealed_artifact_package_digest_sha256_mismatch",
        "repository_package_digest_sha256_mismatch",
    } <= set(binding["public"]["failures"])


def test_dist_tamper_and_non_root_copy_fail_closed(
    central: Any,
    runtime: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    copied_root = _copy_sparse_vendor_tree(tmp_path / "copied-provers")
    index = copied_root / VENDOR_RELATIVE / "package/dist/src/index.js"
    index.chmod(0o644)
    index.write_text(index.read_text(encoding="utf-8") + "\n// tampered\n")
    index.chmod(0o444)

    monkeypatch.setattr(
        central,
        "APPROVED_IMMUTABLE_DEPLOYMENT_ROOTS",
        (tmp_path,),
    )
    monkeypatch.setattr(
        runtime,
        "TYPESCRIPT_PREBUILT_APPROVED_ROOTS",
        (tmp_path,),
    )
    env = central.offline_env(
        {
            central.RUNTIME_MTL_SEALED_ROOT_ENV: str(copied_root),
            "PATH": "/ambient/path/must/not/matter",
        }
    )
    central_result = central._runtime_mtl_managed_prebuilt_binding(
        REPO_ROOT,
        env=env,
    )
    assert central_result["public"]["authenticated"] is False
    assert "sealed_root_not_root_owned" in central_result["public"]["failures"]
    assert "sealed_artifact_artifact_sha256_mismatch" in (
        central_result["public"]["failures"]
    )

    focused = runtime._authenticate_typescript_prebuilt(
        REPO_ROOT,
        sealed_root=copied_root,
        timeout_seconds=10,
    )
    assert focused["valid"] is False
    assert "sealed_root_not_root_owned_directory" in focused["failures"]
    assert "artifact_artifact_sha256_mismatch" in focused["failures"]


def test_writable_root_is_rejected(
    central: Any,
    runtime: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    writable_root = tmp_path / "writable-provers"
    writable_root.mkdir(mode=0o755)
    monkeypatch.setattr(
        central,
        "APPROVED_IMMUTABLE_DEPLOYMENT_ROOTS",
        (tmp_path,),
    )
    monkeypatch.setattr(
        runtime,
        "TYPESCRIPT_PREBUILT_APPROVED_ROOTS",
        (tmp_path,),
    )
    env = central.offline_env(
        {
            central.RUNTIME_MTL_SEALED_ROOT_ENV: str(writable_root),
            "PATH": "/unused",
        }
    )
    central_result = central._runtime_mtl_managed_prebuilt_binding(
        REPO_ROOT,
        env=env,
    )
    assert "sealed_root_writable" in central_result["public"]["failures"]
    focused = runtime._authenticate_typescript_prebuilt(
        REPO_ROOT,
        sealed_root=writable_root,
        timeout_seconds=10,
    )
    assert "sealed_root_writable" in focused["failures"]


def test_managed_parity_never_resolves_node_from_ambient_path(
    runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _require_sealed_root()

    def forbidden_which(*_args: Any, **_kwargs: Any) -> str:
        raise AssertionError("managed Runtime MTL parity queried ambient PATH")

    monkeypatch.setattr(runtime.shutil, "which", forbidden_which)
    check, detail = runtime.run_python_typescript_parity(
        repo_root=REPO_ROOT,
        typescript_prebuilt_root=root,
    )
    assert check.status == "passed"
    assert detail["ambient_path_used"] is False


def test_timeout_is_a_deterministic_failed_check(
    runtime: Any,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = _require_sealed_root()
    real_run = subprocess.run

    def timeout_after_identity(argv: list[str], **kwargs: Any):
        if argv[1:] == ["--version"]:
            return real_run(argv, **kwargs)
        raise subprocess.TimeoutExpired(argv, kwargs.get("timeout", 0))

    monkeypatch.setattr(runtime.subprocess, "run", timeout_after_identity)
    check, detail = runtime.run_python_typescript_parity(
        repo_root=REPO_ROOT,
        typescript_prebuilt_root=root,
    )
    assert check.status == "failed"
    assert check.observed == "typescript_parity_timeout"
    assert detail["execution_error"] == "typescript_parity_timeout"


def test_no_boolean_or_arbitrary_path_prebuilt_api(runtime: Any) -> None:
    parameters = inspect.signature(
        runtime.certify_runtime_mtl_semantics
    ).parameters
    assert "typescript_prebuilt" not in parameters
    assert "typescript_prebuilt_root" in parameters
