"""Fail-closed install boundary for Runtime MTL semantic certification."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
CERTIFIER_PATH = REPO_ROOT / "tools" / "logic" / "certification" / "runtime_mtl.py"


@pytest.fixture(scope="module")
def certifier():
    for candidate in (REPO_ROOT, REPO_ROOT / "ipfs_datasets_py"):
        text = str(candidate)
        if text not in sys.path:
            sys.path.insert(0, text)
    spec = importlib.util.spec_from_file_location(
        "runtime_mtl_offline_boundary_certifier",
        CERTIFIER_PATH,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_missing_prebuilt_artifact_never_invokes_package_manager(
    certifier,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    package = tmp_path / certifier.TS_PACKAGE_RELATIVE
    (package / "src").mkdir(parents=True)
    (package / "package.json").write_text('{"name":"offline-test"}\n')
    (package / "src" / "index.ts").write_text("export const value = 1;\n")

    looked_up: list[str] = []

    def resolve_program(name: str) -> str | None:
        looked_up.append(name)
        if name == "node":
            return "/usr/bin/node"
        raise AssertionError(f"offline certification queried forbidden tool {name!r}")

    def forbid_process(*_args, **_kwargs):
        raise AssertionError("missing prebuilt artifact spawned a process")

    monkeypatch.setattr(certifier.shutil, "which", resolve_program)
    monkeypatch.setattr(certifier.subprocess, "run", forbid_process)

    assert certifier._ensure_typescript_built(tmp_path) is None
    assert looked_up == ["node"]


def test_existing_prebuilt_artifact_is_resolved_without_build(
    certifier,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    index = (
        tmp_path
        / certifier.TS_PACKAGE_RELATIVE
        / "dist"
        / "src"
        / "index.js"
    )
    index.parent.mkdir(parents=True)
    index.write_text("export const evaluateCase = value => value;\n")

    monkeypatch.setattr(
        certifier.shutil,
        "which",
        lambda name: "/usr/bin/node" if name == "node" else None,
    )
    monkeypatch.setattr(
        certifier.subprocess,
        "run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("prebuilt resolution spawned a process")
        ),
    )

    assert certifier._ensure_typescript_built(tmp_path) == index


def test_certificate_declares_and_obeys_offline_prebuilt_policy(certifier) -> None:
    receipt = certifier.certify_runtime_mtl_semantics(repo_root=REPO_ROOT)
    policy = receipt["policy"]

    assert policy["no_external_parity_install"] is True
    assert policy["requires_prebuilt_typescript_artifact"] is True
    assert policy["certification_never_builds_typescript"] is True
    assert receipt["parity"]["certification_builds_or_installs"] is False
    if receipt["parity"].get("prebuilt"):
        binding = receipt["parity"]["prebuilt"]
        assert all(
            binding[key]
            for key in (
                "index_sha256",
                "package_json_sha256",
                "package_lock_sha256",
                "source_index_sha256",
                "node_executable_sha256",
            )
        )


def test_missing_prebuilt_parity_blocks_semantic_certification(
    certifier,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unavailable = certifier.CheckResult(
        check_id="parity.python_typescript",
        kind="parity",
        status="skipped",
        expected="prebuilt_digest_bound_parity",
        observed="typescript_prebuilt_unavailable",
        detail="offline prebuilt artifact is absent",
    )
    monkeypatch.setattr(
        certifier,
        "run_python_typescript_parity",
        lambda **_kwargs: (
            unavailable,
            {
                "package_present": True,
                "prebuilt_required": True,
                "certification_builds_or_installs": False,
            },
        ),
    )

    receipt = certifier.certify_runtime_mtl_semantics(repo_root=REPO_ROOT)

    assert receipt["certified"] is False
    assert receipt["production_certified"] is False
    assert receipt["promotion_blocked"] is True
    assert "typescript_parity_unavailable" in receipt["block_reasons"]
