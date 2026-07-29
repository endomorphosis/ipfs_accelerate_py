"""Tests for explicit lazy contract-repair dependency provisioning."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from ipfs_accelerate_py.agent_supervisor.integrations import (
    contract_repair_dependencies as dependencies,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def _completed(
    command: list[str],
    *,
    returncode: int = 0,
    stdout: str = "",
    stderr: str = "",
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(command, returncode, stdout, stderr)


def _write_typescript_fixture(root: Path, version: str) -> None:
    package_dir = root / "node_modules" / "typescript"
    (package_dir / "lib").mkdir(parents=True)
    (package_dir / "bin").mkdir(parents=True)
    (root / "node_modules" / ".bin").mkdir(parents=True)
    (package_dir / "package.json").write_text(
        json.dumps({"name": "typescript", "version": version}),
        encoding="utf-8",
    )
    (package_dir / "lib" / "typescript.js").write_text(
        (
            "module.exports = {"
            f'version: "{version}", '
            "createSourceFile() {}, createProgram() {}, transpileModule() {}, "
            "SyntaxKind: {}};\n"
        ),
        encoding="utf-8",
    )
    (package_dir / "bin" / "tsc").write_text(
        "#!/usr/bin/env node\n",
        encoding="utf-8",
    )
    tsc = root / "node_modules" / ".bin" / ("tsc.cmd" if sys.platform == "win32" else "tsc")
    tsc.write_text("#!/usr/bin/env node\n", encoding="utf-8")
    tsc.chmod(0o755)


def _typescript_probe_result(command: list[str]) -> subprocess.CompletedProcess[str]:
    if "-e" in command:
        return _completed(
            command,
            stdout=json.dumps(
                {
                    "version": dependencies.PINNED_TYPESCRIPT_VERSION,
                    "createSourceFile": True,
                    "createProgram": True,
                    "transpileModule": True,
                    "SyntaxKind": True,
                }
            ),
        )
    return _completed(
        command,
        stdout=f"Version {dependencies.PINNED_TYPESCRIPT_VERSION}\n",
    )


def test_module_import_is_cold_for_optional_toolchains() -> None:
    code = """
import sys
import ipfs_accelerate_py.agent_supervisor.integrations.contract_repair_dependencies
assert 'z3' not in sys.modules
assert 'cvc5' not in sys.modules
assert 'mypy' not in sys.modules
assert 'ruff' not in sys.modules
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=False,
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env={"PYTHONPATH": str(REPO_ROOT), "IPFS_ACCEL_SKIP_CORE": "1"},
    )
    assert completed.returncode == 0, completed.stderr


def test_python_requirements_are_packaging_source_of_truth() -> None:
    declared = {
        line.strip()
        for line in (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }
    assert set(dependencies.CONTRACT_REPAIR_PYTHON_REQUIREMENTS) <= declared
    assert not any(line.lower().startswith("typescript") for line in declared)

    setup_source = (REPO_ROOT / "setup.py").read_text(encoding="utf-8")
    for distribution in ("z3-solver", "cvc5", "mypy", "ruff"):
        assert distribution in setup_source
    assert "_require_contract_repair_distributions(install_requires)" in setup_source


def test_python_loader_requires_explicit_install(monkeypatch: pytest.MonkeyPatch) -> None:
    missing = dependencies.ContractRepairDependencyReceipt(
        "z3",
        "missing",
        dependencies.PYTHON_DEPENDENCY_SPECS["z3"].requirement,
        reason="requested_module_missing",
    )
    monkeypatch.setattr(dependencies, "probe_python_dependency", lambda _name: missing)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("pip must not run without explicit opt-in")

    receipt = dependencies.ensure_python_dependency(
        "z3",
        auto_install=False,
        runner=forbidden,
    )
    assert receipt.status == "install_disabled"
    assert receipt.install_attempted is False


def test_python_loader_uses_only_allowlisted_requirement(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    spec = dependencies.PYTHON_DEPENDENCY_SPECS["mypy"]
    receipts = iter(
        (
            dependencies.ContractRepairDependencyReceipt(
                "mypy", "missing", spec.requirement, reason="requested_module_missing"
            ),
            dependencies.ContractRepairDependencyReceipt(
                "mypy", "missing", spec.requirement, reason="requested_module_missing"
            ),
            dependencies.ContractRepairDependencyReceipt(
                "mypy",
                "available",
                spec.requirement,
                version="1.20.2",
                module_path="/managed/mypy/__init__.py",
                executable_path="/managed/bin/mypy",
            ),
        )
    )
    monkeypatch.setattr(
        dependencies,
        "probe_python_dependency",
        lambda _name: next(receipts),
    )
    monkeypatch.setattr(dependencies, "_pip_install_flags", lambda: [])
    monkeypatch.setattr(dependencies.tempfile, "gettempdir", lambda: str(tmp_path))
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        calls.append(list(command))
        return _completed(list(command))

    receipt = dependencies.ensure_python_dependency(
        "mypy",
        auto_install=True,
        runner=runner,
    )
    assert receipt.available
    assert receipt.install_attempted is True
    assert calls == [[sys.executable, "-m", "pip", "install", spec.requirement]]


@pytest.mark.parametrize(
    ("dependency_id", "version"),
    (
        ("z3", "3.0.0"),
        ("cvc5", "1.3.3rc1"),
        ("mypy", "2.0.0"),
        ("ruff", "1.0.0"),
    ),
)
def test_python_probe_rejects_out_of_range_distributions(
    dependency_id: str,
    version: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(dependencies.importlib.metadata, "version", lambda _name: version)
    receipt = dependencies.probe_python_dependency(
        dependency_id,
        importer=lambda _name: SimpleNamespace(__file__="/fixture/module.py"),
        which=lambda command: f"/fixture/bin/{command}",
    )
    assert receipt.status == "incompatible"
    assert receipt.reason == "distribution_version_mismatch"


def test_typescript_probe_binds_cli_and_compiler_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "typescript"
    _write_typescript_fixture(root, dependencies.PINNED_TYPESCRIPT_VERSION)
    monkeypatch.setattr(
        dependencies.shutil,
        "which",
        lambda command, **_kwargs: "/usr/bin/node" if command == "node" else None,
    )

    def runner(command, **_kwargs):
        return _typescript_probe_result(list(command))

    receipt = dependencies.probe_typescript_toolchain(root=root, runner=runner)
    assert receipt.available
    assert receipt.version == dependencies.PINNED_TYPESCRIPT_VERSION
    assert receipt.executable_path.endswith("node_modules/.bin/tsc")
    assert receipt.module_path.endswith("node_modules/typescript")
    assert receipt.details["compiler_api_path"].endswith("lib/typescript.js")
    assert receipt.details["compiler_api_sha256"]
    assert set(receipt.details["api_symbols_verified"]) == {
        "createSourceFile",
        "createProgram",
        "transpileModule",
        "SyntaxKind",
    }


def test_typescript_probe_rejects_broken_compiler_api(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "typescript"
    _write_typescript_fixture(root, dependencies.PINNED_TYPESCRIPT_VERSION)
    monkeypatch.setattr(
        dependencies.shutil,
        "which",
        lambda command, **_kwargs: "/usr/bin/node" if command == "node" else None,
    )

    def runner(command, **_kwargs):
        command = list(command)
        if "-e" in command:
            return _completed(
                command,
                returncode=3,
                stdout=json.dumps(
                    {
                        "version": dependencies.PINNED_TYPESCRIPT_VERSION,
                        "createSourceFile": False,
                    }
                ),
            )
        return _typescript_probe_result(command)

    receipt = dependencies.probe_typescript_toolchain(root=root, runner=runner)
    assert receipt.status == "incompatible"
    assert receipt.reason == "compiler_api_canary_failed"


def test_typescript_version_mismatch_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "typescript"
    _write_typescript_fixture(root, "5.6.3")
    monkeypatch.setattr(
        dependencies.shutil,
        "which",
        lambda command, **_kwargs: "/usr/bin/node" if command == "node" else None,
    )
    receipt = dependencies.probe_typescript_toolchain(root=root)
    assert receipt.status == "incompatible"
    assert receipt.reason == "package_version_mismatch"
    assert receipt.details["expected_version"] == dependencies.PINNED_TYPESCRIPT_VERSION


def test_typescript_loader_is_explicit_and_uses_pinned_npm_package(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "typescript"
    monkeypatch.setattr(
        dependencies.shutil,
        "which",
        lambda command, **_kwargs: f"/usr/bin/{command}" if command in {"node", "npm"} else None,
    )
    calls: list[list[str]] = []

    def runner(command, **_kwargs):
        command = list(command)
        calls.append(command)
        if "install" in command:
            _write_typescript_fixture(root, dependencies.PINNED_TYPESCRIPT_VERSION)
            return _completed(command)
        return _typescript_probe_result(command)

    disabled = dependencies.ensure_typescript_toolchain(
        auto_install=False,
        root=root,
        runner=runner,
    )
    assert disabled.status == "install_disabled"
    assert calls == []

    installed = dependencies.ensure_typescript_toolchain(
        auto_install=True,
        root=root,
        runner=runner,
    )
    assert installed.available
    assert installed.install_attempted is True
    install_call = next(command for command in calls if "install" in command)
    assert "--ignore-scripts" in install_call
    assert "--save-exact" in install_call
    assert install_call[-1] == dependencies.TYPESCRIPT_REQUIREMENT


def test_managed_environment_is_detect_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    receipt = dependencies.ContractRepairDependencyReceipt(
        "typescript",
        "available",
        dependencies.TYPESCRIPT_REQUIREMENT,
        version=dependencies.PINNED_TYPESCRIPT_VERSION,
        module_path="/managed/node_modules/typescript",
        executable_path="/managed/node_modules/.bin/tsc",
    )
    monkeypatch.setattr(
        dependencies,
        "probe_typescript_toolchain",
        lambda **_kwargs: receipt,
    )
    bindings = dependencies.contract_repair_toolchain_environment(environ={"PATH": "/usr/bin"})
    assert bindings["TYPESCRIPT_PATH"] == receipt.module_path
    assert bindings["PATH"].split(":")[0] == "/managed/node_modules/.bin"
