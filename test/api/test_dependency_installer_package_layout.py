from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import comprehensive_dependency_installer
from scripts import dependency_installer


INSTALLER_MODULES = (
    dependency_installer,
    comprehensive_dependency_installer,
)


def _write_packaging_metadata(package_path: Path) -> None:
    package_path.mkdir(parents=True, exist_ok=True)
    (package_path / "pyproject.toml").write_text(
        "[build-system]\nrequires = []\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize("module", INSTALLER_MODULES)
def test_local_package_resolution_prefers_repository_root(module, tmp_path: Path) -> None:
    repo_root = tmp_path / "ipfs_accelerate"
    repo_root.mkdir()
    root_package = repo_root / "ipfs_datasets_py"
    legacy_package = repo_root / "external" / "ipfs_datasets_py"
    sibling_package = tmp_path / "ipfs_datasets"
    _write_packaging_metadata(root_package)
    _write_packaging_metadata(legacy_package)
    _write_packaging_metadata(sibling_package)

    assert (
        module._resolve_local_package_path(repo_root, "ipfs_datasets_py")
        == root_package
    )


@pytest.mark.parametrize("module", INSTALLER_MODULES)
def test_local_package_resolution_supports_legacy_external_layout(
    module,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "ipfs_accelerate"
    repo_root.mkdir()
    (repo_root / "ipfs_datasets_py").mkdir()
    legacy_package = repo_root / "external" / "ipfs_datasets_py"
    sibling_package = tmp_path / "ipfs_datasets"
    _write_packaging_metadata(legacy_package)
    _write_packaging_metadata(sibling_package)

    assert (
        module._resolve_local_package_path(repo_root, "ipfs_datasets_py")
        == legacy_package
    )


@pytest.mark.parametrize("module", INSTALLER_MODULES)
@pytest.mark.parametrize(
    ("package", "sibling_name"),
    (
        ("ipfs_datasets_py", "ipfs_datasets"),
        ("ipfs_kit_py", "ipfs_kit"),
    ),
)
def test_empty_root_gitlink_resolves_to_valid_workspace_sibling(
    module,
    package: str,
    sibling_name: str,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "ipfs_accelerate"
    repo_root.mkdir()
    (repo_root / package).mkdir()
    (repo_root / "external" / package).mkdir(parents=True)
    sibling_package = tmp_path / sibling_name
    _write_packaging_metadata(sibling_package)

    assert module._resolve_local_package_path(repo_root, package) == sibling_package


@pytest.mark.parametrize("module", INSTALLER_MODULES)
def test_local_package_resolution_uses_root_for_new_clone(
    module,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "ipfs_accelerate"
    repo_root.mkdir()

    assert module._resolve_local_package_path(
        repo_root,
        "ipfs_datasets_py",
    ) == (repo_root / "ipfs_datasets_py")


def test_dependency_installer_registers_ipfs_datasets_source() -> None:
    installer = dependency_installer.DependencyInstaller()

    assert "ipfs_datasets_py" in installer.local_packages
    assert installer.git_sources["ipfs_datasets_py"] == {
        "repo": "https://github.com/endomorphosis/ipfs_datasets_py.git",
        "branch": "main",
    }


def test_comprehensive_installer_registers_ipfs_datasets_source() -> None:
    installer = (
        comprehensive_dependency_installer.ComprehensiveDependencyInstaller()
    )

    assert "ipfs_datasets_py" in installer.local_packages
    assert installer.git_sources["ipfs_datasets_py"] == {
        "repo": "https://github.com/endomorphosis/ipfs_datasets_py.git",
        "branch": "main",
    }


@pytest.mark.parametrize("module", INSTALLER_MODULES)
def test_existing_local_package_is_installed_without_git_commands(
    module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "ipfs_accelerate"
    repo_root.mkdir()
    package_path = repo_root / "ipfs_datasets_py"
    package_path.mkdir()
    (package_path / ".git").mkdir()
    _write_packaging_metadata(package_path)
    installer = (
        module.DependencyInstaller()
        if module is dependency_installer
        else module.ComprehensiveDependencyInstaller()
    )
    installer.repo_root = repo_root
    installer.local_packages = ["ipfs_datasets_py"]
    commands: list[list[str]] = []

    def record_command(command: list[str], **_kwargs):
        commands.append(command)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(module.subprocess, "run", record_command)

    assert installer.install_local_packages() == {"ipfs_datasets_py": True}
    assert commands == [
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(package_path),
        ]
    ]


@pytest.mark.parametrize("module", INSTALLER_MODULES)
def test_existing_nonpackage_path_is_not_replaced_or_executed(
    module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "ipfs_accelerate"
    repo_root.mkdir()
    package_path = repo_root / "ipfs_datasets_py"
    package_path.mkdir()
    marker = package_path / "dirty-worktree-marker"
    marker.write_text("preserve me", encoding="utf-8")
    installer = (
        module.DependencyInstaller()
        if module is dependency_installer
        else module.ComprehensiveDependencyInstaller()
    )
    installer.repo_root = repo_root
    installer.local_packages = ["ipfs_datasets_py"]

    def reject_command(*_args, **_kwargs):
        pytest.fail("existing paths without packaging metadata must run no command")

    monkeypatch.setattr(module.subprocess, "run", reject_command)

    assert installer.install_local_packages() == {"ipfs_datasets_py": False}
    assert marker.read_text(encoding="utf-8") == "preserve me"


@pytest.mark.parametrize("module", INSTALLER_MODULES)
def test_missing_package_is_cloned_to_repository_root_before_install(
    module,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "ipfs_accelerate"
    repo_root.mkdir()
    package_path = repo_root / "ipfs_datasets_py"
    installer = (
        module.DependencyInstaller()
        if module is dependency_installer
        else module.ComprehensiveDependencyInstaller()
    )
    installer.repo_root = repo_root
    installer.local_packages = ["ipfs_datasets_py"]
    commands: list[list[str]] = []

    def record_command(command: list[str], **_kwargs):
        commands.append(command)
        if command[:2] == ["git", "clone"]:
            _write_packaging_metadata(package_path)
        return SimpleNamespace(returncode=0, stderr="", stdout="")

    monkeypatch.setattr(module.subprocess, "run", record_command)

    assert installer.install_local_packages() == {"ipfs_datasets_py": True}
    assert commands == [
        [
            "git",
            "clone",
            "--depth",
            "1",
            "--branch",
            "main",
            "https://github.com/endomorphosis/ipfs_datasets_py.git",
            str(package_path),
        ],
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-e",
            str(package_path),
        ],
    ]


def test_comprehensive_installer_processes_networking_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installer = (
        comprehensive_dependency_installer.ComprehensiveDependencyInstaller()
    )
    installer.dependencies = {
        "network-probe": {
            "pip_name": "network-probe>=1",
            "import_name": "network_probe",
            "description": "network category regression probe",
            "category": "networking",
            "critical": False,
        }
    }
    installed: list[tuple[str, str | None]] = []

    monkeypatch.setattr(installer, "install_local_packages", lambda: {})
    monkeypatch.setattr(
        installer,
        "check_dependency",
        lambda _name, _import_name=None: False,
    )
    monkeypatch.setattr(
        installer,
        "install_package",
        lambda name, pip_name=None, *_args: (
            installed.append((name, pip_name)) or True
        ),
    )
    monkeypatch.setattr(installer, "_verify_installations", lambda: None)
    monkeypatch.setattr(installer, "_save_installation_log", lambda: None)
    monkeypatch.setattr(installer, "_check_critical_dependencies", lambda: {})

    report = installer.install_all_dependencies()

    assert installed == [("network-probe", "network-probe>=1")]
    assert report["installed"] == 1
    assert report["failed"] == 0
