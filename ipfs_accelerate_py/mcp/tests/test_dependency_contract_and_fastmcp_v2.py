"""Regression tests for dependency and legacy FastMCP-v2 wiring contracts."""

from __future__ import annotations

import importlib.metadata
import importlib.util
import inspect
import ast
import hashlib
import os
from functools import partial
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest
from packaging.requirements import Requirement

os.environ.setdefault("IPFS_ACCEL_SKIP_CORE", "1")
os.environ.setdefault("IPFS_ACCEL_AUTO_INSTALL", "0")

from ipfs_accelerate_py.mcp import server as server_module
from ipfs_accelerate_py.utils import auto_install


REPO_ROOT = Path(__file__).resolve().parents[3]
FASTMCP_REQUIREMENT = "fastmcp==2.14.7; python_version >= '3.10'"
FASTAPI_REQUIREMENT = "fastapi>=0.110.0,<1.0.0"
UVICORN_LEGACY_REQUIREMENT = (
    "uvicorn>=0.27.0,<0.35.0; python_version < '3.10'"
)
UVICORN_REQUIREMENT = (
    "uvicorn>=0.35.0,<1.0.0; python_version >= '3.10'"
)
WEBSOCKETS_LEGACY_REQUIREMENT = (
    "websockets==10.4; python_version < '3.10'"
)
WEBSOCKETS_REQUIREMENT = (
    "websockets>=15.0.1; python_version >= '3.10'"
)
AUDITED_FASTMCP_ORIGIN = Path("/audited/site-packages/fastmcp/__init__.py")


class _FakeDistribution:
    def __init__(self, version: str = "2.14.7") -> None:
        self.version = version
        self.files = (Path("fastmcp/__init__.py"),)

    def locate_file(self, path: Path) -> Path:
        return Path("/audited/site-packages") / path


def _fake_fastmcp_module(
    *,
    version: str | None = "2.14.7",
    origin: Path = AUDITED_FASTMCP_ORIGIN,
):
    attributes: dict[str, object] = {
        "FastMCP": _FakeFastMCP2,
        "__file__": str(origin),
        "__spec__": SimpleNamespace(origin=str(origin)),
    }
    if version is not None:
        attributes["__version__"] = version
    return SimpleNamespace(**attributes)


def _requirement_lines(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _find_requirement(requirements: list[str], distribution: str) -> Requirement:
    for raw_requirement in requirements:
        requirement = Requirement(raw_requirement)
        if requirement.name.lower() == distribution.lower():
            return requirement
    raise AssertionError(f"{distribution} is not declared")


def _find_requirements(requirements: list[str], distribution: str) -> list[Requirement]:
    matches = [
        Requirement(raw_requirement)
        for raw_requirement in requirements
        if Requirement(raw_requirement).name.lower() == distribution.lower()
    ]
    if not matches:
        raise AssertionError(f"{distribution} is not declared")
    return matches


def _disable_registration(monkeypatch: pytest.MonkeyPatch) -> None:
    for method_name in ("_register_tools", "_register_resources", "_register_prompts"):
        monkeypatch.setattr(
            server_module.IPFSAccelerateMCPServer,
            method_name,
            lambda self: None,
        )


def test_requirements_are_the_setup_source_of_truth() -> None:
    root_requirements = _requirement_lines(REPO_ROOT / "requirements.txt")
    captured: dict[str, object] = {}

    def _capture_setup(**kwargs: object) -> None:
        captured.update(kwargs)

    with patch("setuptools.setup", side_effect=_capture_setup):
        runpy.run_path(str(REPO_ROOT / "setup.py"), run_name="packaging_contract_test")

    assert captured["install_requires"] == root_requirements
    assert captured["python_requires"] == ">=3.8"

    fastmcp = _find_requirement(root_requirements, "fastmcp")
    assert str(fastmcp.specifier) == "==2.14.7"
    assert fastmcp.marker is not None
    assert "python_version" in str(fastmcp.marker)
    assert "3.10" in str(fastmcp.marker)
    assert fastmcp.marker.evaluate({"python_version": "3.8"}) is False
    assert fastmcp.marker.evaluate({"python_version": "3.10"}) is True

    urllib3 = _find_requirement(root_requirements, "urllib3")
    assert urllib3.specifier.contains("2.0")
    assert urllib3.specifier.contains("2.7.0")
    assert not urllib3.specifier.contains("1.26.20")
    assert not urllib3.specifier.contains("3.0.0")


def test_all_fastmcp_package_and_install_surfaces_use_the_exact_pin() -> None:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    extras = pyproject["project"]["optional-dependencies"]
    for extra_name in ("mcp", "all"):
        requirement = _find_requirement(extras[extra_name], "fastmcp")
        assert str(requirement.specifier) == "==2.14.7"
        assert requirement.marker is not None

    requirement_files = (
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "ipfs_accelerate_py/mcp/requirements.txt",
        REPO_ROOT / "ipfs_accelerate_py/mcp/requirements-mcp.txt",
    )
    for requirement_file in requirement_files:
        requirement = _find_requirement(
            _requirement_lines(requirement_file),
            "fastmcp",
        )
        assert str(requirement.specifier) == "==2.14.7"
        assert requirement.marker is not None

    exact_install_surfaces = (
        REPO_ROOT / "Dockerfile",
        REPO_ROOT / "deployments/docker/docker-entrypoint.sh",
        REPO_ROOT / "scripts/dependency_installer.py",
        REPO_ROOT / "scripts/comprehensive_dependency_installer.py",
        REPO_ROOT / "scripts/comprehensive_mcp_server.py",
    )
    for install_surface in exact_install_surfaces:
        source = install_surface.read_text(encoding="utf-8")
        assert "fastmcp==2.14.7" in source, install_surface
        assert "python_version >= '3.10'" in source, install_surface
        assert "fastmcp>=0.1.0" not in source, install_surface


def test_python38_build_and_server_import_contracts_are_declared() -> None:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    pyproject_source = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    pyproject = tomllib.loads(pyproject_source)
    tomli = _find_requirement(pyproject["build-system"]["requires"], "tomli")
    assert tomli.specifier.contains("2.0.1")
    assert not tomli.specifier.contains("3.0.0")
    assert tomli.marker is not None
    assert tomli.marker.evaluate({"python_version": "3.8"}) is True
    assert tomli.marker.evaluate({"python_version": "3.11"}) is False

    server_source = (REPO_ROOT / "ipfs_accelerate_py/mcp/server.py").read_text(
        encoding="utf-8"
    )
    assert "from __future__ import annotations" in server_source
    ast.parse(server_source, feature_version=(3, 8))


def test_all_changed_requirement_surfaces_are_pep508_parseable() -> None:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    requirement_files = (
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "ipfs_accelerate_py/mcp/requirements.txt",
        REPO_ROOT / "ipfs_accelerate_py/mcp/requirements-mcp.txt",
    )
    for path in requirement_files:
        for raw_requirement in _requirement_lines(path):
            Requirement(raw_requirement)

    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text("utf-8"))
    for requirements in pyproject["project"]["optional-dependencies"].values():
        for raw_requirement in requirements:
            Requirement(raw_requirement)


def test_every_lazy_mcp_distribution_has_aligned_declarative_surfaces() -> None:
    try:
        import tomllib
    except ImportError:  # pragma: no cover - Python 3.10
        import tomli as tomllib

    expected = {
        "fastmcp": [Requirement(FASTMCP_REQUIREMENT)],
        "fastapi": [Requirement(FASTAPI_REQUIREMENT)],
        "uvicorn": [
            Requirement(UVICORN_LEGACY_REQUIREMENT),
            Requirement(UVICORN_REQUIREMENT),
        ],
        "websockets": [
            Requirement(WEBSOCKETS_LEGACY_REQUIREMENT),
            Requirement(WEBSOCKETS_REQUIREMENT),
        ],
    }
    root = _requirement_lines(REPO_ROOT / "requirements.txt")
    mcp_files = (
        REPO_ROOT / "ipfs_accelerate_py/mcp/requirements.txt",
        REPO_ROOT / "ipfs_accelerate_py/mcp/requirements-mcp.txt",
    )
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    mcp_extra = pyproject["project"]["optional-dependencies"]["mcp"]

    for distribution, expected_requirements in expected.items():
        declaration_surfaces = [
            _find_requirements(root, distribution),
            _find_requirements(mcp_extra, distribution),
            *(
                _find_requirements(_requirement_lines(path), distribution)
                for path in mcp_files
            ),
        ]
        expected_pairs = {
            (str(requirement.specifier), str(requirement.marker or ""))
            for requirement in expected_requirements
        }
        for declarations in declaration_surfaces:
            assert {
                (str(declaration.specifier), str(declaration.marker or ""))
                for declaration in declarations
            } == expected_pairs


def test_fastmcp_transitive_uvicorn_and_websocket_intersections_are_satisfiable() -> None:
    modern_uvicorn = Requirement(UVICORN_REQUIREMENT)
    legacy_uvicorn = Requirement(UVICORN_LEGACY_REQUIREMENT)
    assert modern_uvicorn.specifier.contains("0.35.0")
    assert not modern_uvicorn.specifier.contains("0.34.3")
    assert legacy_uvicorn.specifier.contains("0.34.3")
    assert not legacy_uvicorn.specifier.contains("0.35.0")

    mcp_requirements = _requirement_lines(
        REPO_ROOT / "ipfs_accelerate_py/mcp/requirements-mcp.txt"
    )
    websocket_requirements = _find_requirements(mcp_requirements, "websockets")
    applicable_on_38 = [
        requirement
        for requirement in websocket_requirements
        if requirement.marker is not None
        and requirement.marker.evaluate(
            {"python_version": "3.8", "extra": ""}
        )
    ]
    applicable_on_310 = [
        requirement
        for requirement in websocket_requirements
        if requirement.marker is not None
        and requirement.marker.evaluate(
            {"python_version": "3.10", "extra": ""}
        )
    ]
    assert len(applicable_on_38) == 1
    assert applicable_on_38[0].specifier.contains("10.4")
    assert not applicable_on_38[0].specifier.contains("15.0.1")
    assert len(applicable_on_310) == 1
    assert applicable_on_310[0].specifier.contains("15.0.1")
    assert applicable_on_310[0].specifier.contains("16.0")
    assert applicable_on_310[0].specifier.contains("17.0")
    assert not applicable_on_310[0].specifier.contains("10.4")

    ipfs_kit = _find_requirement(mcp_requirements, "ipfs_kit_py")
    libp2p = _find_requirement(mcp_requirements, "libp2p")
    assert ipfs_kit.marker is not None
    assert ipfs_kit.marker.evaluate({"python_version": "3.11"}) is False
    assert ipfs_kit.marker.evaluate({"python_version": "3.12"}) is True
    assert libp2p.marker is not None
    assert libp2p.marker.evaluate({"python_version": "3.9"}) is False
    assert libp2p.marker.evaluate({"python_version": "3.10"}) is True


def test_all_legacy_urllib3_constraints_accept_the_vendored_intersection() -> None:
    constraint_surfaces = (
        REPO_ROOT / "requirements.txt",
        REPO_ROOT / "requirements_enhanced_scraper.txt",
        REPO_ROOT / "requirements_webnn_webgpu.txt",
        REPO_ROOT / "scripts/zero_touch_install.sh",
    )
    for constraint_surface in constraint_surfaces:
        source = constraint_surface.read_text(encoding="utf-8")
        assert "urllib3<2" not in source, constraint_surface
        assert "urllib3>=2,<3" in source, constraint_surface


def test_mcp_package_import_is_cold_even_when_auto_install_is_enabled() -> None:
    script = r"""
import os
import subprocess
import sys

os.environ["IPFS_ACCEL_SKIP_CORE"] = "1"
os.environ["IPFS_ACCEL_AUTO_INSTALL"] = "1"

def forbidden(*args, **kwargs):
    raise AssertionError("package import attempted a subprocess")

subprocess.run = forbidden
subprocess.Popen = forbidden
subprocess.check_call = forbidden
subprocess.check_output = forbidden

from ipfs_accelerate_py.utils import auto_install
auto_install.ensure_packages = forbidden

import ipfs_accelerate_py.mcp

assert "fastmcp" not in sys.modules
print("cold-import-ok")
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(REPO_ROOT)
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, completed.stderr
    assert "cold-import-ok" in completed.stdout


def test_active_fastmcp_scripts_do_not_mutate_global_import_state() -> None:
    sources = {
        path: path.read_text(encoding="utf-8")
        for path in (
            REPO_ROOT / "ipfs_accelerate_py/mcp/ai_model_server.py",
            REPO_ROOT / "scripts/comprehensive_mcp_server.py",
        )
    }
    for path, source in sources.items():
        assert "sys.path.append" not in source, path
        assert "del sys.modules" not in source, path
    assert "instructions=" in sources[
        REPO_ROOT / "ipfs_accelerate_py/mcp/ai_model_server.py"
    ]
    comprehensive_source = sources[REPO_ROOT / "scripts/comprehensive_mcp_server.py"]
    assert "instructions=" in comprehensive_source
    assert "description=\"Complete AI inference platform" not in comprehensive_source
    for source in sources.values():
        assert "await self.mcp.run_async(" in source
        assert "await self.mcp.run(" not in source


def test_comprehensive_installer_respects_python_and_auto_install_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.comprehensive_mcp_server as comprehensive_module
    import subprocess as stdlib_subprocess

    subprocess_calls: list[object] = []

    def _forbidden_subprocess(*args, **kwargs):
        subprocess_calls.append(args)
        raise AssertionError("installer subprocess was not authorized")

    monkeypatch.setattr(stdlib_subprocess, "run", _forbidden_subprocess)
    installer = comprehensive_module.ComprehensiveMCPServer._install_dependencies
    instance = object.__new__(comprehensive_module.ComprehensiveMCPServer)

    monkeypatch.setenv("IPFS_ACCEL_AUTO_INSTALL", "1")
    monkeypatch.setattr(comprehensive_module.sys, "version_info", (3, 9, 19))
    installer(instance)

    monkeypatch.setenv("IPFS_ACCEL_AUTO_INSTALL", "0")
    monkeypatch.setattr(comprehensive_module.sys, "version_info", (3, 12, 0))
    installer(instance)

    assert subprocess_calls == []


@pytest.mark.parametrize("version", [(3, 8, 20), (3, 9, 20)])
def test_executable_installers_skip_libp2p_below_python310(
    monkeypatch: pytest.MonkeyPatch,
    version: tuple[int, int, int],
) -> None:
    import scripts.comprehensive_dependency_installer as comprehensive_installer
    import scripts.dependency_installer as dependency_installer

    subprocess_calls: list[object] = []

    def _forbidden_subprocess(*args, **kwargs):
        subprocess_calls.append(args)
        raise AssertionError("inapplicable libp2p attempted a subprocess")

    monkeypatch.setattr(sys, "version_info", version)
    basic = dependency_installer.DependencyInstaller()
    comprehensive = comprehensive_installer.ComprehensiveDependencyInstaller()
    monkeypatch.setattr(subprocess, "run", _forbidden_subprocess)
    assert basic.install_package(
        dependency_installer.LIBP2P_REQUIREMENT,
        "libp2p",
    ) is True
    assert comprehensive.install_package(
        "libp2p",
        comprehensive_installer.LIBP2P_REQUIREMENT,
    ) is True
    assert subprocess_calls == []


@pytest.mark.parametrize(
    "version",
    [(3, 8, 20), (3, 9, 20), (3, 10, 15), (3, 11, 9)],
)
def test_executable_installers_skip_ipfs_kit_below_python312(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version: tuple[int, int, int],
) -> None:
    import scripts.comprehensive_dependency_installer as comprehensive_installer
    import scripts.dependency_installer as dependency_installer

    external_dir = tmp_path / "external"
    external_dir.mkdir()
    subprocess_calls: list[object] = []

    def _forbidden_subprocess(*args, **kwargs):
        subprocess_calls.append(args)
        raise AssertionError("inapplicable ipfs_kit_py attempted a subprocess")

    monkeypatch.setattr(sys, "version_info", version)
    basic = dependency_installer.DependencyInstaller()
    basic.external_dir = external_dir
    basic.local_packages = ["ipfs_kit_py"]
    comprehensive = comprehensive_installer.ComprehensiveDependencyInstaller()
    comprehensive.external_dir = external_dir
    comprehensive.local_packages = ["ipfs_kit_py"]
    monkeypatch.setattr(subprocess, "run", _forbidden_subprocess)

    assert basic.install_local_packages() == {"ipfs_kit_py": True}
    assert comprehensive.install_local_packages() == {"ipfs_kit_py": True}
    assert subprocess_calls == []


def test_legacy_installer_reports_inapplicable_critical_dependencies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scripts.comprehensive_dependency_installer as comprehensive_installer

    monkeypatch.setattr(sys, "version_info", (3, 9, 20))
    installer = comprehensive_installer.ComprehensiveDependencyInstaller()
    monkeypatch.setattr(installer, "check_dependency", lambda *args: False)

    verification = installer._verify_installations()
    critical = installer._check_critical_dependencies()

    assert verification["fastmcp"] is None
    assert critical["fastmcp"] is None
    assert comprehensive_installer._critical_dependency_counts(critical) == (
        sum(value is True for value in critical.values()),
        sum(value is not None for value in critical.values()),
    )


@pytest.mark.parametrize(
    ("version", "expected_calls", "expected_not_applicable"),
    [
        ((3, 8, 20), [], 1),
        ((3, 10, 15), ["libp2p"], 0),
    ],
)
def test_comprehensive_installer_traverses_networking_dependencies(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    version: tuple[int, int, int],
    expected_calls: list[str],
    expected_not_applicable: int,
) -> None:
    import scripts.comprehensive_dependency_installer as comprehensive_installer

    monkeypatch.setattr(sys, "version_info", version)
    installer = comprehensive_installer.ComprehensiveDependencyInstaller(
        log_file=str(tmp_path / "install.json")
    )
    install_calls: list[str] = []
    monkeypatch.setattr(installer, "install_local_packages", lambda: {})
    monkeypatch.setattr(installer, "check_dependency", lambda *args: True)

    def _record_install(package_name: str, *args, **kwargs) -> bool:
        install_calls.append(package_name)
        return True

    monkeypatch.setattr(installer, "install_package", _record_install)

    report = installer.install_all_dependencies(categories=["networking"])

    assert install_calls == expected_calls
    assert report["not_applicable"] == expected_not_applicable
    assert report["failed"] == 0
    assert report["success_rate"] == 100.0


def test_active_fastmcp_wrappers_dispatch_async_transport_contracts() -> None:
    import anyio
    import ipfs_accelerate_py.mcp.ai_model_server as ai_module
    import scripts.comprehensive_mcp_server as comprehensive_module

    class _Runtime:
        def __init__(self) -> None:
            self.calls: list[dict[str, object]] = []

        def run(self, **kwargs: object) -> None:
            raise AssertionError("synchronous FastMCP.run must not be used")

        async def run_async(self, **kwargs: object) -> None:
            self.calls.append(dict(kwargs))

    class _ModelManager:
        def close(self) -> None:
            return None

    for server_class in (
        ai_module.AIModelMCPServer,
        comprehensive_module.ComprehensiveMCPServer,
    ):
        instance = object.__new__(server_class)
        instance.mcp = _Runtime()
        instance.model_manager = _ModelManager()
        if server_class is comprehensive_module.ComprehensiveMCPServer:
            instance.available_model_types = {}

        anyio.run(
            partial(
                instance.run,
                transport="stdio",
                host="127.0.0.1",
                port=8123,
            )
        )
        anyio.run(
            partial(
                instance.run,
                transport="sse",
                host="127.0.0.1",
                port=8123,
            )
        )

        assert instance.mcp.calls == [
            {"transport": "stdio"},
            {"transport": "sse", "host": "127.0.0.1", "port": 8123},
        ]


def test_first_use_lazy_loader_requests_the_exact_fastmcp_pin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import_calls: list[dict[str, str]] = []
    distribution_calls: list[dict[str, str]] = []

    def _capture_imports(packages: dict[str, str]) -> dict[str, str]:
        import_calls.append(dict(packages))
        return {name: "skipped" for name in packages}

    def _capture_distributions(packages: dict[str, str]) -> dict[str, str]:
        distribution_calls.append(dict(packages))
        return {name: "skipped" for name in packages}

    monkeypatch.setattr(auto_install, "ensure_packages", _capture_imports)
    monkeypatch.setattr(
        auto_install,
        "ensure_distributions",
        _capture_distributions,
    )

    status = server_module._ensure_mcp_runtime_dependencies()

    expected = {
        "fastapi": FASTAPI_REQUIREMENT,
        "uvicorn": (
            UVICORN_REQUIREMENT
            if sys.version_info >= (3, 10)
            else UVICORN_LEGACY_REQUIREMENT
        ),
        "websockets": (
            WEBSOCKETS_REQUIREMENT
            if sys.version_info >= (3, 10)
            else WEBSOCKETS_LEGACY_REQUIREMENT
        ),
    }
    assert import_calls == [expected]
    expected_status = {name: "skipped" for name in expected}
    if sys.version_info >= (3, 10):
        assert distribution_calls == [{"fastmcp": FASTMCP_REQUIREMENT}]
        expected_status["fastmcp"] = "skipped"
    else:
        assert distribution_calls == []
    assert status == expected_status


def test_fastmcp_metadata_ensure_never_imports_package_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(auto_install, "_should_auto_install", lambda: False)
    monkeypatch.setattr(
        auto_install.importlib.metadata,
        "version",
        lambda distribution: "2.14.7",
    )
    monkeypatch.setattr(
        auto_install.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(
            AssertionError("metadata-only ensure imported package code")
        ),
    )

    result = auto_install.ensure_distributions(
        {"fastmcp": FASTMCP_REQUIREMENT}
    )

    assert result == {"fastmcp": "ok"}


def test_first_use_rejects_exact_metadata_shadow_without_executing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    shadow_imports: list[str] = []
    shadow_origin = REPO_ROOT / "fastmcp.py"

    monkeypatch.setattr(auto_install, "_should_auto_install", lambda: False)
    monkeypatch.setattr(
        auto_install.importlib.metadata,
        "version",
        lambda distribution: "2.14.7",
    )
    monkeypatch.setattr(
        auto_install,
        "ensure_packages",
        lambda packages: {name: "ok" for name in packages},
    )
    monkeypatch.setattr(
        auto_install.importlib,
        "import_module",
        lambda name: shadow_imports.append(f"auto:{name}"),
    )
    monkeypatch.setattr(
        server_module.importlib.metadata,
        "distribution",
        lambda distribution: _FakeDistribution(),
    )
    monkeypatch.setattr(
        server_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(shadow_origin)),
    )
    monkeypatch.setattr(
        server_module.importlib,
        "import_module",
        lambda name: shadow_imports.append(f"loader:{name}"),
    )

    dependency_status = server_module._ensure_mcp_runtime_dependencies()
    loaded, reason = server_module._import_fastmcp_v2()

    assert dependency_status["fastmcp"] == "ok"
    assert loaded is None
    assert "not owned" in reason
    assert shadow_imports == []


def test_fastmcp_loader_requires_exact_metadata_without_mutating_sys_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_module = _fake_fastmcp_module()
    before = list(sys.path)
    imports: list[str] = []

    monkeypatch.setattr(
        server_module.importlib.metadata,
        "distribution",
        lambda distribution: _FakeDistribution(),
    )
    monkeypatch.setattr(
        server_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(AUDITED_FASTMCP_ORIGIN)),
    )

    def _import(name: str):
        imports.append(name)
        return fake_module

    monkeypatch.setattr(server_module.importlib, "import_module", _import)

    loaded, reason = server_module._import_fastmcp_v2()

    assert loaded is fake_module
    assert reason == ""
    assert imports == ["fastmcp"]
    assert sys.path == before
    assert "sys.path" not in inspect.getsource(server_module._import_fastmcp_v2)


def test_fastmcp_loader_rejects_nearby_version_before_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    monkeypatch.setattr(
        server_module.importlib.metadata,
        "distribution",
        lambda distribution: _FakeDistribution("2.14.6"),
    )
    monkeypatch.setattr(
        server_module.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    loaded, reason = server_module._import_fastmcp_v2()

    assert loaded is None
    assert imports == []
    assert "2.14.6" in reason
    assert "2.14.7" in reason


def test_fastmcp_loader_skips_all_lookup_on_python39(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    lookups: list[str] = []
    monkeypatch.setattr(server_module.sys, "version_info", (3, 9, 19))
    monkeypatch.setattr(
        server_module.importlib.metadata,
        "distribution",
        lambda name: lookups.append(f"metadata:{name}"),
    )
    monkeypatch.setattr(
        server_module.importlib.util,
        "find_spec",
        lambda name: lookups.append(f"spec:{name}"),
    )
    monkeypatch.setattr(
        server_module.importlib,
        "import_module",
        lambda name: lookups.append(f"import:{name}"),
    )

    loaded, reason = server_module._import_fastmcp_v2()

    assert loaded is None
    assert lookups == []
    assert "Python 3.10" in reason


def test_fastmcp_loader_rejects_missing_module_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server_module.importlib.metadata,
        "distribution",
        lambda distribution: _FakeDistribution(),
    )
    monkeypatch.setattr(
        server_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(AUDITED_FASTMCP_ORIGIN)),
    )
    monkeypatch.setattr(
        server_module.importlib,
        "import_module",
        lambda name: _fake_fastmcp_module(version=None),
    )

    loaded, reason = server_module._import_fastmcp_v2()

    assert loaded is None
    assert "missing" in reason


def test_fastmcp_loader_rejects_shadow_origin_before_executing_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    imports: list[str] = []
    shadow_origin = REPO_ROOT / "fastmcp.py"
    monkeypatch.setattr(
        server_module.importlib.metadata,
        "distribution",
        lambda distribution: _FakeDistribution(),
    )
    monkeypatch.setattr(
        server_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(shadow_origin)),
    )
    monkeypatch.setattr(
        server_module.importlib,
        "import_module",
        lambda name: imports.append(name),
    )

    loaded, reason = server_module._import_fastmcp_v2()

    assert loaded is None
    assert imports == []
    assert "not owned" in reason


def test_fastmcp_loader_rechecks_origin_after_import(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server_module.importlib.metadata,
        "distribution",
        lambda distribution: _FakeDistribution(),
    )
    monkeypatch.setattr(
        server_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(AUDITED_FASTMCP_ORIGIN)),
    )
    monkeypatch.setattr(
        server_module.importlib,
        "import_module",
        lambda name: _fake_fastmcp_module(
            origin=REPO_ROOT / "raced-shadow-fastmcp.py"
        ),
    )

    loaded, reason = server_module._import_fastmcp_v2()

    assert loaded is None
    assert "changed during import" in reason


def test_fastmcp_loader_does_not_leak_secret_bearing_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        server_module.importlib.metadata,
        "distribution",
        lambda distribution: _FakeDistribution(),
    )
    monkeypatch.setattr(
        server_module.importlib.util,
        "find_spec",
        lambda name: SimpleNamespace(origin=str(AUDITED_FASTMCP_ORIGIN)),
    )
    monkeypatch.setattr(
        server_module.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(ImportError("token=supersecret")),
    )

    loaded, reason = server_module._import_fastmcp_v2()

    assert loaded is None
    assert "supersecret" not in reason
    assert "receipt=" in reason


def test_missing_transitive_import_can_trigger_authorized_metadata_only_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    expected_module = _fake_fastmcp_module()
    imports = iter(
        [
            (
                None,
                "FastMCP 2.14.7 could not be imported "
                "(ModuleNotFoundError; receipt=0123456789abcdef)",
            ),
            (expected_module, ""),
        ]
    )
    repair_calls: list[tuple[dict[str, str], bool]] = []
    monkeypatch.setattr(server_module, "_import_fastmcp_v2", lambda: next(imports))

    def _repair(packages: dict[str, str], *, force: bool = False):
        repair_calls.append((dict(packages), force))
        return {"fastmcp": "installed"}

    monkeypatch.setattr(auto_install, "ensure_distributions", _repair)

    loaded, reason = server_module._import_fastmcp_v2_with_repair()

    assert loaded is expected_module
    assert reason == ""
    assert repair_calls == [({"fastmcp": FASTMCP_REQUIREMENT}, True)]


def test_missing_transitive_import_does_not_repair_without_authority(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    failure = (
        None,
        "FastMCP 2.14.7 could not be imported "
        "(ModuleNotFoundError; receipt=0123456789abcdef)",
    )
    monkeypatch.setattr(server_module, "_import_fastmcp_v2", lambda: failure)
    monkeypatch.setattr(auto_install, "_should_auto_install", lambda: False)

    loaded, reason = server_module._import_fastmcp_v2_with_repair()

    assert (loaded, reason) == failure


def test_auto_installer_repairs_an_importable_wrong_fastmcp_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_versions = iter(["3.4.5", "2.14.7"])
    pip_requests: list[str] = []
    events: list[str] = []

    monkeypatch.setattr(
        auto_install.importlib,
        "import_module",
        lambda import_name: (
            events.append(f"import:{import_name}")
            or SimpleNamespace(__name__=import_name)
        ),
    )
    monkeypatch.setattr(
        auto_install.importlib.metadata,
        "version",
        lambda distribution: next(installed_versions),
    )
    monkeypatch.setattr(auto_install, "_should_auto_install", lambda: True)

    def _fake_pip_install(requirement: str) -> tuple[bool, str]:
        events.append("pip")
        pip_requests.append(requirement)
        return True, ""

    monkeypatch.setattr(auto_install, "_pip_install", _fake_pip_install)

    result = auto_install.ensure_packages({"fastmcp": FASTMCP_REQUIREMENT})

    assert pip_requests == [FASTMCP_REQUIREMENT]
    assert events == ["pip", "import:fastmcp"]
    assert result == {"fastmcp": "installed"}


def test_constrained_requirement_fails_closed_without_packaging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import builtins
    import scripts.comprehensive_dependency_installer as comprehensive_installer
    import scripts.dependency_installer as dependency_installer

    original_import = builtins.__import__

    def _without_packaging(name, *args, **kwargs):
        if name == "packaging" or name.startswith("packaging."):
            raise ImportError("packaging intentionally unavailable")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _without_packaging)

    helpers = (
        auto_install,
        dependency_installer,
        comprehensive_installer,
    )
    for helper in helpers:
        assert (
            helper._requirement_applies(
                "fastmcp==2.14.7; python_version < '3.10'"
            )
            is False
        )
        assert (
            helper._requirement_applies(
                "fastmcp==2.14.7; python_version >= '3.10'"
            )
            is True
        )
        assert (
            helper._installed_requirement_satisfied(
                "definitely-missing-contract-package==9.9.9; "
                "python_version >= '3.10'"
            )
            is False
        )

    # The pip-vendored parser is a supported fallback. Exercise the true
    # no-parser branch explicitly instead of making this assertion depend on
    # which FastMCP metadata happens to be installed in the test interpreter.
    for helper in helpers:
        monkeypatch.setattr(helper, "_requirement_parser", lambda: None)
        assert helper._requirement_applies(FASTMCP_REQUIREMENT) is True
        assert helper._installed_requirement_satisfied(FASTMCP_REQUIREMENT) is False
        assert helper._installed_requirement_satisfied("fastmcp == 2.14.7") is False
        assert helper._installed_requirement_satisfied("fastmcp >= 2, < 3") is False
        assert (
            helper._installed_requirement_satisfied(
                "demo @ git+https://user:secret@example.invalid/demo.git@main"
            )
            is True
        )


def test_auto_installer_redacts_post_import_failures_and_direct_urls(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    installed_versions = iter(["2.14.6", "2.14.7"])
    monkeypatch.setattr(
        auto_install.importlib.metadata,
        "version",
        lambda distribution: next(installed_versions),
    )
    monkeypatch.setattr(auto_install, "_should_auto_install", lambda: True)
    monkeypatch.setattr(auto_install, "_pip_install", lambda requirement: (True, ""))
    monkeypatch.setattr(
        auto_install.importlib,
        "import_module",
        lambda name: (_ for _ in ()).throw(
            ImportError("token=supersecret&password=alsosecret")
        ),
    )

    result = auto_install.ensure_packages({"fastmcp": FASTMCP_REQUIREMENT})

    status = result["fastmcp"]
    assert status.startswith("failed:post-import:ImportError:receipt=")
    assert "supersecret" not in status
    assert "alsosecret" not in status
    redacted = auto_install._redact_sensitive(
        "demo @ https://user:password@example.invalid/x?access_token=abc123"
    )
    assert "user" not in redacted
    assert "password" not in redacted
    assert "abc123" not in redacted


def test_pip_failure_status_persists_only_exit_code_and_receipt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    secret_output = (
        "Authorization: Bearer supersecret\n"
        "https://example.invalid/package#token=fragmentsecret"
    )
    monkeypatch.setattr(
        auto_install.subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=17,
            stdout=secret_output,
            stderr="password=alsosecret",
        ),
    )

    ok, diagnostic = auto_install._pip_install(
        "demo @ https://user:password@example.invalid/demo#token=packagesecret"
    )

    assert ok is False
    assert diagnostic.startswith("exit-code=17:receipt=")
    for secret in (
        "supersecret",
        "fragmentsecret",
        "alsosecret",
        "packagesecret",
        "password",
    ):
        assert secret not in diagnostic


def test_executable_installers_persist_only_failure_receipts(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    import scripts.comprehensive_dependency_installer as comprehensive_installer
    import scripts.dependency_installer as dependency_installer

    basic = dependency_installer.DependencyInstaller()
    comprehensive = comprehensive_installer.ComprehensiveDependencyInstaller()
    secret_output = (
        "Authorization: Bearer bearer-secret\n"
        "https://example.invalid/package#token=fragment-secret"
    )
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *args, **kwargs: SimpleNamespace(
            returncode=19,
            stdout=secret_output,
            stderr="password=stderr-secret",
        ),
    )
    requirement = (
        "demo @ https://user:url-secret@example.invalid/demo.whl"
        "#token=requirement-secret"
    )

    assert basic.install_package(requirement, "demo") is False
    assert comprehensive.install_package("demo", requirement) is False

    persisted = repr(
        (
            basic.installation_log,
            basic.failed_installations,
            comprehensive.installation_log,
            comprehensive.failed_installations,
            caplog.text,
        )
    )
    assert "exit-code=19:receipt=" in persisted
    for secret in (
        "bearer-secret",
        "fragment-secret",
        "stderr-secret",
        "url-secret",
        "requirement-secret",
    ):
        assert secret not in persisted


class _FakeASGIApp:
    def __init__(self) -> None:
        self.middleware: list[tuple[object, dict[str, object]]] = []

    def add_middleware(self, middleware: object, **kwargs: object) -> None:
        self.middleware.append((middleware, kwargs))

    async def __call__(self, scope: object, receive: object, send: object) -> None:
        return None


class _FakeFastMCP2:
    instances: list["_FakeFastMCP2"] = []

    def __init__(self, name: str) -> None:
        self.name = name
        self.events: list[tuple[str, str]] = []
        self.app = _FakeASGIApp()
        self.http_path: str | None = None
        self.__class__.instances.append(self)

    def tool(self, *, name: str, description: str | None = None):
        def _decorator(function):
            self.events.append(("tool", name))
            return function

        return _decorator

    def resource(self, uri: str, description: str | None = None):
        def _decorator(function):
            self.events.append(("resource", uri))
            return function

        return _decorator

    def prompt(self, *, name: str, description: str | None = None):
        def _decorator(function):
            self.events.append(("prompt", name))
            return function

        return _decorator

    def http_app(self, *, path: str):
        self.http_path = path
        return self.app

    def create_fastapi_app(self, **kwargs: object):
        raise AssertionError("legacy create_fastapi_app must not be used")


def test_native_fastmcp_decorators_populate_legacy_compatibility_views() -> None:
    from ipfs_accelerate_py.mcp.fastmcp_compat import (
        ensure_register_prompt_compat,
        ensure_register_resource_compat,
        ensure_register_tool_compat,
    )

    fake = _FakeFastMCP2("native-views")
    ensure_register_tool_compat(fake)
    ensure_register_resource_compat(fake)
    ensure_register_prompt_compat(fake)

    @fake.tool(name="native.tool", description="native tool")
    def _native_tool(value: str) -> str:
        return value

    @fake.resource("native://resource", description="native resource")
    def _native_resource() -> str:
        return "resource"

    @fake.prompt(name="native.prompt", description="native prompt")
    def _native_prompt(topic: str) -> str:
        return topic

    assert fake.tools["native.tool"]["function"] is _native_tool
    assert fake.tools["native.tool"]["input_schema"]["required"] == ["value"]
    assert fake.resources["native://resource"]["function"] is _native_resource
    assert fake.prompts["native.prompt"]["function"] is _native_prompt


def test_signature_schema_resolves_postponed_boolean_object_and_integer_types() -> None:
    from ipfs_accelerate_py.mcp.fastmcp_compat import function_input_schema

    def _typed_contract(
        enabled: bool,
        options: dict[str, str],
        retries: int,
    ) -> None:
        return None

    schema = function_input_schema(_typed_contract)
    assert schema["properties"]["enabled"] == {"type": "boolean"}
    assert schema["properties"]["options"]["type"] == "object"
    assert schema["properties"]["options"]["additionalProperties"] == {
        "type": "string"
    }
    assert schema["properties"]["retries"] == {"type": "integer"}
    assert schema["required"] == ["enabled", "options", "retries"]


def test_native_schema_is_authoritative_when_legacy_schema_drifts() -> None:
    from ipfs_accelerate_py.mcp.fastmcp_compat import (
        ensure_register_tool_compat,
        get_registration_failures,
        get_schema_drifts,
    )

    class _ProtocolTool:
        parameters = {
            "type": "object",
            "properties": {"count": {"type": "integer"}},
            "required": ["count"],
        }

    class _SchemaFastMCP(_FakeFastMCP2):
        def tool(self, *, name: str, description: str | None = None):
            def _decorator(function):
                self.events.append(("tool", name))
                return _ProtocolTool()

            return _decorator

    fake = _SchemaFastMCP("schema-canonical")
    ensure_register_tool_compat(fake)

    def _tool(count: int) -> int:
        return count

    fake.register_tool(
        name="contract.schema",
        function=_tool,
        input_schema={"type": "object", "properties": {}, "required": []},
    )

    assert fake.tools["contract.schema"]["input_schema"] == _ProtocolTool.parameters
    assert get_registration_failures(fake) == ()
    drifts = get_schema_drifts(fake)
    assert len(drifts) == 1
    assert drifts[0]["name_sha256"] == hashlib.sha256(
        b"contract.schema"
    ).hexdigest()


def test_docker_tool_schemas_retain_required_arguments() -> None:
    from ipfs_accelerate_py.mcp.fastmcp_compat import function_input_schema
    from ipfs_accelerate_py.mcp.tools import docker_tools

    expected_required = {
        "execute_docker_container": {"image"},
        "build_and_execute_github_repo": {"repo_url"},
        "execute_with_payload": {"image", "payload"},
        "stop_container": {"container_id"},
        "pull_docker_image": {"image"},
    }
    for name, required in expected_required.items():
        schema = function_input_schema(docker_tools.MCP_DOCKER_TOOLS[name])
        assert set(schema["required"]) == required
        assert required <= set(schema["properties"])

    execute_schema = function_input_schema(
        docker_tools.MCP_DOCKER_TOOLS["execute_docker_container"]
    )
    assert execute_schema["properties"]["timeout"]["type"] == "integer"
    environment_options = execute_schema["properties"]["environment"]["anyOf"]
    assert {option.get("type") for option in environment_options} == {
        "object",
        "null",
    }


def test_registration_failure_receipts_are_bounded_and_do_not_store_messages() -> None:
    from ipfs_accelerate_py.mcp.fastmcp_compat import (
        _MAX_FAILURE_RECEIPTS,
        _record_failure,
        get_registration_failures,
    )

    fake = SimpleNamespace()
    for index in range(_MAX_FAILURE_RECEIPTS + 25):
        _record_failure(
            fake,
            "tool",
            f"contract.tool.{index}",
            ValueError("secret-bearing implementation detail"),
        )

    failures = get_registration_failures(fake)
    assert len(failures) == _MAX_FAILURE_RECEIPTS
    assert set(failures[0]) == {"kind", "name_sha256", "error_type"}
    assert "secret-bearing" not in repr(failures)


def test_corrupt_registration_failure_ledgers_are_safely_normalized() -> None:
    from ipfs_accelerate_py.mcp.fastmcp_compat import get_registration_failures

    fake = SimpleNamespace(
        _fastmcp_compat_registration_failures=[
            {
                "kind": "tool",
                "name": "secret-bearing-name",
                "error_type": "ValueError",
            }
        ]
    )

    failures = get_registration_failures(fake)

    assert failures == (
        {
            "kind": "adapter",
            "name_sha256": hashlib.sha256(b"failure-ledger-entry").hexdigest(),
            "error_type": "InvalidLedger",
        },
    )
    assert "secret-bearing-name" not in repr(failures)


def test_standalone_app_initializes_with_router_entries_without_paths() -> None:
    standalone = server_module.StandaloneMCP("fastapi-route-contract")
    app = standalone.create_fastapi_app(
        title="FastAPI route contract",
        description="FastAPI route contract",
        version="1",
        docs_url="/docs",
        redoc_url="/redoc",
        mount_path="/mcp",
    )

    assert callable(app)
    assert app.routes


def test_fake_fastmcp_v2_uses_http_app_after_all_compatibility_shims(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeFastMCP2.instances.clear()
    monkeypatch.setattr(server_module, "_ensure_mcp_runtime_dependencies", lambda: {})
    monkeypatch.setattr(
        server_module,
        "_import_fastmcp_v2",
        lambda: (
            SimpleNamespace(__version__="2.14.7", FastMCP=_FakeFastMCP2),
            "",
        ),
    )

    def _register_tools(instance: server_module.IPFSAccelerateMCPServer) -> None:
        assert callable(instance.mcp.register_tool)
        instance.mcp.register_tool(
            name="contract.tool",
            function=lambda: {"ok": True},
        )

    def _register_resources(instance: server_module.IPFSAccelerateMCPServer) -> None:
        assert callable(instance.mcp.register_resource)
        instance.mcp.register_resource(
            uri="contract/resource",
            function=lambda: {"ok": True},
        )

    def _register_prompts(instance: server_module.IPFSAccelerateMCPServer) -> None:
        assert callable(instance.mcp.register_prompt)
        instance.mcp.register_prompt(
            name="contract.prompt",
            template="repair the contract",
        )

    monkeypatch.setattr(
        server_module.IPFSAccelerateMCPServer,
        "_register_tools",
        _register_tools,
    )
    monkeypatch.setattr(
        server_module.IPFSAccelerateMCPServer,
        "_register_resources",
        _register_resources,
    )
    monkeypatch.setattr(
        server_module.IPFSAccelerateMCPServer,
        "_register_prompts",
        _register_prompts,
    )

    server = server_module.IPFSAccelerateMCPServer(mount_path="/contract-mcp")
    server.setup()

    fake = _FakeFastMCP2.instances[-1]
    assert server._using_fastmcp is True
    assert server.mcp is fake
    assert server.fastapi_app is fake.app
    assert fake.http_path == "/contract-mcp"
    assert [event[0] for event in fake.events] == ["tool", "resource", "prompt"]
    assert fake.tools["contract.tool"]["function"]() == {"ok": True}
    assert fake.resources["contract/resource"]["function"]() == {"ok": True}
    assert fake.prompts["contract.prompt"]["template"] == "repair the contract"


def test_missing_fastmcp_v2_http_app_falls_back_without_false_claim(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    class _LegacyFastMCP(_FakeFastMCP2):
        http_app = None

    monkeypatch.setattr(server_module, "_ensure_mcp_runtime_dependencies", lambda: {})
    monkeypatch.setattr(
        server_module,
        "_import_fastmcp_v2",
        lambda: (
            SimpleNamespace(__version__="2.14.7", FastMCP=_LegacyFastMCP),
            "",
        ),
    )
    _disable_registration(monkeypatch)

    server = server_module.IPFSAccelerateMCPServer()
    server.setup()

    assert server._using_fastmcp is False
    assert isinstance(server.mcp, server_module.StandaloneMCP)
    assert server.fastapi_app is not None
    assert "does not expose the v2 http_app API" in caplog.text


def test_fastmcp_wiring_repair_is_attempted_only_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _BrokenRuntime(_FakeFastMCP2):
        constructions = 0
        http_app = None

        def __init__(self, name: str) -> None:
            type(self).constructions += 1
            super().__init__(name)

    runtime_module = SimpleNamespace(
        __version__="2.14.7",
        FastMCP=_BrokenRuntime,
    )
    repair_calls: list[str] = []
    monkeypatch.setattr(server_module, "_ensure_mcp_runtime_dependencies", lambda: {})
    monkeypatch.setattr(
        server_module,
        "_import_fastmcp_v2_with_repair",
        lambda: (runtime_module, ""),
    )

    def _repair_once():
        repair_calls.append("repair")
        return runtime_module, ""

    monkeypatch.setattr(server_module, "_repair_fastmcp_v2_runtime", _repair_once)
    _disable_registration(monkeypatch)

    server = server_module.IPFSAccelerateMCPServer()
    server.setup()

    assert repair_calls == ["repair"]
    assert _BrokenRuntime.constructions == 2
    assert server._using_fastmcp is False
    assert isinstance(server.mcp, server_module.StandaloneMCP)


def test_non_exact_fastmcp_is_rejected_before_transport_claim(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    constructions: list[str] = []

    class _FastMCP3:
        def __init__(self, name: str) -> None:
            constructions.append(name)

    monkeypatch.setattr(server_module, "_ensure_mcp_runtime_dependencies", lambda: {})
    monkeypatch.setattr(
        server_module,
        "_import_fastmcp_v2",
        lambda: (
            SimpleNamespace(__version__="2.14.6", FastMCP=_FastMCP3),
            "",
        ),
    )
    _disable_registration(monkeypatch)

    server = server_module.IPFSAccelerateMCPServer()
    server.setup()

    assert constructions == []
    assert server._using_fastmcp is False
    assert isinstance(server.mcp, server_module.StandaloneMCP)


def test_unsatisfied_transitive_status_prevents_fastmcp_claim(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    imports: list[str] = []

    monkeypatch.setattr(
        server_module,
        "_ensure_mcp_runtime_dependencies",
        lambda: {
            "fastmcp": "ok",
            "fastapi": "ok",
            "uvicorn": "skipped",
            "websockets": "skipped",
        },
    )

    def _forbidden_import():
        imports.append("fastmcp")
        raise AssertionError("FastMCP loader ran with an unsatisfied runtime")

    monkeypatch.setattr(
        server_module,
        "_import_fastmcp_v2_with_repair",
        _forbidden_import,
    )
    _disable_registration(monkeypatch)

    server = server_module.IPFSAccelerateMCPServer()
    server.setup()

    assert imports == []
    assert server._using_fastmcp is False
    assert isinstance(server.mcp, server_module.StandaloneMCP)
    assert "dependency contract is unsatisfied" in caplog.text
    assert "uvicorn" in caplog.text
    assert "websockets" in caplog.text


def test_swallowed_fastmcp_registration_failure_forces_honest_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class _RejectingFastMCP(_FakeFastMCP2):
        def tool(self, *, name: str, description: str | None = None):
            def _decorator(function):
                raise ValueError(f"rejected {name}")

            return _decorator

    monkeypatch.setattr(server_module, "_ensure_mcp_runtime_dependencies", lambda: {})
    monkeypatch.setattr(
        server_module,
        "_import_fastmcp_v2",
        lambda: (
            SimpleNamespace(__version__="2.14.7", FastMCP=_RejectingFastMCP),
            "",
        ),
    )

    def _register_tools(instance: server_module.IPFSAccelerateMCPServer) -> None:
        try:
            instance.mcp.register_tool(name="rejected.tool", function=lambda: None)
        except ValueError:
            # Reproduce a legacy registrar that catches an optional component
            # failure. The adapter ledger must still prevent a false success.
            pass

    monkeypatch.setattr(
        server_module.IPFSAccelerateMCPServer,
        "_register_tools",
        _register_tools,
    )
    monkeypatch.setattr(
        server_module.IPFSAccelerateMCPServer,
        "_register_resources",
        lambda self: None,
    )
    monkeypatch.setattr(
        server_module.IPFSAccelerateMCPServer,
        "_register_prompts",
        lambda self: None,
    )

    server = server_module.IPFSAccelerateMCPServer()
    server.setup()

    candidate = _RejectingFastMCP.instances[-1]
    assert candidate.tools == {}
    assert candidate._fastmcp_compat_registration_failures == [
        {
            "kind": "tool",
            "name_sha256": hashlib.sha256(b"rejected.tool").hexdigest(),
            "error_type": "ValueError",
        }
    ]
    assert server._using_fastmcp is False
    assert isinstance(server.mcp, server_module.StandaloneMCP)
    assert server.fastapi_app is not None


def test_trio_required_registrar_failure_retries_with_standalone_registry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from ipfs_accelerate_py.mcplusplus_module.trio.server import (
        ServerConfig,
        TrioMCPServer,
    )
    from ipfs_accelerate_py.mcp_server import resources as canonical_resources
    from ipfs_accelerate_py.mcp_server import server as canonical_server

    monkeypatch.setattr(server_module, "_ensure_mcp_runtime_dependencies", lambda: {})
    monkeypatch.setattr(
        server_module,
        "_import_fastmcp_v2_with_repair",
        lambda: (
            SimpleNamespace(__version__="2.14.7", FastMCP=_FakeFastMCP2),
            "",
        ),
    )
    attempts: list[str] = []

    def _reject_required_tools(mcp, **kwargs):
        attempts.append(type(mcp).__name__)
        raise RuntimeError("partial required registrar")

    monkeypatch.setattr(canonical_server, "register_all_tools", _reject_required_tools)
    monkeypatch.setattr(canonical_resources, "register_all_resources", lambda mcp: None)

    trio_server = TrioMCPServer(ServerConfig(enable_p2p_tools=False))
    monkeypatch.setattr(trio_server, "_create_fastapi_app", lambda: _FakeASGIApp())
    trio_server.setup()

    assert attempts == ["_FakeFastMCP2", "StandaloneMCP"]
    assert trio_server._using_fastmcp_registry is False
    assert isinstance(trio_server.mcp, canonical_server.StandaloneMCP)
    assert trio_server.fastapi_app is not None


def test_real_fastmcp_v2_http_app_smoke_without_installing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("FastMCP is not installed; fake-v2 coverage remains active")

    version = importlib.metadata.version("fastmcp")
    if version != "2.14.7":
        pytest.skip(f"installed FastMCP is not the supported version ({version})")

    from fastmcp import FastMCP as RealFastMCP

    assert inspect.iscoroutinefunction(RealFastMCP.run_async)
    assert not inspect.iscoroutinefunction(RealFastMCP.run)
    assert "host" not in inspect.signature(RealFastMCP.run_stdio_async).parameters
    assert "port" not in inspect.signature(RealFastMCP.run_stdio_async).parameters
    assert "host" in inspect.signature(RealFastMCP.run_http_async).parameters
    assert "port" in inspect.signature(RealFastMCP.run_http_async).parameters

    monkeypatch.setattr(server_module, "_ensure_mcp_runtime_dependencies", lambda: {})
    _disable_registration(monkeypatch)

    server = server_module.IPFSAccelerateMCPServer(mount_path="/real-v2-smoke")
    server.setup()

    assert server._using_fastmcp is True
    assert callable(server.fastapi_app)
    assert callable(getattr(server.mcp, "register_tool", None))
    assert callable(getattr(server.mcp, "register_resource", None))
    assert callable(getattr(server.mcp, "register_prompt", None))

    def _real_tool(value: str) -> str:
        return value

    def _real_resource() -> str:
        return "resource"

    server.mcp.register_tool(name="real.tool", function=_real_tool)
    server.mcp.register_resource(
        uri="real/resource",
        function=_real_resource,
        description="real resource",
    )
    server.mcp.register_prompt(name="real.prompt", template="real prompt")

    def _real_native(value: str) -> str:
        return value

    server.mcp.tool(name="real.native")(_real_native)

    assert server.mcp.tools["real.tool"]["function"] is _real_tool
    assert server.mcp.tools["real.native"]["function"] is _real_native
    assert server.mcp.resources["real/resource"]["function"] is _real_resource
    assert server.mcp.prompts["real.prompt"]["template"] == "real prompt"


def test_real_fastmcp_full_registry_contract_without_installing() -> None:
    if importlib.util.find_spec("fastmcp") is None:
        pytest.skip("FastMCP is not installed; fake-v2 coverage remains active")
    version = importlib.metadata.version("fastmcp")
    if version != "2.14.7":
        pytest.skip(f"installed FastMCP is not the supported version ({version})")

    import anyio
    from ipfs_accelerate_py.mcp.fastmcp_compat import (
        get_registration_failures,
        get_schema_drifts,
    )

    server = server_module.IPFSAccelerateMCPServer(mount_path="/full-contract")
    server.setup()

    assert server._using_fastmcp is True
    assert get_registration_failures(server.mcp) == ()
    assert get_schema_drifts(server.mcp) == ()

    async def _snapshot():
        return (
            await server.mcp.get_tools(),
            await server.mcp.get_resources(),
            await server.mcp.get_resource_templates(),
            await server.mcp.get_prompts(),
        )

    native_tools, native_resources, native_templates, native_prompts = anyio.run(
        _snapshot
    )
    assert len(server.mcp.tools) == len(native_tools) == 106
    assert len(server.mcp.resources) == 9
    assert len(native_resources) == 8
    assert len(native_templates) == 1
    assert len(server.mcp.prompts) == len(native_prompts) == 1
    assert set(server.mcp.tools) == set(native_tools)
    assert set(server.mcp.prompts) == set(native_prompts)
    for name, tool in native_tools.items():
        assert server.mcp.tools[name]["input_schema"] == tool.parameters

    standalone_server = server_module.IPFSAccelerateMCPServer(
        mount_path="/standalone-full-contract"
    )
    standalone_server._setup_standalone("registry parity audit")
    standalone_server._register_components()
    assert set(standalone_server.mcp.tools) == set(native_tools)
    for name, tool in native_tools.items():
        assert standalone_server.mcp.tools[name]["input_schema"] == tool.parameters

    expected_enums = {
        ("github_list_prs", "state"): {"open", "closed", "merged", "all"},
        ("github_list_issues", "state"): {"open", "closed", "all"},
        ("hardware_test", "accelerator"): {"cuda", "cpu", "all"},
        ("hardware_test", "test_level"): {"basic", "comprehensive"},
        ("hardware_recommend", "task"): {
            "inference",
            "training",
            "fine-tuning",
        },
        ("test_hardware", "accelerator"): {
            "cuda",
            "cpu",
            "webgpu",
            "webnn",
            "all",
        },
        ("test_hardware", "test_level"): {"basic", "comprehensive"},
        ("recommend_hardware", "task"): {
            "inference",
            "training",
            "fine-tuning",
        },
        ("list_hf_inference_models", "model_kind"): {"llm", "embedding"},
        ("build_hf_inference_ipld_document", "model_kind"): {
            "llm",
            "embedding",
        },
        ("get_hf_inference_ipld_cid", "model_kind"): {"llm", "embedding"},
        ("publish_hf_inference_ipld_to_ipfs", "model_kind"): {
            "llm",
            "embedding",
        },
    }
    for (tool_name, property_name), enum_values in expected_enums.items():
        property_schema = native_tools[tool_name].parameters["properties"][
            property_name
        ]
        observed = property_schema.get("enum")
        if observed is None:
            for option in property_schema.get("anyOf", []):
                if isinstance(option, dict) and option.get("enum") is not None:
                    observed = option["enum"]
                    break
        assert set(observed or []) == enum_values

    normalized_compat_resources = {
        str(record.get("normalized_uri", uri))
        for uri, record in server.mcp.resources.items()
    }
    assert normalized_compat_resources == {
        *(str(uri) for uri in native_resources),
        *(str(uri) for uri in native_templates),
    }

    from ipfs_accelerate_py.mcplusplus_module.trio.server import (
        ServerConfig,
        TrioMCPServer,
    )

    trio_native = TrioMCPServer(ServerConfig(enable_p2p_tools=False))
    trio_native.setup()
    trio_standalone = TrioMCPServer(ServerConfig(enable_p2p_tools=False))
    trio_standalone.setup(_force_standalone=True)
    assert trio_native._using_fastmcp_registry is True
    assert trio_standalone._using_fastmcp_registry is False
    assert get_registration_failures(trio_native.mcp) == ()
    assert get_schema_drifts(trio_native.mcp) == ()

    async def _trio_tools():
        return await trio_native.mcp.get_tools()

    native_trio_tools = anyio.run(_trio_tools)
    assert len(native_trio_tools) == len(trio_standalone.mcp.tools) == 5
    assert set(native_trio_tools) == set(trio_standalone.mcp.tools)
    for name, tool in native_trio_tools.items():
        assert trio_standalone.mcp.tools[name]["input_schema"] == tool.parameters
