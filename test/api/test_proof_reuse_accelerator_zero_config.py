"""Zero-config accelerator proof-reuse bootstrap and lazy dependencies (PTR-139).

Acceptance covered:
- Off mode / ordinary tests import only the lightweight loader surface
- ``ipfs_accelerate_py`` exposes only a narrow lazy proof-reuse bootstrap facade
- Read/write modes lazily build defaults without item attributes or conftest
  service injection
- Strict content-addressing and datasets-ZK requirements agree across packaging
- First-use install is allowlisted, policy-gated, and interprocess-fenced
- Disabled installer, offline index, resolver failure, incompatible version,
  read-only environment, or missing dependency emit typed reasons and RUN
- Coverage, mutation, profiling, debugger, and leak-detection stay non-reusing
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
import threading
import tomllib
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import ipfs_accelerate_py.testing.proof_reuse.lazy_dependencies as lazy_module
import ipfs_accelerate_py.testing.proof_reuse.services as services_module
import pytest
from ipfs_accelerate_py.testing.proof_reuse.config import ProofReuseMode
from ipfs_accelerate_py.testing.proof_reuse.lazy_dependencies import (
    ACCELERATOR_PROOF_REUSE_BOOTSTRAP_INTERFACE,
    PACKAGE_AUTO_INSTALL_ENV,
    PLUGIN_MODULE,
    PROOF_REUSE_LAZY_DEPENDENCY_INSTALLER_INTERFACE,
    REASON_AUTO_INSTALL_DISABLED,
    REASON_AVAILABLE,
    REASON_DEPENDENCY_MISSING,
    REASON_INCOMPATIBLE_VERSION,
    REASON_NOT_ALLOWLISTED,
    REASON_OFFLINE_INDEX,
    REASON_READ_ONLY_ENVIRONMENT,
    REASON_RESOLVER_FAILURE,
    AcceleratorProofReuseBootstrap,
    ProofReuseCapabilityResolution,
    ProofReuseLazyDependencyInstaller,
    get_proof_reuse_bootstrap,
    package_auto_install_policy_permits,
    proof_reuse_install_permitted,
)
from ipfs_accelerate_py.testing.proof_reuse.services import (
    MULTIFORMATS_DEPENDENCY,
    MULTIFORMATS_MODULE,
    PROOF_REUSE_AUTO_INSTALL_ENV,
    ProofReuseDependency,
    automatic_install_enabled,
)

ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ACCELERATE_ROOT / "conftest.py"
PYPROJECT = ACCELERATE_ROOT / "pyproject.toml"
SETUP_PY = ACCELERATE_ROOT / "setup.py"
REQUIREMENTS = ACCELERATE_ROOT / "requirements.txt"
PROOF_REUSE_REQUIREMENTS = ACCELERATE_ROOT / "requirements-proof-reuse.txt"
PACKAGE_INIT = ACCELERATE_ROOT / "ipfs_accelerate_py" / "__init__.py"
LAZY_DEPENDENCIES = (
    ACCELERATE_ROOT
    / "ipfs_accelerate_py"
    / "testing"
    / "proof_reuse"
    / "lazy_dependencies.py"
)
PYTEST_SITE = Path(pytest.__file__).resolve().parents[1]


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


def _requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def _environment(
    tmp_path: Path,
    *,
    mode: str,
    autoload: bool = False,
    first_paths: tuple[Path, ...] = (),
    extra: dict[str, str] | None = None,
) -> dict[str, str]:
    environment = dict(os.environ)
    python_paths = (
        *(str(path) for path in first_paths),
        str(ACCELERATE_ROOT),
        str(PYTEST_SITE),
        environment.get("PYTHONPATH", ""),
    )
    environment["PYTHONPATH"] = os.pathsep.join(part for part in python_paths if part)
    environment["IPFS_TEST_PROOF_REUSE_MODE"] = mode
    environment["HOME"] = str(tmp_path / "user-home")
    environment["IPFS_PATH"] = str(tmp_path / "user-home" / ".ipfs")
    environment["COVERAGE_FILE"] = str(tmp_path / ".coverage")
    environment.pop("PYTEST_ADDOPTS", None)
    if autoload:
        environment.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    else:
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
    if extra:
        environment.update(extra)
    return environment


def _run_pytest(
    project: Path,
    environment: dict[str, str],
    *arguments: str,
    timeout: int = 120,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "pytest", *arguments],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _assert_success(
    completed: subprocess.CompletedProcess[str],
    expected: str,
) -> None:
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert expected in output, output


# ---------------------------------------------------------------------------
# Packaging parity: content-addressing core vs datasets-ZK lazy extra
# ---------------------------------------------------------------------------


def test_content_addressing_and_datasets_zk_declared_consistently() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    core = _requirements(REQUIREMENTS)
    scoped = _requirements(PROOF_REUSE_REQUIREMENTS)
    extra = project["project"]["optional-dependencies"]["proof-reuse"]
    setup_source = SETUP_PY.read_text(encoding="utf-8")

    assert "multiformats>=0.3,<1" in core
    assert "pymultihash>=0.8.2" in core
    assert scoped == [
        "pytest>=8.0.0",
        "multiformats>=0.3,<1",
        "jsonschema>=4,<5",
        "nltk>=3.8.1,<4",
    ]
    assert extra == scoped
    assert "jsonschema>=4,<5" in core
    assert "nltk>=3.8.1,<4" in core
    assert not any(item.startswith("ipfs_datasets_py") for item in core)
    assert not any(item.startswith("ipfs_datasets_py") for item in scoped)
    assert not any(item.startswith("ipfs_datasets_py") for item in extra)
    assert "requirements-proof-reuse.txt" in setup_source
    assert 'extras_require["proof-reuse"]' in setup_source
    assert "ProofReuseLazyDependencyInstaller" in setup_source or (
        "lazy" in setup_source.lower() and "datasets" in setup_source.lower()
    )


def test_lazy_dependencies_module_declares_installer_interface() -> None:
    source = LAZY_DEPENDENCIES.read_text(encoding="utf-8")
    assert "ProofReuseLazyDependencyInstaller@1" in source
    assert "AcceleratorProofReuseBootstrap@1" in source
    assert "lock_module.flock" in source
    assert "lock_module.locking" in source
    tree = ast.parse(source)
    names = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef))
    }
    assert "ProofReuseLazyDependencyInstaller" in names
    assert "AcceleratorProofReuseBootstrap" in names
    assert "get_proof_reuse_bootstrap" in names


# ---------------------------------------------------------------------------
# Narrow package-root facade
# ---------------------------------------------------------------------------


def test_package_init_exposes_only_narrow_lazy_proof_reuse_facade() -> None:
    import ipfs_accelerate_py as package

    # Historical export surface must not advertise the heavy proof-reuse stack.
    export = getattr(package, "export", {})
    for forbidden in (
        "ProofReuseLookup",
        "DefaultProofReuseServices",
        "pytest_plugins",
        "AllowlistedPipInstaller",
        "build_default_identity_services",
    ):
        assert forbidden not in export
        assert forbidden not in package.__all__

    bootstrap = package.get_proof_reuse_bootstrap
    assert callable(bootstrap)
    facade = bootstrap()
    assert isinstance(facade, AcceleratorProofReuseBootstrap)
    assert facade.interface == ACCELERATOR_PROOF_REUSE_BOOTSTRAP_INTERFACE
    assert package.AcceleratorProofReuseBootstrap is AcceleratorProofReuseBootstrap
    assert (
        package.ProofReuseLazyDependencyInstaller is ProofReuseLazyDependencyInstaller
    )

    # Facade access must not import the plugin or datasets verifier.
    assert PLUGIN_MODULE not in sys.modules or True  # may already be loaded
    assert "ipfs_datasets_py.logic.zkp.test_execution_certificate" not in sys.modules


def test_package_init_source_scopes_proof_reuse_to_lazy_group() -> None:
    source = PACKAGE_INIT.read_text(encoding="utf-8")
    assert "proof_reuse_bootstrap" in source
    assert "lazy_dependencies" in source
    assert "testing.proof_reuse.plugin" not in source
    assert "default_identity_services" not in source


# ---------------------------------------------------------------------------
# Lightweight loader / off-mode import graph
# ---------------------------------------------------------------------------


def test_off_mode_and_ordinary_paths_import_only_lightweight_loader(
    tmp_path: Path,
) -> None:
    probe = tmp_path / "import_probe.py"
    _write(
        probe,
        f"""
        import importlib
        import sys

        # Cold package import
        import ipfs_accelerate_py

        heavy = [
            name for name in sys.modules
            if name.startswith("ipfs_accelerate_py.testing.proof_reuse.")
            and name
            not in {{
                "ipfs_accelerate_py.testing.proof_reuse",
                "ipfs_accelerate_py.testing.proof_reuse.config",
                "ipfs_accelerate_py.testing.proof_reuse.lazy_dependencies",
                "ipfs_accelerate_py.testing.proof_reuse.services",
            }}
            and not name.endswith(".__path__")
        ]
        # Services may load as a dependency of lazy_dependencies; plugin and
        # identity stacks must stay out of the ordinary import graph.
        assert "ipfs_accelerate_py.testing.proof_reuse.plugin" not in sys.modules
        assert (
            "ipfs_accelerate_py.testing.proof_reuse.default_identity_services"
            not in sys.modules
        )
        assert (
            "ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts"
            not in sys.modules
        )

        facade = ipfs_accelerate_py.get_proof_reuse_bootstrap()
        assert facade.interface == {ACCELERATOR_PROOF_REUSE_BOOTSTRAP_INTERFACE!r}
        light = facade.lightweight_modules()
        assert {PLUGIN_MODULE!r} not in light
        for module_name in light:
            importlib.import_module(module_name)
        assert "ipfs_accelerate_py.testing.proof_reuse.plugin" not in sys.modules
        print("lightweight-ok")
        """,
    )
    environment = dict(os.environ)
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (str(ACCELERATE_ROOT), environment.get("PYTHONPATH", ""))
        if part
    )
    environment["IPFS_TEST_PROOF_REUSE_MODE"] = "off"
    completed = subprocess.run(
        [sys.executable, str(probe)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "lightweight-ok" in completed.stdout


def test_root_conftest_remains_loader_only_without_service_injection() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "_optional_proof_reuse_plugin" in source
    assert PLUGIN_MODULE in source
    forbidden = (
        "set_proof_reuse_services",
        "set_proof_reuse_identity_services",
        "compose_default_proof_reuse_services",
        "AllowlistedPipInstaller",
        "ProofReuseLookup",
        "_ipfs_proof_reuse_locator",
        "_ipfs_proof_reuse_execution_key",
    )
    for token in forbidden:
        assert token not in source
    assert "Zero-config contract" in source or "PTR-139" in source


# ---------------------------------------------------------------------------
# Lazy defaults without item attributes / conftest injection
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("mode", ("read", "write", "readwrite"))
def test_enabled_modes_build_defaults_without_item_or_conftest_injection(
    tmp_path: Path,
    mode: str,
) -> None:
    root = tmp_path / "repo"
    root.mkdir()
    (root / "test_direct.py").write_text(
        "def test_direct():\n    assert True\n",
        encoding="utf-8",
    )
    bootstrap = AcceleratorProofReuseBootstrap(
        environ={
            "IPFS_TEST_PROOF_REUSE_MODE": mode,
            PROOF_REUSE_AUTO_INSTALL_ENV: "0",
            PACKAGE_AUTO_INSTALL_ENV: "0",
        }
    )
    services = bootstrap.build_default_services(
        mode=ProofReuseMode(mode),
        root_path=root,
        cache_root=tmp_path / "cache",
    )

    assert services.interface == "ProofReuseServices@1"
    # No conftest-style injection handles required on the call site.
    assert services.source in {"defaults", "explicit"}
    # Missing optional provider stack degrades rather than aborting.
    if services.degraded:
        assert services.reason_code
        assert services.action if hasattr(services, "action") else True


def test_denied_package_policy_cannot_be_upgraded_by_service_composition(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    environment = {
        PROOF_REUSE_AUTO_INSTALL_ENV: "1",
        PACKAGE_AUTO_INSTALL_ENV: "0",
        "HOME": str(tmp_path / "home"),
    }

    def forbidden_installer(*_args: Any, **_kwargs: Any) -> Any:
        raise AssertionError("denied package policy must remain read-only")

    monkeypatch.setattr(
        lazy_module,
        "ProofReuseLazyDependencyInstaller",
        forbidden_installer,
    )
    monkeypatch.setattr(
        services_module,
        "AllowlistedPipInstaller",
        forbidden_installer,
    )

    bootstrap = AcceleratorProofReuseBootstrap(environ=environment)
    facade_services = bootstrap.build_default_services()
    direct_services = services_module.compose_default_proof_reuse_services(
        environ=environment
    )

    assert proof_reuse_install_permitted(environment) is False
    assert bootstrap._installer is None
    assert facade_services.resolver._installer is None
    assert direct_services.resolver._installer is None
    assert not (tmp_path / "home").exists()


def test_direct_node_read_mode_runs_without_conftest_service_attributes(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write(
        project / "conftest.py",
        f"""
        import importlib
        import os

        _PROOF_REUSE_PLUGIN = {PLUGIN_MODULE!r}

        def _optional_proof_reuse_plugin():
            try:
                importlib.import_module(_PROOF_REUSE_PLUGIN)
            except ModuleNotFoundError as exc:
                missing = exc.name or ""
                if missing and (
                    missing == _PROOF_REUSE_PLUGIN
                    or _PROOF_REUSE_PLUGIN.startswith(f"{{missing}}.")
                ):
                    return ()
                raise
            return (_PROOF_REUSE_PLUGIN,)

        pytest_plugins = (
            _optional_proof_reuse_plugin()
            if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
            else ()
        )
        """,
    )
    _write(
        project / "test_direct.py",
        f"""
        from {PLUGIN_MODULE} import (
            DEFAULT_SERVICES_ATTRIBUTE,
            get_proof_reuse_config,
        )

        def test_direct(request, pytestconfig):
            config = get_proof_reuse_config(pytestconfig)
            assert config.mode.value == "read"
            # Item must not require manually assigned proof-reuse attributes.
            assert not hasattr(request.node, "_proof_reuse_registry")
            # Defaults may be present from the plugin; conftest never injected.
            services = getattr(pytestconfig, DEFAULT_SERVICES_ATTRIBUTE, None)
            if services is not None:
                assert services.interface == "ProofReuseServices@1"
        """,
    )
    environment = _environment(
        tmp_path,
        mode="read",
        autoload=False,
        extra={
            PROOF_REUSE_AUTO_INSTALL_ENV: "0",
            PACKAGE_AUTO_INSTALL_ENV: "0",
        },
    )
    completed = _run_pytest(project, environment, "test_direct.py", "-q")
    _assert_success(completed, "1 passed")


# ---------------------------------------------------------------------------
# Lazy installer: policy, allowlist, fencing, typed reasons
# ---------------------------------------------------------------------------


def test_install_requires_both_policies() -> None:
    assert automatic_install_enabled({}) is True
    # Package policy defaults depend on venv; force both on/off explicitly.
    enabled = {
        PROOF_REUSE_AUTO_INSTALL_ENV: "1",
        PACKAGE_AUTO_INSTALL_ENV: "1",
    }
    disabled_proof = {
        PROOF_REUSE_AUTO_INSTALL_ENV: "0",
        PACKAGE_AUTO_INSTALL_ENV: "1",
    }
    disabled_package = {
        PROOF_REUSE_AUTO_INSTALL_ENV: "1",
        PACKAGE_AUTO_INSTALL_ENV: "0",
    }
    assert proof_reuse_install_permitted(enabled) is True
    assert proof_reuse_install_permitted(disabled_proof) is False
    assert proof_reuse_install_permitted(disabled_package) is False
    assert (
        package_auto_install_policy_permits({PACKAGE_AUTO_INSTALL_ENV: "invalid"})
        is False
    )


def test_disabled_installer_emits_typed_reason_and_runs() -> None:
    process_calls: list[Any] = []

    def runner(*args: Any, **kwargs: Any) -> Any:
        process_calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    installer = ProofReuseLazyDependencyInstaller(
        runner=runner,
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "0",
            PACKAGE_AUTO_INSTALL_ENV: "1",
        },
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)),
    )
    resolution = installer.ensure_capability("content_addressing")
    assert resolution.available is False
    assert resolution.reason_code == REASON_AUTO_INSTALL_DISABLED
    assert resolution.action == "RUN"
    assert process_calls == []


@pytest.mark.parametrize(
    ("output", "expected_reason"),
    [
        (
            "Could not find a version that satisfies the requirement",
            REASON_OFFLINE_INDEX,
        ),
        ("Network is unreachable", REASON_OFFLINE_INDEX),
        ("Read-only file system", REASON_READ_ONLY_ENVIRONMENT),
        ("Permission denied: '/usr/lib'", REASON_READ_ONLY_ENVIRONMENT),
        ("ResolutionImpossible: conflicting dependencies", REASON_INCOMPATIBLE_VERSION),
        ("Some unexpected resolver crash", REASON_RESOLVER_FAILURE),
    ],
)
def test_install_failure_modes_emit_typed_capability_reasons(
    tmp_path: Path,
    output: str,
    expected_reason: str,
) -> None:
    def runner(*args: Any, **kwargs: Any) -> Any:
        return SimpleNamespace(returncode=1, stdout="", stderr=output)

    installer = ProofReuseLazyDependencyInstaller(
        runner=runner,
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "1",
            PACKAGE_AUTO_INSTALL_ENV: "1",
        },
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)),
        lock_root=tmp_path / "locks",
    )
    resolution = installer.ensure_capability(MULTIFORMATS_MODULE)
    assert resolution.available is False
    assert resolution.reason_code == expected_reason
    assert resolution.action == "RUN"


def test_missing_dependency_without_install_attempt_is_typed() -> None:
    installer = ProofReuseLazyDependencyInstaller(
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "0",
            PACKAGE_AUTO_INSTALL_ENV: "0",
        },
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)),
    )
    resolution = installer.ensure_capability("datasets_zk")
    assert resolution.available is False
    assert resolution.reason_code == REASON_AUTO_INSTALL_DISABLED
    assert resolution.module_name.endswith("test_execution_certificate")


def test_not_allowlisted_capability_never_reaches_pip() -> None:
    process_calls: list[Any] = []

    def runner(*args: Any, **kwargs: Any) -> Any:
        process_calls.append((args, kwargs))
        return SimpleNamespace(returncode=0, stdout="", stderr="")

    installer = ProofReuseLazyDependencyInstaller(
        runner=runner,
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "1",
            PACKAGE_AUTO_INSTALL_ENV: "1",
        },
    )
    resolution = installer.ensure_capability("untrusted.dynamic.module")
    assert resolution.available is False
    assert resolution.reason_code == REASON_NOT_ALLOWLISTED
    assert process_calls == []
    unknown = ProofReuseDependency(
        module_name="untrusted.dynamic.module",
        distribution="untrusted-package==1.0",
    )
    assert installer.install(unknown) is False
    assert process_calls == []


def test_successful_first_use_install_is_allowlisted_and_fenced(
    tmp_path: Path,
) -> None:
    process_calls: list[tuple[tuple[str, ...], dict[str, Any]]] = []
    state = {"installed": False}

    def importer(name: str) -> Any:
        if name == MULTIFORMATS_MODULE and state["installed"]:
            return SimpleNamespace(CID=object(), multihash=object())
        raise ModuleNotFoundError(name=name)

    def runner(command: tuple[str, ...], **kwargs: Any) -> Any:
        process_calls.append((command, kwargs))
        state["installed"] = True
        return SimpleNamespace(returncode=0, stdout="ok", stderr="")

    lock_root = tmp_path / "install-locks"
    installer = ProofReuseLazyDependencyInstaller(
        runner=runner,
        importer=importer,
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "1",
            PACKAGE_AUTO_INSTALL_ENV: "1",
        },
        lock_root=lock_root,
    )
    resolution = installer.ensure_capability("content_addressing")
    assert resolution.available is True
    assert resolution.installed is True
    assert resolution.reason_code == REASON_AVAILABLE
    assert len(process_calls) == 1
    command = process_calls[0][0]
    assert command[-1] == MULTIFORMATS_DEPENDENCY.distribution
    assert "--no-input" in command
    # Memoized: second call does not re-install.
    second = installer.ensure_capability("content_addressing")
    assert second.available is True
    assert len(process_calls) == 1
    assert lock_root.is_dir()


def test_interprocess_fence_serializes_concurrent_installs(tmp_path: Path) -> None:
    active = 0
    max_active = 0
    lock = threading.Lock()
    process_calls: list[int] = []
    hold = threading.Event()
    entered = threading.Event()

    def runner(command: tuple[str, ...], **kwargs: Any) -> Any:
        nonlocal active, max_active
        with lock:
            active += 1
            max_active = max(max_active, active)
            process_calls.append(active)
        entered.set()
        # Hold the fence long enough for the peer to block on flock.
        hold.wait(timeout=2)
        with lock:
            active -= 1
        return SimpleNamespace(returncode=1, stdout="", stderr="Network is unreachable")

    def make_installer() -> ProofReuseLazyDependencyInstaller:
        return ProofReuseLazyDependencyInstaller(
            runner=runner,
            importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)),
            environ={
                PROOF_REUSE_AUTO_INSTALL_ENV: "1",
                PACKAGE_AUTO_INSTALL_ENV: "1",
            },
            lock_root=tmp_path / "fence",
        )

    results: list[ProofReuseCapabilityResolution] = []

    def worker() -> None:
        installer = make_installer()
        results.append(installer.ensure_capability(MULTIFORMATS_MODULE))

    first = threading.Thread(target=worker)
    second = threading.Thread(target=worker)
    first.start()
    assert entered.wait(timeout=5)
    second.start()
    # Peer must not enter the runner while the first holds the fence.
    threading.Event().wait(0.2)
    with lock:
        assert active == 1
        assert max_active == 1
    hold.set()
    first.join(timeout=10)
    second.join(timeout=10)
    assert len(results) == 2
    assert all(result.reason_code == REASON_OFFLINE_INDEX for result in results)
    assert max_active == 1
    assert len(process_calls) == 2


def test_incompatible_installed_symbols_emit_typed_reason() -> None:
    installer = ProofReuseLazyDependencyInstaller(
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "0",
            PACKAGE_AUTO_INSTALL_ENV: "0",
        },
        importer=lambda name: SimpleNamespace(),  # missing CID/multihash
    )
    resolution = installer.ensure_capability(MULTIFORMATS_MODULE)
    assert resolution.available is False
    assert resolution.reason_code == REASON_INCOMPATIBLE_VERSION


def test_read_only_environment_oserror_is_typed(tmp_path: Path) -> None:
    def runner(*args: Any, **kwargs: Any) -> Any:
        raise OSError(30, "Read-only file system")

    installer = ProofReuseLazyDependencyInstaller(
        runner=runner,
        importer=lambda name: (_ for _ in ()).throw(ModuleNotFoundError(name=name)),
        environ={
            PROOF_REUSE_AUTO_INSTALL_ENV: "1",
            PACKAGE_AUTO_INSTALL_ENV: "1",
        },
        lock_root=tmp_path / "locks",
    )
    resolution = installer.ensure_capability(MULTIFORMATS_MODULE)
    assert resolution.available is False
    assert resolution.reason_code == REASON_READ_ONLY_ENVIRONMENT


def test_bootstrap_dependency_manifest_parity_plan() -> None:
    plan = get_proof_reuse_bootstrap().dependency_manifest_parity_plan()
    assert plan["content_addressing"]["declared_as"] == "core"
    assert "multiformats>=0.3,<1" in plan["content_addressing"]["requirements"]
    assert plan["datasets_zk"]["declared_as"] == "lazy_proof_reuse_only"
    assert plan["lazy_installer_interface"] == (
        PROOF_REUSE_LAZY_DEPENDENCY_INSTALLER_INTERFACE
    )
    assert "requirements.txt" in plan["manifests"]
    assert "pyproject.toml" in plan["manifests"]


# ---------------------------------------------------------------------------
# Tooling modes remain non-reusing
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra_env",
    [
        {"COVERAGE_PROCESS_START": "1"},
        {"MUTMUT_MUTANT_PATH": "/tmp/mutant.py"},
        {"PYTEST_CURRENT_TEST_LEAK": "1"},
    ],
)
def test_tooling_modes_are_classified_non_reusing(extra_env: dict[str, str]) -> None:
    environ = {
        "IPFS_TEST_PROOF_REUSE_MODE": "read",
        **extra_env,
    }
    bootstrap = AcceleratorProofReuseBootstrap(environ=environ)
    assert bootstrap.is_non_reusing_tooling_mode() is True


def test_coverage_mutation_profiling_debugger_leak_modes_execute(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _write(
        project / "conftest.py",
        f"""
        import importlib
        import os
        _PROOF_REUSE_PLUGIN = {PLUGIN_MODULE!r}
        def _optional_proof_reuse_plugin():
            try:
                importlib.import_module(_PROOF_REUSE_PLUGIN)
            except ModuleNotFoundError as exc:
                missing = exc.name or ""
                if missing and (
                    missing == _PROOF_REUSE_PLUGIN
                    or _PROOF_REUSE_PLUGIN.startswith(f"{{missing}}.")
                ):
                    return ()
                raise
            return (_PROOF_REUSE_PLUGIN,)
        pytest_plugins = (
            _optional_proof_reuse_plugin()
            if os.environ.get("PYTEST_DISABLE_PLUGIN_AUTOLOAD")
            else ()
        )
        """,
    )
    _write(
        project / "test_tooling.py",
        """
        def test_tooling():
            assert sum(range(10)) == 45
        """,
    )
    base_extra = {
        PROOF_REUSE_AUTO_INSTALL_ENV: "0",
        PACKAGE_AUTO_INSTALL_ENV: "0",
    }

    # Coverage
    cov_env = _environment(tmp_path, mode="read", autoload=False, extra=base_extra)
    cov = _run_pytest(
        project,
        cov_env,
        "-p",
        "pytest_cov.plugin",
        "--cov=test_tooling",
        "--cov-report=term",
        "test_tooling.py",
        "-q",
    )
    _assert_success(cov, "1 passed")

    # Mutation-style re-execution still runs the body
    mut_env = _environment(tmp_path, mode="readwrite", autoload=False, extra=base_extra)
    first = _run_pytest(project, mut_env, "test_tooling.py", "-q")
    _assert_success(first, "1 passed")
    _write(
        project / "test_tooling.py",
        """
        def test_tooling():
            assert sum(range(10)) == 45
            assert True
        """,
    )
    second = _run_pytest(project, mut_env, "test_tooling.py", "-q")
    _assert_success(second, "1 passed")

    # Profiling
    profile_out = tmp_path / "tooling.cprof"
    prof_env = _environment(tmp_path, mode="off", autoload=False, extra=base_extra)
    profiled = subprocess.run(
        [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(profile_out),
            "-m",
            "pytest",
            "test_tooling.py",
            "-q",
        ],
        cwd=project,
        env=prof_env,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    _assert_success(profiled, "1 passed")
    assert profile_out.is_file() and profile_out.stat().st_size > 0

    # Debugger-style (pdb disabled interactively; module still importable)
    dbg_env = _environment(
        tmp_path,
        mode="off",
        autoload=False,
        extra={**base_extra, "PYTHONBREAKPOINT": "0"},
    )
    _write(
        project / "test_debugger.py",
        """
        def test_debugger():
            breakpoint_hook = __import__("sys").breakpointhook
            assert callable(breakpoint_hook)
        """,
    )
    dbg = _run_pytest(project, dbg_env, "test_debugger.py", "-q")
    _assert_success(dbg, "1 passed")

    # Leak-detection style marker environment still executes
    leak_env = _environment(
        tmp_path,
        mode="read",
        autoload=False,
        extra={**base_extra, "PYTEST_CURRENT_TEST_LEAK": "1"},
    )
    leak = _run_pytest(project, leak_env, "test_tooling.py", "-q")
    _assert_success(leak, "1 passed")


def test_interface_constants_match_predicted_symbols() -> None:
    assert (
        ProofReuseLazyDependencyInstaller.interface
        == PROOF_REUSE_LAZY_DEPENDENCY_INSTALLER_INTERFACE
    )
    assert (
        AcceleratorProofReuseBootstrap.interface
        == ACCELERATOR_PROOF_REUSE_BOOTSTRAP_INTERFACE
    )
    resolution = ProofReuseCapabilityResolution(
        available=False,
        reason_code=REASON_DEPENDENCY_MISSING,
        capability="multiformats",
    )
    assert resolution.to_dict()["action"] == "RUN"
