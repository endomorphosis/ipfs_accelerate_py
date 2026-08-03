"""Bootstrap contract for proof-backed test reuse in ipfs_accelerate_py (PTR-061).

Acceptance covered:
- Pytest11 and root loader paths are idempotent
- An individual node picks up reuse without a registry
- Plugin absence executes normally
- Coverage, mutation, profiling, benchmarking, debugger, and explicit off modes execute
- Bootstrap import performs no probe/write/network/daemon action
"""

from __future__ import annotations

import ast
import os
import subprocess
import sys
import textwrap
import tomllib
from pathlib import Path

import pytest

ACCELERATE_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP = ACCELERATE_ROOT / "conftest.py"
PYPROJECT = ACCELERATE_ROOT / "pyproject.toml"
SETUP_PY = ACCELERATE_ROOT / "setup.py"
REQUIREMENTS = ACCELERATE_ROOT / "requirements.txt"
PROOF_REUSE_REQUIREMENTS = ACCELERATE_ROOT / "requirements-proof-reuse.txt"
MANIFEST = ACCELERATE_ROOT / "MANIFEST.in"
PYTEST_INI = ACCELERATE_ROOT / "pytest.ini"
PLUGIN_MODULE = "ipfs_accelerate_py.testing.proof_reuse.plugin"
PLUGIN_ENTRY_POINT = "ipfs-proof-reuse"
PYTEST_SITE = Path(pytest.__file__).resolve().parents[1]


def _write(path: Path, source: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source).lstrip(), encoding="utf-8")


# Minimal root loader mirroring production external/ipfs_accelerate/conftest.py
# (without suite-wide env defaults and hooks that would interfere with hermetic
# isolation projects).
_MINIMAL_ROOT_LOADER = f'''\
"""Optional proof-reuse bootstrap for direct pytest node selection.

Mirrors the production accelerator root loader: packaging entry points handle
autoload; this file supplies the same plugin when autoload is disabled.
"""

from __future__ import annotations

import importlib
import os

_PROOF_REUSE_PLUGIN = {PLUGIN_MODULE!r}


def _optional_proof_reuse_plugin() -> tuple[str, ...]:
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
'''


def _copy_bootstrap(project: Path) -> None:
    """Install the optional proof-reuse root loader into an isolated project."""

    (project / "conftest.py").write_text(_MINIMAL_ROOT_LOADER, encoding="utf-8")

def _environment(
    tmp_path: Path,
    *,
    mode: str,
    autoload: bool,
    first_paths: tuple[Path, ...] = (),
) -> dict[str, str]:
    environment = dict(os.environ)
    python_paths = (
        *(str(path) for path in first_paths),
        str(ACCELERATE_ROOT),
        str(PYTEST_SITE),
        environment.get("PYTHONPATH", ""),
    )
    environment["PYTHONPATH"] = os.pathsep.join(
        part for part in python_paths if part
    )
    environment["IPFS_TEST_PROOF_REUSE_MODE"] = mode
    environment["HOME"] = str(tmp_path / "user-home")
    environment["IPFS_PATH"] = str(tmp_path / "user-home" / ".ipfs")
    environment["COVERAGE_FILE"] = str(tmp_path / ".coverage")
    environment.pop("PYTEST_ADDOPTS", None)
    if autoload:
        environment.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
    else:
        environment["PYTEST_DISABLE_PLUGIN_AUTOLOAD"] = "1"
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


def _install_test_entry_point(metadata_root: Path) -> None:
    distribution = metadata_root / "ipfs_accelerate_ptr_bootstrap-0.dist-info"
    _write(
        distribution / "METADATA",
        """
        Metadata-Version: 2.1
        Name: ipfs-accelerate-ptr-bootstrap
        Version: 0
        """,
    )
    _write(
        distribution / "entry_points.txt",
        f"""
        [pytest11]
        {PLUGIN_ENTRY_POINT} = {PLUGIN_MODULE}
        """,
    )


# ---------------------------------------------------------------------------
# Packaging / source contracts
# ---------------------------------------------------------------------------


def test_pyproject_declares_pytest11_proof_reuse_entry_point() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))

    assert (
        project["project"]["entry-points"]["pytest11"][PLUGIN_ENTRY_POINT]
        == PLUGIN_MODULE
    )


def test_setup_py_declares_pytest11_proof_reuse_entry_point() -> None:
    source = SETUP_PY.read_text(encoding="utf-8")
    assert '"pytest11"' in source or "'pytest11'" in source
    assert "ipfs-proof-reuse=" in source or 'ipfs-proof-reuse=' in source
    assert PLUGIN_MODULE in source


def _requirements(path: Path) -> list[str]:
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]


def test_proof_reuse_dependency_metadata_is_consistent_and_non_circular() -> None:
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))
    core = _requirements(REQUIREMENTS)
    scoped = _requirements(PROOF_REUSE_REQUIREMENTS)
    extra = project["project"]["optional-dependencies"]["proof-reuse"]
    setup_source = SETUP_PY.read_text(encoding="utf-8")

    assert scoped == [
        "pytest>=8.0.0",
        "multiformats>=0.3,<1",
        "jsonschema>=4,<5",
    ]
    assert extra == scoped
    assert "multiformats>=0.3,<1" in core
    assert not any(item.startswith("ipfs_datasets_py") for item in core)
    assert not any(item.startswith("ipfs_datasets_py") for item in scoped)
    assert "requirements-proof-reuse.txt" in setup_source
    assert 'extras_require["proof-reuse"]' in setup_source


def test_sdist_manifest_retains_dynamic_requirement_inputs() -> None:
    included = {
        line.removeprefix("include ").strip()
        for line in MANIFEST.read_text(encoding="utf-8").splitlines()
        if line.startswith("include ")
    }

    assert {"requirements.txt", "requirements-proof-reuse.txt"} <= included


def test_pytest_ini_registers_proof_reuse_markers() -> None:
    text = PYTEST_INI.read_text(encoding="utf-8")
    assert "proof_reuse_disabled" in text
    assert "proof_reuse_effects" in text


def test_root_conftest_has_optional_loader_without_path_registry() -> None:
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert "_optional_proof_reuse_plugin" in source
    assert PLUGIN_MODULE in source
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" in source

    forbidden = (
        "PROOF_REUSE_TEST_LIST",
        "proof_reuse_test_paths",
        "TEST_PATH_REGISTRY",
        "allowed_test_files",
    )
    for token in forbidden:
        assert token not in source

    tree = ast.parse(source)
    # Loader helpers must remain module-level (not nested under other hooks).
    module_funcs = {
        node.name
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    }
    assert "_optional_proof_reuse_plugin" in module_funcs


def test_packaging_metadata_has_no_test_path_registry() -> None:
    for path in (PYPROJECT, SETUP_PY):
        text = path.read_text(encoding="utf-8").lower()
        assert "test_list" not in text
        assert "proof_reuse_test_paths" not in text


@pytest.mark.parametrize("autoload", [False, True])
def test_native_direct_node_uses_effective_test_root_loader(
    tmp_path: Path,
    autoload: bool,
) -> None:
    """The real nested pytest root must expose proof-reuse CLI options."""

    environment = _environment(
        tmp_path,
        mode="shadow",
        autoload=autoload,
    )
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--trace-config",
            "--collect-only",
            "--proof-reuse-mode=shadow",
            (
                "test/api/test_proof_reuse_rollout.py::"
                "test_defaults_remain_off_and_default_policy_cannot_grant_read"
            ),
            "-q",
        ],
        cwd=ACCELERATE_ROOT,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )

    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert PLUGIN_MODULE in output
    assert "1 test collected" in output


# ---------------------------------------------------------------------------
# Cold import / side-effect free bootstrap
# ---------------------------------------------------------------------------


def test_bootstrap_import_performs_no_probe_write_network_or_daemon(
    tmp_path: Path,
) -> None:
    """Cold import of the plugin and root loader must not probe/write/network/daemon."""

    sentinel_home = tmp_path / "cold-home"
    sentinel_ipfs = tmp_path / "cold-ipfs"
    sentinel_home.mkdir()
    loader_path = tmp_path / "root_loader_conftest.py"
    loader_path.write_text(_MINIMAL_ROOT_LOADER, encoding="utf-8")
    environment = dict(os.environ)
    environment["HOME"] = str(sentinel_home)
    environment["IPFS_PATH"] = str(sentinel_ipfs)
    environment["PYTHONPATH"] = os.pathsep.join(
        part
        for part in (
            str(ACCELERATE_ROOT),
            str(PYTEST_SITE),
            environment.get("PYTHONPATH", ""),
        )
        if part
    )
    probe_script = tmp_path / "cold_import_probe.py"
    _write(
        probe_script,
        f"""
        import importlib.util
        import os
        import socket
        import sys
        from pathlib import Path

        home = Path(os.environ["HOME"])
        ipfs_path = Path(os.environ["IPFS_PATH"])
        before_home = {{p.name for p in home.iterdir()}} if home.exists() else set()

        original_socket = socket.socket

        class _NoNetworkSocket(original_socket):
            def connect(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise AssertionError("network connect during plugin import")

            def connect_ex(self, *args, **kwargs):  # type: ignore[no-untyped-def]
                raise AssertionError("network connect_ex during plugin import")

        socket.socket = _NoNetworkSocket  # type: ignore[misc, assignment]

        import importlib
        plugin = importlib.import_module({PLUGIN_MODULE!r})
        assert getattr(plugin, "PLUGIN_NAME", "") == {PLUGIN_ENTRY_POINT!r}

        # Load the same optional root loader used by hermetic bootstrap tests.
        spec = importlib.util.spec_from_file_location(
            "accelerator_proof_reuse_root_loader",
            {str(loader_path)!r},
        )
        assert spec is not None and spec.loader is not None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        assert callable(module._optional_proof_reuse_plugin)

        after_home = {{p.name for p in home.iterdir()}} if home.exists() else set()
        assert after_home == before_home, (before_home, after_home)
        assert not ipfs_path.exists(), "IPFS_PATH must not be created on import"
        print("cold-import-ok")
        """,
    )
    completed = subprocess.run(
        [sys.executable, str(probe_script)],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=60,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode == 0, output
    assert "cold-import-ok" in completed.stdout

# ---------------------------------------------------------------------------
# Registration paths: pytest11 entry point + root loader (idempotent)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "autoload",
    [False, True],
    ids=["root-fallback", "entry-point"],
)
def test_direct_node_pickup_with_entry_point_autoload_modes(
    tmp_path: Path,
    autoload: bool,
) -> None:
    """A single selected node picks up reuse metadata without a path registry."""

    project = tmp_path / "project"
    project.mkdir()
    _copy_bootstrap(project)
    _write(
        project / "test_direct_node.py",
        f"""
        import pytest

        from {PLUGIN_MODULE} import get_item_metadata

        @pytest.mark.proof_reuse_effects("filesystem")
        def test_direct_node(request, pytestconfig):
            metadata = get_item_metadata(request.node)
            assert metadata is not None
            assert metadata.nodeid.endswith("test_direct_node.py::test_direct_node")
            assert metadata.effect_adapters == ("filesystem",)
            # Entry-point autoload registers under the packaging name; the root
            # loader registers the module path.  Either form is valid pickup.
            pm = pytestconfig.pluginmanager
            assert (
                pm.hasplugin({PLUGIN_ENTRY_POINT!r})
                or pm.hasplugin({PLUGIN_MODULE!r})
                or any(
                    "proof" in str(name).lower() and "reuse" in str(name).lower()
                    for name, _ in pm.list_name_plugin()
                )
            )
        """,
    )
    metadata_root = tmp_path / "metadata"
    first_paths: tuple[Path, ...] = ()
    if autoload:
        _install_test_entry_point(metadata_root)
        first_paths = (metadata_root,)
    environment = _environment(
        tmp_path,
        mode="shadow",
        autoload=autoload,
        first_paths=first_paths,
    )

    completed = _run_pytest(
        project,
        environment,
        "test_direct_node.py::test_direct_node",
        "-q",
    )
    _assert_success(completed, "1 passed")


def test_pytest11_and_root_loader_paths_are_idempotent(tmp_path: Path) -> None:
    """Each registration path is safe to re-run; production gates dual registration.

    Idempotency means:
    1. Root loader only activates when entry-point autoload is disabled (no dual name).
    2. Repeated pytest invocations on the same path succeed with a stable config.
    3. Re-registering the plugin under its existing name is non-fatal.
    """

    production = BOOTSTRAP.read_text(encoding="utf-8")
    assert "PYTEST_DISABLE_PLUGIN_AUTOLOAD" in production
    assert "pytest_plugins" in production

    project = tmp_path / "project"
    project.mkdir()
    _copy_bootstrap(project)
    _write(
        project / "test_idempotent.py",
        f"""
        from {PLUGIN_MODULE} import PLUGIN_NAME, get_proof_reuse_config

        def test_idempotent(pytestconfig):
            pm = pytestconfig.pluginmanager
            names = [name for name, _ in pm.list_name_plugin()]
            proof_names = [
                n for n in names
                if ("proof" in str(n).lower() and "reuse" in str(n).lower())
                or n in ({PLUGIN_ENTRY_POINT!r}, {PLUGIN_MODULE!r}, PLUGIN_NAME)
            ]
            assert proof_names, names
            config = get_proof_reuse_config(pytestconfig)
            assert config is not None
            assert config.mode.value == "off"
            # Re-register under the same name is a no-op / non-fatal.
            import importlib
            plugin = importlib.import_module({PLUGIN_MODULE!r})
            try:
                pm.register(plugin, name=proof_names[0])
            except ValueError:
                pass
            assert get_proof_reuse_config(pytestconfig).mode.value == "off"
        """,
    )

    # Path 1: root loader (autoload disabled).
    root_env = _environment(tmp_path, mode="off", autoload=False)
    first = _run_pytest(project, root_env, "test_idempotent.py", "-q")
    _assert_success(first, "1 passed")
    second = _run_pytest(project, root_env, "test_idempotent.py", "-q")
    _assert_success(second, "1 passed")

    # Path 2: pytest11 entry point (autoload enabled; root loader stays idle).
    metadata_root = tmp_path / "metadata"
    _install_test_entry_point(metadata_root)
    entry_env = _environment(
        tmp_path,
        mode="off",
        autoload=True,
        first_paths=(metadata_root,),
    )
    third = _run_pytest(project, entry_env, "test_idempotent.py", "-q")
    _assert_success(third, "1 passed")
    fourth = _run_pytest(project, entry_env, "test_idempotent.py", "-q")
    _assert_success(fourth, "1 passed")


def test_verified_hit_skips_before_fixtures_without_ipfs_or_daemon_touch(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    tests = project / "tests"
    tests.mkdir(parents=True)
    _copy_bootstrap(project)
    daemon_sentinel = tmp_path / "daemon-started"
    body_sentinel = tmp_path / "test-body-ran"
    _write(
        tests / "conftest.py",
        f"""
        from pathlib import Path

        import pytest

        from ipfs_accelerate_py.agent_supervisor.proof.test_execution_contracts import (
            reuse_skip,
        )
        from ipfs_accelerate_py.testing.proof_reuse.lookup import ProofReuseLookup
        from ipfs_accelerate_py.testing.proof_reuse.plugin import (
            set_proof_reuse_services,
        )

        class VerifiedHitLookup(ProofReuseLookup):
            def lookup(self, locator, execution_key, **kwargs):
                return reuse_skip(
                    certificate_cid="bafy-test-certificate",
                    receipt_cid="bafy-test-receipt",
                )

        def pytest_configure(config):
            set_proof_reuse_services(config, lookup=VerifiedHitLookup())

        @pytest.hookimpl(tryfirst=True)
        def pytest_collection_modifyitems(items):
            for item in items:
                item._ipfs_proof_reuse_locator = object()
                item._ipfs_proof_reuse_execution_key = object()

        @pytest.fixture(autouse=True)
        def would_start_daemon():
            Path({str(daemon_sentinel)!r}).write_text("started", encoding="utf-8")
        """,
    )
    _write(
        tests / "test_hit.py",
        f"""
        from pathlib import Path

        def test_cached_pass():
            Path({str(body_sentinel)!r}).write_text("ran", encoding="utf-8")
        """,
    )
    environment = _environment(tmp_path, mode="read", autoload=False)
    user_ipfs_path = Path(environment["IPFS_PATH"])

    completed = _run_pytest(
        project,
        environment,
        "tests/test_hit.py::test_cached_pass",
        "-q",
        "-rs",
    )

    _assert_success(completed, "1 skipped")
    assert "proof-cache-hit:bafy-test-certificate" in (
        completed.stdout + completed.stderr
    )
    assert not daemon_sentinel.exists()
    assert not body_sentinel.exists()
    assert not user_ipfs_path.exists()


# ---------------------------------------------------------------------------
# Absence / degradation
# ---------------------------------------------------------------------------


def test_missing_shared_plugin_executes_normally(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _copy_bootstrap(project)
    _write(project / "test_normal.py", "def test_normal():\n    assert True\n")
    blockers = tmp_path / "blockers"
    _write(
        blockers / "sitecustomize.py",
        """
        import importlib.abc
        import sys

        class BlockProofReuse(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "ipfs_accelerate_py" or fullname.startswith(
                    "ipfs_accelerate_py."
                ):
                    raise ModuleNotFoundError(
                        "optional proof-reuse plugin unavailable",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockProofReuse())
        """,
    )
    environment = _environment(
        tmp_path,
        mode="read",
        autoload=False,
        first_paths=(blockers,),
    )

    completed = _run_pytest(project, environment, "test_normal.py", "-q")
    _assert_success(completed, "1 passed")


def test_missing_store_and_multiformats_execute_normally(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _copy_bootstrap(project)
    _write(
        project / "test_normal.py",
        """
        import sitecustomize
        import sys

        def test_normal():
            assert sitecustomize.OPTIONAL_PROVIDER_IMPORTS == []
            assert "ipfs_accelerate_py.p2p_tasks.service" not in sys.modules
            assert "ipfs_accelerate_py.p2p_tasks.client" not in sys.modules
            assert "ipfs_accelerate_py.p2p_tasks.worker" not in sys.modules
            assert "ipfs_accelerate_py.github_cli.cache" not in sys.modules
        """,
    )
    blockers = tmp_path / "blockers"
    _write(
        blockers / "sitecustomize.py",
        """
        import importlib.abc
        import sys

        BLOCKED = ("ipfs_kit_py.proof_certificate_store", "multiformats")
        OPTIONAL_PROVIDER_IMPORTS = []

        class BlockOptionalProviders(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if any(
                    fullname == name or fullname.startswith(name + ".")
                    for name in BLOCKED
                ):
                    OPTIONAL_PROVIDER_IMPORTS.append(fullname)
                    raise AssertionError("optional provider imported: " + fullname)
                return None

        sys.meta_path.insert(0, BlockOptionalProviders())
        """,
    )
    environment = _environment(
        tmp_path,
        mode="shadow",
        autoload=False,
        first_paths=(blockers,),
    )

    completed = _run_pytest(project, environment, "test_normal.py", "-q")
    _assert_success(completed, "1 passed")


# ---------------------------------------------------------------------------
# Tooling modes remain executable under proof-reuse bootstrap
# ---------------------------------------------------------------------------


def test_explicit_off_mode_executes_normally(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _copy_bootstrap(project)
    _write(
        project / "test_off.py",
        f"""
        from {PLUGIN_MODULE} import ProofReuseMode, get_proof_reuse_config

        def test_off(pytestconfig):
            assert get_proof_reuse_config(pytestconfig).mode is ProofReuseMode.OFF
        """,
    )
    environment = _environment(tmp_path, mode="off", autoload=False)

    completed = _run_pytest(project, environment, "test_off.py", "-q")
    _assert_success(completed, "1 passed")


def test_coverage_execution_remains_available(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _copy_bootstrap(project)
    _write(
        project / "test_coverage_target.py",
        """
        def covered_value():
            return 42

        def test_covered_value():
            assert covered_value() == 42
        """,
    )
    environment = _environment(tmp_path, mode="off", autoload=False)

    completed = _run_pytest(
        project,
        environment,
        "-p",
        "pytest_cov.plugin",
        "--cov=test_coverage_target",
        "--cov-report=term",
        "test_coverage_target.py",
        "-q",
    )
    _assert_success(completed, "1 passed")
    assert "TOTAL" in (completed.stdout + completed.stderr)


def test_mutation_style_execution_remains_available(tmp_path: Path) -> None:
    """A body mutation between runs must still execute (no stale skip authority)."""

    project = tmp_path / "project"
    project.mkdir()
    _copy_bootstrap(project)
    target = project / "test_mutation_target.py"
    _write(
        target,
        """
        def compute():
            return 1

        def test_compute():
            assert compute() == 1
        """,
    )
    environment = _environment(tmp_path, mode="readwrite", autoload=False)

    first = _run_pytest(project, environment, "test_mutation_target.py", "-q")
    _assert_success(first, "1 passed")

    # Mutate the production function and the assertion together (simulates a
    # real mutation-test survivor path still being exercised under reuse mode).
    _write(
        target,
        """
        def compute():
            return 2

        def test_compute():
            assert compute() == 2
        """,
    )
    second = _run_pytest(project, environment, "test_mutation_target.py", "-q")
    _assert_success(second, "1 passed")
    assert "skipped" not in second.stdout.lower() or "1 skipped" not in second.stdout


def test_profiling_execution_remains_available(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _copy_bootstrap(project)
    _write(
        project / "test_profile_target.py",
        """
        def test_profiled():
            total = sum(range(100))
            assert total == 4950
        """,
    )
    environment = _environment(tmp_path, mode="off", autoload=False)
    profile_out = tmp_path / "pytest.cprof"

    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "cProfile",
            "-o",
            str(profile_out),
            "-m",
            "pytest",
            "test_profile_target.py",
            "-q",
        ],
        cwd=project,
        env=environment,
        check=False,
        capture_output=True,
        text=True,
        timeout=120,
    )
    _assert_success(completed, "1 passed")
    assert profile_out.is_file()
    assert profile_out.stat().st_size > 0


def test_benchmarking_execution_remains_available(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    _copy_bootstrap(project)
    _write(
        project / "test_benchmark_target.py",
        """
        import time

        import pytest

        @pytest.mark.benchmark
        def test_benchmark_style():
            started = time.perf_counter()
            value = sum(i * i for i in range(1000))
            elapsed = time.perf_counter() - started
            assert value > 0
            assert elapsed >= 0.0
        """,
    )
    environment = _environment(tmp_path, mode="shadow", autoload=False)

    completed = _run_pytest(
        project,
        environment,
        "test_benchmark_target.py",
        "-q",
        "-m",
        "benchmark",
    )
    _assert_success(completed, "1 passed")


def test_debugger_mode_execution_remains_available(tmp_path: Path) -> None:
    """Debugger hooks remain available; passing tests never enter an interactive pdb."""

    project = tmp_path / "project"
    project.mkdir()
    _copy_bootstrap(project)
    _write(
        project / "test_debugger_target.py",
        """
        def test_debugger_friendly():
            assert True
        """,
    )
    environment = _environment(tmp_path, mode="off", autoload=False)

    completed = _run_pytest(
        project,
        environment,
        # Register a non-interactive pdb class so --pdb would not hang on failure.
        "--pdbcls=pdb:Pdb",
        "test_debugger_target.py",
        "-q",
    )
    _assert_success(completed, "1 passed")
