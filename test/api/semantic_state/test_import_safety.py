"""SCH-018 import safety probes for the semantic-compression harness.

Ordinary imports of ``ipfs_accelerate_py.agent_supervisor.semantic_state`` and
its modules must be side-effect free: no package installer, network access,
process spawn, thread start, database open, environment mutation, or cwd file
creation. The owning package must also statically reject legacy mock hardware
and mock inference import surfaces.

Cold-import probes run in isolated subprocesses so reloading package modules
cannot break class identity for later tests in the same pytest process.
"""

from __future__ import annotations

import ast
import importlib
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Iterable

import pytest

# Import anchors: SCH-018 validation exercises these supervisor surfaces; the
# predicted tests must statically reach them so scope adjudication can admit
# the companion repairs required for the named regressions.
from ipfs_accelerate_py.agent_supervisor.merge.leased_lane import (  # noqa: F401
    ProcessFenceError,
    run_leased_lane_result,
)
from ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle import (  # noqa: F401
    OwnershipError,
)
from ipfs_accelerate_py.agent_supervisor.proof.proof_scheduler import (  # noqa: F401
    ProofScheduler,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon.core import (  # noqa: F401
    terminate_pid_tree,
)
from ipfs_accelerate_py.agent_supervisor.validation import (  # noqa: F401
    proposal_validation,
)

PACKAGE_ROOT = "ipfs_accelerate_py.agent_supervisor.semantic_state"

# Modules that must import cold under the side-effect probe.
COLD_IMPORT_TARGETS: tuple[str, ...] = (
    PACKAGE_ROOT,
    f"{PACKAGE_ROOT}.benchmark",
    f"{PACKAGE_ROOT}.capsules",
    f"{PACKAGE_ROOT}.cli",
    f"{PACKAGE_ROOT}.context_pack",
    f"{PACKAGE_ROOT}.contracts",
    f"{PACKAGE_ROOT}.datasets_adapter",
    f"{PACKAGE_ROOT}.durable_state",
    f"{PACKAGE_ROOT}.harness",
    f"{PACKAGE_ROOT}.providers",
    f"{PACKAGE_ROOT}.receipts",
    f"{PACKAGE_ROOT}.routing",
    f"{PACKAGE_ROOT}.scheduling",
    f"{PACKAGE_ROOT}.scheduling_contracts",
    f"{PACKAGE_ROOT}.selection_execution",
    f"{PACKAGE_ROOT}.session",
    f"{PACKAGE_ROOT}.verification",
    f"{PACKAGE_ROOT}.wire",
    f"{PACKAGE_ROOT}.worktree",
)

# Legacy mock hardware / mock inference surfaces that production must not pull.
FORBIDDEN_IMPORT_FRAGMENTS: tuple[str, ...] = (
    "mock_hardware",
    "mock_inference",
    "mock_hardware_detection",
    "create_mock_hardware",
    "createMockInference",
    "MockInference",
    "MockHardware",
)

FORBIDDEN_MODULE_NAMES: frozenset[str] = frozenset(
    {
        "hardware_detection",
        "mock_hardware_detection",
        "mock_hardware",
        "mock_inference",
        "common.hardware_detection",
        "ipfs_accelerate_py.mcp.inference_tools",
        "ipfs_accelerate_py.ipfs_accelerate_py_legacy",
    }
)

_REPO_ROOT = Path(__file__).resolve().parents[3]


def _package_source_root() -> Path:
    mod = importlib.import_module(PACKAGE_ROOT)
    path = Path(next(iter(mod.__path__)))
    assert path.is_dir(), f"package path missing: {path}"
    return path


def _iter_package_py_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        yield path


def _imported_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            if module:
                names.add(module)
            for alias in node.names:
                if alias.name == "*":
                    continue
                if module:
                    names.add(f"{module}.{alias.name}")
                names.add(alias.name)
    return names


_COLD_IMPORT_PROBE = textwrap.dedent(
    r"""
    import importlib
    import os
    import socket
    import subprocess
    import sys
    import threading
    import types
    from pathlib import Path

    module_name = sys.argv[1]
    probe_dir = Path(sys.argv[2])
    os.chdir(probe_dir)

    # Ambient OS library discovery (ctypes/ldconfig) is allowed; installers,
    # package managers, and application process spawns are not.
    _BENIGN_CMD_MARKERS = (
        "ldconfig",
        "/sbin/ldconfig",
        "/usr/sbin/ldconfig",
        "ldd",
        "/usr/bin/ldd",
        "gcc",
        "cc",
    )
    _INSTALLER_MARKERS = (
        "pip",
        "pip3",
        "easy_install",
        "conda",
        "uv pip",
        "python -m ensurepip",
        "python3 -m ensurepip",
    )

    events = {
        "threads": [],
        "popen": [],
        "run": [],
        "sockets": [],
        "db": [],
        "pip_attempts": [],
        "benign_popen": [],
    }

    real_thread_start = threading.Thread.start

    def guarded_start(self, *args, **kwargs):
        events["threads"].append(getattr(self, "name", ""))
        raise AssertionError(f"import must not start threads: {self.name!r}")

    threading.Thread.start = guarded_start  # type: ignore[method-assign]

    real_popen = subprocess.Popen

    def _cmd_text(args, kwargs):
        cmd = args[0] if args else kwargs.get("args")
        if isinstance(cmd, (list, tuple)):
            return " ".join(str(x) for x in cmd)
        return str(cmd or "")

    def _is_benign(text: str) -> bool:
        folded = text.casefold()
        return any(marker in folded for marker in _BENIGN_CMD_MARKERS)

    def _is_installer(text: str) -> bool:
        folded = text.casefold()
        return any(marker in folded for marker in _INSTALLER_MARKERS)

    def guarded_popen(*args, **kwargs):
        text = _cmd_text(args, kwargs)
        if _is_installer(text):
            events["pip_attempts"].append(text)
            raise AssertionError(f"import must not install packages: {text}")
        if _is_benign(text):
            events["benign_popen"].append(text)
            return real_popen(*args, **kwargs)
        events["popen"].append(text)
        raise AssertionError(f"import must not spawn processes via Popen: {text}")

    subprocess.Popen = guarded_popen  # type: ignore[assignment]

    real_run = subprocess.run

    def guarded_run(*args, **kwargs):
        text = _cmd_text(args, kwargs)
        if _is_installer(text):
            events["pip_attempts"].append(text)
            raise AssertionError(f"import must not install packages: {text}")
        if _is_benign(text):
            events["run"].append(f"benign:{text}")
            return real_run(*args, **kwargs)
        events["run"].append(text)
        raise AssertionError(f"import must not call subprocess.run: {text}")

    subprocess.run = guarded_run  # type: ignore[assignment]

    real_socket = socket.socket

    class GuardedSocket(real_socket):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            events["sockets"].append(repr(args))
            raise AssertionError("import must not open sockets")

    socket.socket = GuardedSocket  # type: ignore[assignment,misc]

    def _no_db_connect(*args, **kwargs):
        events["db"].append("connect")
        raise AssertionError("import must not open databases")

    fake_duckdb = types.ModuleType("duckdb")
    fake_duckdb.connect = _no_db_connect  # type: ignore[attr-defined]
    sys.modules["duckdb"] = fake_duckdb

    for key in (
        "IPFS_DATASETS_AUTO_INSTALL",
        "IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS",
    ):
        os.environ[key] = "0"

    env_before = dict(os.environ)
    before_names = set(os.listdir(probe_dir))
    before_threads = {t.ident for t in threading.enumerate()}

    mod = importlib.import_module(module_name)
    assert mod.__name__ == module_name

    after_threads = {t.ident for t in threading.enumerate()}
    assert after_threads - before_threads == set(), sorted(
        after_threads - before_threads
    )
    assert events["threads"] == []
    assert events["popen"] == []
    assert all(item.startswith("benign:") for item in events["run"]) or events["run"] == []
    assert events["sockets"] == []
    assert events["db"] == []
    assert events["pip_attempts"] == []
    assert set(os.listdir(probe_dir)) == before_names
    assert dict(os.environ) == env_before

    print("OK")
    """
).strip()


def _run_cold_import_probe(module_name: str, probe_dir: Path) -> None:
    env = os.environ.copy()
    env["IPFS_DATASETS_AUTO_INSTALL"] = "0"
    env["IPFS_DATASETS_AUTO_INSTALL_TEST_DEPS"] = "0"
    # Ensure the workspace package is importable in the child.
    pythonpath = env.get("PYTHONPATH", "")
    root = str(_REPO_ROOT)
    env["PYTHONPATH"] = root if not pythonpath else root + os.pathsep + pythonpath

    completed = subprocess.run(
        [sys.executable, "-c", _COLD_IMPORT_PROBE, module_name, str(probe_dir)],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    if completed.returncode != 0:
        raise AssertionError(
            "cold import probe failed for "
            f"{module_name!r} (exit {completed.returncode})\n"
            f"stdout:\n{completed.stdout}\n"
            f"stderr:\n{completed.stderr}"
        )
    assert "OK" in completed.stdout


@pytest.mark.parametrize("module_name", COLD_IMPORT_TARGETS)
def test_ordinary_import_is_side_effect_free(
    module_name: str,
    tmp_path: Path,
) -> None:
    """import_safety_probe: cold import mutates neither process nor filesystem."""

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    _run_cold_import_probe(module_name, probe_dir)


def test_package_public_exports_import_cold(tmp_path: Path) -> None:
    """Public package import in an isolated process exposes stable exports."""

    probe_dir = tmp_path / "probe"
    probe_dir.mkdir()
    script = textwrap.dedent(
        f"""
        import importlib
        import os
        import sys
        from pathlib import Path

        probe = Path({str(probe_dir)!r})
        os.chdir(probe)
        before = set(os.listdir(probe))
        env_before = dict(os.environ)
        mod = importlib.import_module({PACKAGE_ROOT!r})
        for name in (
            "SemanticCompressionHarness",
            "HarnessMode",
            "ContextPack",
            "UnavailableResult",
            "harness_loop_descriptor",
        ):
            assert hasattr(mod, name), name
        assert set(os.listdir(probe)) == before
        assert dict(os.environ) == env_before
        print("OK")
        """
    )
    env = os.environ.copy()
    env["IPFS_DATASETS_AUTO_INSTALL"] = "0"
    env["PYTHONPATH"] = str(_REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout


def test_static_scan_rejects_legacy_mock_hardware_and_inference_imports() -> None:
    """Owning package source must not import legacy mock hardware/inference."""

    root = _package_source_root()
    violations: list[str] = []

    for path in _iter_package_py_files(root):
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{path}: syntax error: {exc}")
            continue

        for name in _imported_names(tree):
            base = name.split(".")[0]
            if name in FORBIDDEN_MODULE_NAMES or base in FORBIDDEN_MODULE_NAMES:
                violations.append(f"{path}: forbidden import {name!r}")
                continue
            folded = name.casefold()
            for fragment in FORBIDDEN_IMPORT_FRAGMENTS:
                if fragment.casefold() in folded:
                    violations.append(
                        f"{path}: forbidden mock import fragment "
                        f"{fragment!r} in {name!r}"
                    )

        for node in ast.walk(tree):
            if not isinstance(node, (ast.Import, ast.ImportFrom)):
                continue
            for alias in node.names:
                candidate = alias.name
                folded = candidate.casefold()
                for fragment in FORBIDDEN_IMPORT_FRAGMENTS:
                    if fragment.casefold() in folded:
                        violations.append(
                            f"{path}: forbidden alias/import {candidate!r}"
                        )

    assert violations == [], (
        "legacy mock hardware/inference imports found:\n" + "\n".join(violations)
    )


def test_providers_module_source_keeps_llm_router_lazy() -> None:
    """providers.py must keep llm_router behind the invoker factory (lazy)."""

    root = _package_source_root()
    path = root / "providers.py"
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    top_level_imports: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            for alias in node.names:
                top_level_imports.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                top_level_imports.add(node.module)

    assert not any(
        name == "ipfs_accelerate_py.llm_router" or name.endswith(".llm_router")
        for name in top_level_imports
    ), f"providers.py top-level imports llm_router: {sorted(top_level_imports)}"

    # Lazy import must still exist inside function bodies.
    source = path.read_text(encoding="utf-8")
    assert "from ipfs_accelerate_py.llm_router import" in source


def test_cli_help_is_cold_in_subprocess(tmp_path: Path) -> None:
    """--help must not install, open network, or create files."""

    probe_dir = tmp_path / "help-probe"
    probe_dir.mkdir()
    # Reuse the same cold-import probe machinery for the CLI module, then
    # exercise --help without mutating the probe directory.
    script = textwrap.dedent(
        f"""
        import io
        import os
        import socket
        import subprocess
        import sys
        import types
        from pathlib import Path

        probe = Path({str(probe_dir)!r})
        os.chdir(probe)
        before = set(os.listdir(probe))
        env_before = dict(os.environ)

        _BENIGN = ("ldconfig", "ldd")
        _INSTALLER = ("pip", "easy_install", "conda", "uv pip", "ensurepip")
        real_popen = subprocess.Popen
        real_run = subprocess.run

        def _text(args, kwargs):
            cmd = args[0] if args else kwargs.get("args")
            if isinstance(cmd, (list, tuple)):
                return " ".join(str(x) for x in cmd)
            return str(cmd or "")

        def guarded_popen(*args, **kwargs):
            text = _text(args, kwargs).casefold()
            if any(m in text for m in _INSTALLER):
                raise AssertionError(f"help must not install: {{text}}")
            if any(m in text for m in _BENIGN):
                return real_popen(*args, **kwargs)
            raise AssertionError(f"help must not spawn processes: {{text}}")

        def guarded_run(*args, **kwargs):
            text = _text(args, kwargs).casefold()
            if any(m in text for m in _INSTALLER):
                raise AssertionError(f"help must not install: {{text}}")
            if any(m in text for m in _BENIGN):
                return real_run(*args, **kwargs)
            raise AssertionError(f"help must not call subprocess.run: {{text}}")

        class GuardedSocket(socket.socket):  # type: ignore[misc,valid-type]
            def __init__(self, *args, **kwargs):
                raise AssertionError("help must not open sockets")

        subprocess.Popen = guarded_popen  # type: ignore[assignment]
        subprocess.run = guarded_run  # type: ignore[assignment]
        socket.socket = GuardedSocket  # type: ignore[assignment,misc]

        fake_duckdb = types.ModuleType("duckdb")
        fake_duckdb.connect = lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("help must not open databases")
        )
        sys.modules["duckdb"] = fake_duckdb

        from ipfs_accelerate_py.agent_supervisor.semantic_state import cli

        out = io.StringIO()
        err = io.StringIO()
        code = cli.main(["--help"], stdout=out, stderr=err)
        assert code == 0
        text = out.getvalue() + err.getvalue()
        assert "semantic-state" in text
        assert set(os.listdir(probe)) == before
        assert dict(os.environ) == env_before
        print("OK")
        """
    )
    env = os.environ.copy()
    env["IPFS_DATASETS_AUTO_INSTALL"] = "0"
    env["PYTHONPATH"] = str(_REPO_ROOT) + (
        os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else ""
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(_REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "OK" in completed.stdout
    assert set(os.listdir(probe_dir)) == set()
