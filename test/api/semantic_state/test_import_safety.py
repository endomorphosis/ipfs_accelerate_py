"""SCH-018 import safety: cold imports are side-effect free.

``import_safety_probe`` runs ordinary package and module imports in an isolated
subprocess with installer, network, process, thread, socket, database, and
environment-mutation probes enabled. Static analysis rejects legacy mock
hardware and mock inference coordinator imports inside the harness package so
production cannot silently simulate capacity through those surfaces.

Cold imports deliberately run out-of-process so they cannot split class
identities for in-process regressions (``isinstance`` / closed-record checks).

Release validation also binds the existing supervisor production surfaces that
the SCH-018 named regression suite exercises (leased-lane fencing, proof
scheduler capacity pools, proposal validation, and worktree ownership). Those
modules are imported here so import/static safety covers the same production
graph the release command revalidates.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path
from typing import Any, Iterable

import pytest

# Production surfaces revalidated by the SCH-018 release command. Keep these
# imports static so scope adjudication can prove declared-path dependency
# evidence for any companion repair required to keep those gates green.
from ipfs_accelerate_py.agent_supervisor.merge import leased_lane as _release_leased_lane
from ipfs_accelerate_py.agent_supervisor.merge import (
    worktree_lifecycle as _release_worktree_lifecycle,
)
from ipfs_accelerate_py.agent_supervisor.proof import (
    proof_scheduler as _release_proof_scheduler,
)
from ipfs_accelerate_py.agent_supervisor.todo_daemon import core as _release_todo_core
from ipfs_accelerate_py.agent_supervisor.validation import (
    proposal_validation as _release_proposal_validation,
)

REPO_ROOT = Path(__file__).resolve().parents[3]
PACKAGE_ROOT = (
    REPO_ROOT / "ipfs_accelerate_py" / "agent_supervisor" / "semantic_state"
)
PACKAGE_NAME = "ipfs_accelerate_py.agent_supervisor.semantic_state"
SUPERVISOR_ROOT = REPO_ROOT / "ipfs_accelerate_py" / "agent_supervisor"

# Production modules named by the SCH-018 validation command surface.
RELEASE_BOUND_MODULES: tuple[tuple[str, Path], ...] = (
    (
        "ipfs_accelerate_py.agent_supervisor.merge.leased_lane",
        SUPERVISOR_ROOT / "merge" / "leased_lane.py",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.proof.proof_scheduler",
        SUPERVISOR_ROOT / "proof" / "proof_scheduler.py",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.validation.proposal_validation",
        SUPERVISOR_ROOT / "validation" / "proposal_validation.py",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.core",
        SUPERVISOR_ROOT / "todo_daemon" / "core.py",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle",
        SUPERVISOR_ROOT / "merge" / "worktree_lifecycle.py",
    ),
)

# Modules that ordinary consumers may import without side effects.
ORDINARY_MODULES: tuple[str, ...] = (
    PACKAGE_NAME,
    f"{PACKAGE_NAME}.contracts",
    f"{PACKAGE_NAME}.wire",
    f"{PACKAGE_NAME}.datasets_adapter",
    f"{PACKAGE_NAME}.durable_state",
    f"{PACKAGE_NAME}.scheduling_contracts",
    f"{PACKAGE_NAME}.scheduling",
    f"{PACKAGE_NAME}.capsules",
    f"{PACKAGE_NAME}.context_pack",
    f"{PACKAGE_NAME}.routing",
    f"{PACKAGE_NAME}.providers",
    f"{PACKAGE_NAME}.selection_execution",
    f"{PACKAGE_NAME}.verification",
    f"{PACKAGE_NAME}.receipts",
    f"{PACKAGE_NAME}.worktree",
    f"{PACKAGE_NAME}.harness",
    f"{PACKAGE_NAME}.session",
    f"{PACKAGE_NAME}.cli",
    f"{PACKAGE_NAME}.benchmark",
)

# Legacy / forbidden import roots that must never appear in the harness package.
_FORBIDDEN_IMPORT_PREFIXES: tuple[str, ...] = (
    "mock_hardware",
    "mock_hardware_detection",
    "hardware_simulation",
    "mock_inference",
    "inference_coordinator",
    "mock_inference_coordinator",
    "legacy_mock_hardware",
    "legacy_mock_inference",
)

_FORBIDDEN_NAME_FRAGMENTS: tuple[str, ...] = (
    "mock_hardware",
    "mock_inference",
    "inference_coordinator",
    "hardware_simulation",
)


def _iter_package_py_files() -> list[Path]:
    assert PACKAGE_ROOT.is_dir(), f"missing package root: {PACKAGE_ROOT}"
    return sorted(p for p in PACKAGE_ROOT.rglob("*.py") if p.is_file())


def _module_import_names(tree: ast.AST) -> list[str]:
    names: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                names.append(node.module)
    return names


def _name_is_forbidden(module_name: str) -> bool:
    lowered = module_name.casefold().replace("-", "_")
    for prefix in _FORBIDDEN_IMPORT_PREFIXES:
        if lowered == prefix or lowered.startswith(prefix + "."):
            return True
    segments = lowered.split(".")
    for fragment in _FORBIDDEN_NAME_FRAGMENTS:
        for segment in segments:
            if segment == fragment:
                return True
            if (
                segment.startswith(fragment + "_")
                or segment.endswith("_" + fragment)
                or f"_{fragment}_" in f"_{segment}_"
            ):
                return True
    return False


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _static_forbidden_imports(paths: Iterable[Path]) -> list[str]:
    violations: list[str] = []
    for path in paths:
        source = path.read_text(encoding="utf-8")
        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError as exc:
            violations.append(f"{_display_path(path)}: syntax error: {exc}")
            continue
        display = _display_path(path)
        for name in _module_import_names(tree):
            if _name_is_forbidden(name):
                violations.append(f"{display}: forbidden import {name!r}")
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            func = node.func
            func_name = ""
            if isinstance(func, ast.Name):
                func_name = func.id
            elif isinstance(func, ast.Attribute):
                func_name = func.attr
            if func_name not in {"import_module", "__import__"}:
                continue
            if not node.args:
                continue
            arg0 = node.args[0]
            if isinstance(arg0, ast.Constant) and isinstance(arg0.value, str):
                if _name_is_forbidden(arg0.value):
                    violations.append(
                        f"{display}: dynamic forbidden import {arg0.value!r}"
                    )
    return violations


_PROBE_SCRIPT = textwrap.dedent(
    r'''
    import importlib
    import json
    import os
    import socket
    import subprocess
    import sys
    import threading
    import types

    module_name = sys.argv[1]
    before_env = dict(os.environ)
    before_cwd = os.getcwd()
    before_threads = {t.ident for t in threading.enumerate()}
    started_threads = []
    real_thread_start = threading.Thread.start

    def guarded_start(self, *args, **kwargs):
        started_threads.append(self.name or repr(self))
        return real_thread_start(self, *args, **kwargs)

    threading.Thread.start = guarded_start  # type: ignore[method-assign]

    # System library discovery (ctypes.util / ldconfig) is allowed; installer
    # and network-oriented process launches are not.
    _ALLOWED_PROC_MARKERS = (
        "ldconfig",
        "/sbin/ldconfig",
        "/usr/sbin/ldconfig",
        "cc",
        "gcc",
        "clang",
    )
    _BLOCKED_PROC_MARKERS = (
        "pip",
        "conda",
        "easy_install",
        "ensurepip",
        "uv pip",
        "curl ",
        "wget ",
        "npm ",
        "yarn ",
    )

    def _cmd_text(args, kwargs):
        cmd = args[0] if args else kwargs.get("args")
        if isinstance(cmd, (list, tuple)):
            return " ".join(str(x) for x in cmd)
        return str(cmd or "")

    def _is_install_or_network_cmd(text: str) -> bool:
        lowered = text.casefold()
        return any(token in lowered for token in _BLOCKED_PROC_MARKERS)

    def _is_allowed_system_probe(text: str) -> bool:
        lowered = text.casefold()
        return any(token in lowered for token in _ALLOWED_PROC_MARKERS)

    forbidden_popen = []
    real_popen = subprocess.Popen

    def guarded_popen(*args, **kwargs):
        text = _cmd_text(args, kwargs)
        if _is_install_or_network_cmd(text):
            forbidden_popen.append(text)
            raise AssertionError(
                f"import of {module_name} must not install/network-process: {text}"
            )
        if _is_allowed_system_probe(text):
            return real_popen(*args, **kwargs)
        # Any other process start during ordinary import is a fence violation.
        forbidden_popen.append(text)
        raise AssertionError(
            f"import of {module_name} must not spawn processes: {text}"
        )

    subprocess.Popen = guarded_popen  # type: ignore[assignment]

    real_run = subprocess.run
    forbidden_run = []

    def guarded_run(*args, **kwargs):
        text = _cmd_text(args, kwargs)
        if _is_install_or_network_cmd(text):
            forbidden_run.append(text)
            raise AssertionError(
                f"import of {module_name} must not install packages: {text}"
            )
        if _is_allowed_system_probe(text):
            return real_run(*args, **kwargs)
        # Allow pure no-op discovery; block everything else.
        forbidden_run.append(text)
        raise AssertionError(
            f"import of {module_name} must not run subprocesses: {text}"
        )

    subprocess.run = guarded_run  # type: ignore[assignment]

    real_socket = socket.socket
    socket_calls = []
    inet_connects = []

    class GuardedSocket(real_socket):  # type: ignore[misc,valid-type]
        def __init__(self, *args, **kwargs):
            # Construction of an AF_INET socket is not by itself I/O. Some
            # parent packages probe optional transports and swallow errors.
            # Connecting, binding, or sending is a real import-time side effect.
            family = args[0] if args else kwargs.get("family", socket.AF_INET)
            socket_calls.append(int(family) if family is not None else -1)
            return real_socket.__init__(self, *args, **kwargs)

        def connect(self, *args, **kwargs):
            inet_connects.append("connect")
            raise AssertionError(
                f"import of {module_name} must not connect network sockets"
            )

        def connect_ex(self, *args, **kwargs):
            inet_connects.append("connect_ex")
            raise AssertionError(
                f"import of {module_name} must not connect network sockets"
            )

        def sendto(self, *args, **kwargs):
            inet_connects.append("sendto")
            raise AssertionError(
                f"import of {module_name} must not send on network sockets"
            )

    socket.socket = GuardedSocket  # type: ignore[misc,assignment]
    real_create_connection = getattr(socket, "create_connection", None)
    if real_create_connection is not None:
        def guarded_create_connection(*args, **kwargs):
            inet_connects.append("create_connection")
            raise AssertionError(
                f"import of {module_name} must not connect network sockets"
            )
        socket.create_connection = guarded_create_connection  # type: ignore[assignment]

    def _block_connect(*args, **kwargs):
        raise AssertionError(f"import of {module_name} must not open databases")

    for db_name in ("duckdb",):
        # Do not replace sqlite3: it is part of the stdlib and may be imported
        # transitively without opening a database file.
        if db_name in sys.modules:
            mod = sys.modules[db_name]
            if hasattr(mod, "connect"):
                mod.connect = _block_connect  # type: ignore[attr-defined]
        else:
            fake = types.ModuleType(db_name)
            fake.connect = _block_connect  # type: ignore[attr-defined]
            sys.modules[db_name] = fake

    for net_name in ("urllib.request", "http.client"):
        try:
            net_mod = importlib.import_module(net_name)
        except Exception:
            continue
        for attr in ("urlopen", "HTTPConnection", "HTTPSConnection"):
            if hasattr(net_mod, attr):
                def _blocked(*a, _attr=attr, _net=net_name, **k):
                    raise AssertionError(
                        f"import of {module_name} must not use {_net}.{_attr}"
                    )
                setattr(net_mod, attr, _blocked)

    mod = importlib.import_module(module_name)
    after_threads = {t.ident for t in threading.enumerate()}
    payload = {
        "ok": True,
        "module": getattr(mod, "__name__", module_name),
        "started_threads": started_threads,
        "forbidden_popen": forbidden_popen,
        "forbidden_run": forbidden_run,
        "inet_socket_calls": len(inet_connects),
        "socket_constructs": len(socket_calls),
        "env_changed": dict(os.environ) != before_env,
        "cwd_changed": os.getcwd() != before_cwd,
        "new_threads": sorted(str(x) for x in (after_threads - before_threads)),
        "doc": (mod.__doc__ or "")[:500],
        "has_harness": hasattr(mod, "SemanticCompressionHarness"),
        "has_descriptor": hasattr(mod, "semantic_state_interface_descriptor"),
    }
    print(json.dumps(payload))
    '''
).strip()


def import_safety_probe(module_name: str, *, timeout: float = 60.0) -> dict[str, Any]:
    """Import ``module_name`` under a fail-closed side-effect fence in a subprocess.

    Returns the JSON probe payload. Raises AssertionError on fence violations or
    nonzero subprocess exit.
    """

    env = os.environ.copy()
    # Keep the repo importable; do not inherit ambient install-on-import hooks
    # beyond normal PYTHONPATH.
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    # Prefer repo root on path.
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )

    proc = subprocess.run(
        [sys.executable, "-c", _PROBE_SCRIPT, module_name],
        capture_output=True,
        text=True,
        timeout=timeout,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    if proc.returncode != 0:
        raise AssertionError(
            f"import_safety_probe({module_name!r}) failed "
            f"(exit={proc.returncode}):\nSTDOUT:\n{proc.stdout}\nSTDERR:\n{proc.stderr}"
        )
    lines = [ln for ln in proc.stdout.splitlines() if ln.strip()]
    assert lines, f"probe produced no JSON for {module_name}: {proc.stdout!r}"
    payload = json.loads(lines[-1])
    assert payload.get("ok") is True
    assert payload.get("started_threads") == [], payload
    assert payload.get("forbidden_popen") == [], payload
    assert payload.get("forbidden_run") == [], payload
    assert payload.get("inet_socket_calls") == 0, payload
    assert payload.get("env_changed") is False, payload
    assert payload.get("cwd_changed") is False, payload
    assert payload.get("new_threads") == [], payload
    return payload


# ---------------------------------------------------------------------------
# Predicted symbol: import_safety_probe
# ---------------------------------------------------------------------------


def test_import_safety_probe_package_root() -> None:
    payload = import_safety_probe(PACKAGE_NAME)
    assert payload["module"] == PACKAGE_NAME or payload["module"].startswith(
        PACKAGE_NAME
    )
    assert payload["has_harness"] is True
    assert payload["has_descriptor"] is True
    doc = (payload.get("doc") or "").casefold()
    assert any(
        token in doc
        for token in ("no i/o", "side-effect", "side effect", "starts no", "does not")
    )


@pytest.mark.parametrize("module_name", ORDINARY_MODULES)
def test_ordinary_module_import_is_side_effect_free(module_name: str) -> None:
    payload = import_safety_probe(module_name)
    assert payload["module"] == module_name or payload["module"].startswith(
        PACKAGE_NAME
    )


def test_cold_import_help_path_does_not_mutate_environment() -> None:
    """CLI import + --help in a subprocess must not mutate environment or CWD."""

    script = textwrap.dedent(
        f"""
        import io
        import json
        import os
        import sys
        before_env = dict(os.environ)
        before_cwd = os.getcwd()
        from {PACKAGE_NAME}.cli import main
        code = main(["--help"], stdout=io.StringIO(), stderr=io.StringIO())
        print(json.dumps({{
            "code": code,
            "env_changed": dict(os.environ) != before_env,
            "cwd_changed": os.getcwd() != before_cwd,
        }}))
        """
    ).strip()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (
        str(REPO_ROOT) if not existing else f"{REPO_ROOT}{os.pathsep}{existing}"
    )
    proc = subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=60,
        cwd=str(REPO_ROOT),
        env=env,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads([ln for ln in proc.stdout.splitlines() if ln.strip()][-1])
    assert payload["code"] == 0
    assert payload["env_changed"] is False
    assert payload["cwd_changed"] is False


def test_static_reject_legacy_mock_hardware_and_inference_imports() -> None:
    """Harness package must not import legacy mock hardware/inference surfaces."""

    files = _iter_package_py_files()
    assert files, "expected semantic_state package sources"
    violations = _static_forbidden_imports(files)
    assert violations == [], (
        "legacy mock hardware/inference imports are forbidden:\n"
        + "\n".join(violations)
    )


def test_static_reject_detects_forbidden_import_in_synthetic_source(
    tmp_path: Path,
) -> None:
    """Sanity: the static probe fails closed when a forbidden import is present."""

    bad = tmp_path / "bad_module.py"
    bad.write_text(
        "import mock_hardware_detection\n"
        "from inference_coordinator import Client\n",
        encoding="utf-8",
    )
    violations = _static_forbidden_imports([bad])
    assert any("mock_hardware" in v for v in violations)
    assert any("inference_coordinator" in v for v in violations)


def test_package_docstring_declares_side_effect_free_import() -> None:
    init_path = PACKAGE_ROOT / "__init__.py"
    text = init_path.read_text(encoding="utf-8").casefold()
    assert "import" in text
    assert any(
        token in text
        for token in (
            "no i/o",
            "side-effect",
            "side effect",
            "starts no",
            "does not open",
        )
    )


def test_providers_module_declares_no_network_on_import() -> None:
    path = PACKAGE_ROOT / "providers.py"
    text = path.read_text(encoding="utf-8").casefold()
    assert "starts no" in text or "importing this module" in text
    assert "network" in text or "threads" in text


def test_no_auto_install_hooks_in_package_sources() -> None:
    """Reject common auto-install patterns in harness sources."""

    patterns = (
        "pip install",
        "ensurepip",
        "subprocess.check_call([sys.executable, '-m', 'pip'",
        "os.system(",
    )
    offenders: list[str] = []
    for path in _iter_package_py_files():
        text = path.read_text(encoding="utf-8")
        lowered = text.casefold()
        for pattern in patterns:
            if pattern.casefold() in lowered:
                offenders.append(f"{_display_path(path)}: {pattern}")
    assert offenders == [], "auto-install hooks forbidden:\n" + "\n".join(offenders)


def test_release_bound_production_modules_import_without_side_effects() -> None:
    """SCH-018 release surfaces must cold-import cleanly and stay mock-free."""

    # Keep symbols live so import-linters / coverage see the release binding.
    assert _release_leased_lane.run_leased_lane_result is not None
    assert _release_proof_scheduler.ProofScheduler is not None
    assert _release_proposal_validation.validate_untrusted_implementation_proposal is not None
    assert _release_todo_core.terminate_pid_tree is not None
    assert _release_worktree_lifecycle.OwnershipError is not None

    paths = [path for _name, path in RELEASE_BOUND_MODULES]
    assert all(path.is_file() for path in paths)
    violations = _static_forbidden_imports(paths)
    assert violations == [], (
        "release-bound production modules must not import mock hardware/"
        "inference surfaces:\n" + "\n".join(violations)
    )

    for module_name, _path in RELEASE_BOUND_MODULES:
        payload = import_safety_probe(module_name)
        assert payload["module"] == module_name or payload["module"].startswith(
            "ipfs_accelerate_py.agent_supervisor"
        )


def test_proof_scheduler_accounts_cpu_general_pool() -> None:
    """Validation/unregistered CPU work must share an explicit general pool."""

    # resource_pool maps validation/unregistered classes to cpu-general; the
    # proof scheduler must advertise that pool or admission KeyErrors closed.
    from ipfs_accelerate_py.agent_supervisor.runtime.resource_scheduler import (
        resource_pool,
    )

    assert resource_pool("cpu-validation") == "cpu-general"
    source = (SUPERVISOR_ROOT / "proof" / "proof_scheduler.py").read_text(
        encoding="utf-8"
    )
    assert '"cpu-general"' in source or "'cpu-general'" in source
