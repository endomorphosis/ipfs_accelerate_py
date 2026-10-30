"""Test API fixtures and path setup."""

from __future__ import annotations

import os
import stat
import sys
from pathlib import Path

import pytest

repo_root = Path(__file__).resolve().parents[2]
repo_root_text = str(repo_root)

# Pytest may prepend this checkout's parent while importing ``test.api`` as a
# package.  That parent can contain sibling checkouts with the same top-level
# package names (notably ``ipfs_datasets_py``), so merely checking whether the
# local root is present leaves the sibling ahead of the nested submodule.
# Re-promote the local checkout deterministically before API tests are imported.
sys.path[:] = [entry for entry in sys.path if entry != repo_root_text]
sys.path.insert(0, repo_root_text)

def _pin_local_datasets_package(checkout_root: Path) -> None:
    """Pin an initialized local checkout without importing an optional global install.

    Native implementation worktrees may contain an uninitialized datasets
    gitlink. Importing its name in that case selects a different, installed
    distribution and makes every API test depend on its optional cloud/ML
    stack. Tests which need datasets still import it explicitly and retain
    their ordinary dependency failures.
    """
    import importlib.util

    candidates = (
        checkout_root / "ipfs_datasets_py" / "ipfs_datasets_py",
        checkout_root / "ipfs_datasets_py",
    )
    package = next((path for path in candidates if (path / "__init__.py").is_file()), None)
    if package is None:
        return
    spec = importlib.util.spec_from_file_location(
        "ipfs_datasets_py", package / "__init__.py",
        submodule_search_locations=[str(package)],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load local datasets package: {package}")
    module = importlib.util.module_from_spec(spec)
    previous = sys.modules.get(spec.name)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if previous is None:
            sys.modules.pop(spec.name, None)
        else:
            sys.modules[spec.name] = previous
        raise


# Later imports may add sibling roots to sys.path. Pin only actual local code.
_pin_local_datasets_package(repo_root)

# Hermetic validation sets PYTEST_DISABLE_PLUGIN_AUTOLOAD, so pytest11 entry
# points do not load. conftest still does. Load the board pytest ledger from
# this checkout's files, not ``sys.modules['ipfs_accelerate_py']`` — the
# sealed owner capsule is often an older generation and cannot write outside
# the landlocked worktree.
def _load_aseh_board_pytest_plugin():
    import importlib.util
    import types

    runtime = repo_root / "ipfs_accelerate_py" / "agent_supervisor" / "runtime"
    pkg_name = "aseh_board_pytest_reuse"
    pkg = types.ModuleType(pkg_name)
    pkg.__path__ = [str(runtime)]
    pkg.__package__ = pkg_name
    sys.modules[pkg_name] = pkg
    loaded = None
    for name in (
        "pytest_item_ledger",
        "board_pytest_selection",
        "pytest_item_ledger_plugin",
    ):
        path = runtime / f"{name}.py"
        spec = importlib.util.spec_from_file_location(
            f"{pkg_name}.{name}",
            path,
            submodule_search_locations=[str(runtime)],
        )
        if spec is None or spec.loader is None:
            raise ImportError(f"cannot load {path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        setattr(pkg, name, module)
        loaded = module
    return loaded


_R45_EVIDENCE_NAME = (
    "bootstrap-repair-historical-live-evidence-revision-closure-transition.json"
)


def _seed_r45_receipt_evidence() -> None:
    """Hermetic 061 worktrees do not have gitignored ``data/aseh`` evidence."""

    dest = (
        repo_root
        / "data"
        / "aseh"
        / "evidence"
        / "bootstrap"
        / _R45_EVIDENCE_NAME
    )
    try:
        if dest.is_file():
            dest.chmod(0o600)
            return
    except OSError:
        pass
    fixture = (
        repo_root
        / "test"
        / "api"
        / "agent_supervisor"
        / "efficiency_state_hardening"
        / "fixtures"
        / _R45_EVIDENCE_NAME
    )
    if not fixture.is_file():
        return
    try:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(fixture.read_bytes())
        dest.chmod(0o600)
    except OSError:
        return


try:
    _aseh_board_pytest_plugin = _load_aseh_board_pytest_plugin()
except Exception:  # pragma: no cover - fail closed: run every item
    _aseh_board_pytest_plugin = None
    _plugin_configure = None
else:
    _plugin_configure = _aseh_board_pytest_plugin.pytest_configure
    pytest_collection_modifyitems = (
        _aseh_board_pytest_plugin.pytest_collection_modifyitems
    )
    pytest_runtest_logreport = _aseh_board_pytest_plugin.pytest_runtest_logreport
    pytest_runtest_call = _aseh_board_pytest_plugin.pytest_runtest_call


_CONTROL_PLANE_MODE_SNAPSHOT: dict[Path, int] = {}


def _control_plane_mode_targets(root: Path) -> list[Path]:
    targets = [root / "scripts" / "run_agent_supervisor_efficiency_state_hardening.py"]
    supervisor = root / "ipfs_accelerate_py" / "agent_supervisor"
    if supervisor.is_dir():
        targets.extend(supervisor.rglob("*.py"))
    return targets


def _snapshot_control_plane_modes(root: Path) -> None:
    _CONTROL_PLANE_MODE_SNAPSHOT.clear()
    for path in _control_plane_mode_targets(root):
        try:
            metadata = path.stat()
        except OSError:
            continue
        if stat.S_ISREG(metadata.st_mode):
            _CONTROL_PLANE_MODE_SNAPSHOT[path] = stat.S_IMODE(metadata.st_mode)


def _restore_control_plane_modes() -> None:
    """Undo umask-002 strips tests apply to ROOT so completion can accept SHA-stable protected paths."""

    for path, mode in _CONTROL_PLANE_MODE_SNAPSHOT.items():
        try:
            metadata = path.stat()
        except OSError:
            continue
        current = stat.S_IMODE(metadata.st_mode)
        if current == mode:
            continue
        try:
            os.chmod(path, mode)
        except OSError:
            continue


@pytest.fixture(autouse=True)
def _isolate_agent_orchestration_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Default supervisor DuckDB catalogs must not land in the real home directory."""

    monkeypatch.setenv(
        "IPFS_ACCELERATE_AGENT_ORCHESTRATION_DIR",
        str(tmp_path / "agent-orchestration"),
    )


def pytest_configure(config) -> None:
    _seed_r45_receipt_evidence()
    _snapshot_control_plane_modes(repo_root)
    if _plugin_configure is not None:
        _plugin_configure(config)


def pytest_sessionfinish(session, exitstatus) -> None:
    _restore_control_plane_modes()


pytest_plugins = ("ipfs_accelerate_py.testing.proof_reuse.plugin",)
