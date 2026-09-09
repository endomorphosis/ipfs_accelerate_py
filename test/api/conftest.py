"""Test API fixtures and path setup."""

from __future__ import annotations

import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parents[2]
repo_root_text = str(repo_root)

# Pytest may prepend this checkout's parent while importing ``test.api`` as a
# package.  That parent can contain sibling checkouts with the same top-level
# package names (notably ``ipfs_datasets_py``), so merely checking whether the
# local root is present leaves the sibling ahead of the nested submodule.
# Re-promote the local checkout deterministically before API tests are imported.
sys.path[:] = [entry for entry in sys.path if entry != repo_root_text]
sys.path.insert(0, repo_root_text)

# Pin the package while the local checkout is first.  Later imports from
# ipfs_accelerate_py/ipfs_kit_py may add sibling roots to ``sys.path``.
import ipfs_datasets_py as _ipfs_datasets_py  # noqa: E402,F401

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


try:
    _aseh_board_pytest_plugin = _load_aseh_board_pytest_plugin()
except Exception:  # pragma: no cover - fail closed: run every item
    _aseh_board_pytest_plugin = None
else:
    pytest_configure = _aseh_board_pytest_plugin.pytest_configure
    pytest_collection_modifyitems = (
        _aseh_board_pytest_plugin.pytest_collection_modifyitems
    )
    pytest_runtest_logreport = _aseh_board_pytest_plugin.pytest_runtest_logreport

pytest_plugins = ("ipfs_accelerate_py.testing.proof_reuse.plugin",)
