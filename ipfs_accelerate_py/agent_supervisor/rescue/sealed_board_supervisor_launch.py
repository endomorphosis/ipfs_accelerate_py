#!/usr/bin/env python3
"""Run a sealed board operator with an origin supervisor overlay first.

SPAR inserts its checkout at sys.path[0], which hides PYTHONPATH. This
launcher keeps the overlay first so clause auto-heal can load without
amending the sealed git HEAD.
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path


def _abspath(path: str) -> str:
    return str(Path(path).resolve())


class _OverlayPath(list):
    """Keep the overlay import root ahead of a sealed checkout."""

    def __init__(self, values, *, overlay: str, source_root: str) -> None:
        super().__init__(values)
        self._overlay = overlay
        self._source_root = source_root

    def insert(self, index, path):  # type: ignore[no-untyped-def]
        resolved = _abspath(path) if isinstance(path, str) else path
        if resolved == self._source_root and index == 0:
            return super().insert(1, path)
        return super().insert(index, path)


def retain_after_supervise_drain(
    source_root: str,
    argv: list[str],
    code: int,
    *,
    wait=None,
) -> int:
    """Keep a sealed SPAR owner when supervise drain-exits 0.

    Native SPAR calls retain_owner_for_closeout after lane drain. Overlayed
    supervise can still return 0; systemd then treats success as a stop unless
    this launcher holds until OPERATOR_STOP.
    """
    if code != 0 or "supervise" not in argv:
        return code
    stop = (
        Path(source_root)
        / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1"
        / "OPERATOR_STOP"
    )
    sleeper = time.sleep if wait is None else wait
    while not stop.exists() and not stop.is_symlink():
        sleeper(10)
    return 0


def pin_sealed_sys_path(source_root: str) -> None:
    """Keep the sealed checkout ahead of ipfs_kit_py after overlay imports."""
    root = _abspath(source_root)
    script_dir = str(Path(root) / "scripts")
    for item in (script_dir, root):
        while item in sys.path:
            sys.path.remove(item)
    sys.path.insert(0, script_dir)
    sys.path.insert(0, root)
    prefer_sealed_scripts(source_root)


def prefer_sealed_scripts(source_root: str) -> None:
    """ipfs_kit_py.scripts must not shadow SPAR scripts.ops."""
    sealed = str(Path(source_root).resolve() / "scripts")
    for name in list(sys.modules):
        if name != "scripts" and not name.startswith("scripts."):
            continue
        module = sys.modules[name]
        file_name = str(getattr(module, "__file__", "") or "")
        paths = [str(item) for item in (getattr(module, "__path__", None) or [])]
        if sealed in file_name or any(sealed in item for item in paths):
            continue
        del sys.modules[name]


def overlay_module(qualname: str, path: str, *, package: str) -> None:
    """Load one overlay module into a sealed package. Relative imports stay sealed."""
    import importlib
    import importlib.util

    importlib.import_module(package)
    spec = importlib.util.spec_from_file_location(qualname, path)
    if spec is None or spec.loader is None:
        raise ImportError("overlay module spec failed: " + qualname)
    module = importlib.util.module_from_spec(spec)
    module.__package__ = package
    sys.modules[qualname] = module
    spec.loader.exec_module(module)
    parent = importlib.import_module(package)
    setattr(parent, qualname.rsplit(".", 1)[-1], module)


def install_overlay(overlay: str, source_root: str) -> None:
    overlay = _abspath(overlay)
    source_root = _abspath(source_root)
    current = [str(item) for item in sys.path]
    if overlay in current:
        current.remove(overlay)
    current.insert(0, overlay)
    sys.path = _OverlayPath(current, overlay=overlay, source_root=source_root)  # type: ignore[assignment]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    overlay = os.environ.get("IPFS_ACCELERATE_SUPERVISOR_OVERLAY", "")
    source_root = os.environ.get("IPFS_ACCELERATE_SEALED_SOURCE_ROOT", "")
    while args:
        if args[0] == "--overlay":
            overlay = args[1]
            args = args[2:]
            continue
        if args[0] == "--source-root":
            source_root = args[1]
            args = args[2:]
            continue
        if args[0] == "--":
            args = args[1:]
            break
        break
    if not overlay or not source_root or not args:
        raise SystemExit(
            "usage: sealed_board_supervisor_launch.py "
            "--overlay DIR --source-root DIR -- script.py [args...]"
        )
    os.chdir(source_root)
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    overlay_root = Path(overlay)
    overlay_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.spar_accepted_root",
        str(overlay_root / "ipfs_accelerate_py/agent_supervisor/semantic_state/spar_accepted_root.py"),
        package="ipfs_accelerate_py.agent_supervisor.semantic_state",
    )
    overlay_module(
        "ipfs_accelerate_py.agent_supervisor.task_sources.spar_goal_settlement",
        str(overlay_root / "ipfs_accelerate_py/agent_supervisor/task_sources/spar_goal_settlement.py"),
        package="ipfs_accelerate_py.agent_supervisor.task_sources",
    )
    pin_sealed_sys_path(source_root)
    script = Path(args[0])
    if not script.is_absolute():
        script = Path(source_root) / script
    sys.argv = [str(script), *args[1:]]
    compiled = compile(script.read_text(encoding="utf-8"), str(script), "exec")
    namespace = {"__name__": "__main__", "__file__": str(script), "__package__": None}
    try:
        exec(compiled, namespace)
    except SystemExit as exc:
        code = exc.code
        if code is None:
            code = 0
        elif not isinstance(code, int):
            raise
        return retain_after_supervise_drain(source_root, args, code)
    return retain_after_supervise_drain(source_root, args, 0)


if __name__ == "__main__":
    raise SystemExit(main())
