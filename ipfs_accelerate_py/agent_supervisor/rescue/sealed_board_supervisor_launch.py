#!/usr/bin/env python3
"""Run a sealed board operator with an origin supervisor overlay first.

SPAR inserts its checkout at sys.path[0], which hides PYTHONPATH. This
launcher keeps the overlay first so clause auto-heal can load without
amending the sealed git HEAD.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path


def _abspath(path: str) -> str:
    return str(Path(path).resolve())


class _OverlayPath(list):
    """Keep the overlay import root ahead of a sealed checkout.

    Board operators insert their checkout (or nested ``external/ipfs_accelerate``)
    at ``sys.path[0]``, which hides PYTHONPATH and loads a second extra-gate
    package without supervisor heals. One exclusive owner; overlay stays first.
    """

    def __init__(self, values, *, overlay: str, source_root: str) -> None:
        super().__init__(values)
        self._overlay = overlay
        source = _abspath(source_root)
        self._source_root = source
        self._sealed_roots = {
            source,
            _abspath(str(Path(source) / "external" / "ipfs_accelerate")),
        }

    def insert(self, index, path):  # type: ignore[no-untyped-def]
        resolved = _abspath(path) if isinstance(path, str) else path
        if resolved in self._sealed_roots and index == 0:
            return super().insert(1, path)
        return super().insert(index, path)


def overlay_supervise_exit_code(
    source_root: str,
    argv: list[str],
    code: int,
    *,
    owner_lifecycle: str | None = None,
) -> int:
    """Do not hold a dead owner. Relaunch overlay if supervise returned 0 after stop.

    Native SPAR retain_owner_for_closeout loops inside supervise. If overlayed
    supervise returns, the Quack owner is already gone. Sleeping here made
    systemd think the unit was active while lifecycle=stopped.
    """
    if code != 0 or "supervise" not in argv:
        return code
    lifecycle = owner_lifecycle
    if lifecycle is None:
        status_path = (
            Path(source_root)
            / "data/agent_supervisor/semantic_preserving_autonomous_remodularization_v1"
            / "quack-owner"
            / "quack-state-server.status.json"
        )
        try:
            payload = json.loads(status_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            payload = {}
        lifecycle = str(payload.get("lifecycle") or "")
    if lifecycle == "ready":
        return 0
    return 1


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


def hold_retain_owner_until_operator_stop() -> None:
    """SPAR bootstrap closeout must not stop the owner after overlay auto-heal.

    Native retain_owner_for_closeout returns ``accepted`` when complete=true,
    then SPAR's finally block stops Quack. Overlayed clause/goal admission can
    make that true; keep the owner until OPERATOR_STOP instead.
    """
    import ipfs_accelerate_py.agent_supervisor.runtime.terminal_closeout as terminal

    def retain_owner_for_closeout(
        *,
        observe,
        wait,
        stopped,
        check_owner,
        output,
        interval_seconds: float = 10.0,
        produce=None,
    ) -> str:
        if interval_seconds <= 0:
            raise ValueError("closeout interval must be positive")
        output("implementation lanes drained; retaining native owner for acceptance")
        while not stopped():
            check_owner()
            if produce is not None:
                produce()
            observation = observe()
            if (
                observation.get("completion_authority") is True
                and observation.get("complete") is True
            ):
                output(
                    "native closeout admitted; retaining owner until operator stop"
                )
            if wait(interval_seconds):
                return "stopped"
        return "stopped"

    terminal.retain_owner_for_closeout = retain_owner_for_closeout


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


SUPERVISOR_HEAL_OVERLAY_MODULES = (
    (
        "ipfs_accelerate_py.agent_supervisor.task_sources.quack_owner_command",
        "ipfs_accelerate_py/agent_supervisor/task_sources/quack_owner_command.py",
        "ipfs_accelerate_py.agent_supervisor.task_sources",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py.agent_supervisor.runtime",
    ),
)


def install_supervisor_heal_overlay(overlay: str) -> None:
    """Pin supervisor heals into the sealed extra-gate package.

    Board scripts insert their own ``ipfs_accelerate_py`` first. Surgical
    ``sys.modules`` pins keep token republish and owner-side unstall on the
    one exclusive owner instead of launching a second extra-gate.
    """

    overlay_root = Path(_abspath(overlay))
    for qualname, relative, package in SUPERVISOR_HEAL_OVERLAY_MODULES:
        overlay_module(qualname, str(overlay_root / relative), package=package)


def _is_spar_source(source_root: str, args: list[str]) -> bool:
    blob = f"{source_root} {' '.join(args)}"
    return "semantic_preserving" in blob or "materialize_semantic_preserving" in blob


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
    nested = Path(source_root) / "external" / "ipfs_accelerate"
    if nested.is_dir():
        nested_root = str(nested.resolve())
        if nested_root not in sys.path:
            sys.path.insert(0, nested_root)
    overlay_root = Path(overlay)
    if _is_spar_source(source_root, args):
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
        hold_retain_owner_until_operator_stop()
        hold_retain_owner_until_operator_stop()
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
        return overlay_supervise_exit_code(source_root, args, code)
    return overlay_supervise_exit_code(source_root, args, 0)


if __name__ == "__main__":
    raise SystemExit(main())
