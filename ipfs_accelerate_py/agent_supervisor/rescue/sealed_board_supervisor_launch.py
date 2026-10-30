#!/usr/bin/env python3
"""Run a sealed board operator with an origin supervisor overlay first.

SPAR inserts its checkout at sys.path[0], which hides PYTHONPATH. This
launcher keeps the overlay first so clause auto-heal can load without
amending the sealed git HEAD.

Explicit SAWM --sealed-runtime mode keeps the sealed package and provider
policy first. Only quack-start replaces the two schema compatibility modules;
client commands retain the complete sealed package.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import stat
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
        "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_migrations.py",
        "ipfs_accelerate_py.agent_supervisor.task_sources",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_schema.py",
        "ipfs_accelerate_py.agent_supervisor.task_sources",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state",
        "ipfs_accelerate_py/agent_supervisor/task_sources/duckdb_state.py",
        "ipfs_accelerate_py.agent_supervisor.task_sources",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.expired_attempt_custody",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/expired_attempt_custody.py",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_attempt_fairness",
        "ipfs_accelerate_py/agent_supervisor/todo_daemon/retained_attempt_fairness.py",
        "ipfs_accelerate_py.agent_supervisor.todo_daemon",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py",
        "ipfs_accelerate_py.agent_supervisor.runtime",
    ),
)


PIN_ONLY_EXTRA_GATE_MODULES = (
    (
        "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_migrations.py",
        "ipfs_accelerate_py.agent_supervisor.task_sources",
    ),
)


SAWM_SCHEMA_COMPATIBILITY_MODULES = (
    PIN_ONLY_EXTRA_GATE_MODULES[0],
    (
        "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_schema.py",
        "ipfs_accelerate_py.agent_supervisor.task_sources",
    ),
)


SAWM_SQL_RESOURCES = frozenset((
    "0001_control_plane.sql",
    "0002_causal_event_federation_core.sql",
    "0003_state_server_restart_identity.sql",
    "0004_hash_observations.sql",
    "0005_tasks_index_repair.sql",
))


def _sawm_runtime_manifest(overlay: str, expected_sha256: str) -> dict:
    """Read the closed compatibility declaration before importing package code."""
    path = Path(overlay) / "sawm-runtime-manifest.json"
    descriptor = os.open(path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode) or not 0 < before.st_size <= 65536:
            raise ValueError("bounded regular SAWM runtime manifest required")
        raw = os.read(descriptor, 65537)
        after = os.fstat(descriptor)
        if (len(raw) != before.st_size or
                (before.st_size, before.st_mtime_ns, before.st_ctime_ns) !=
                (after.st_size, after.st_mtime_ns, after.st_ctime_ns)):
            raise ValueError("SAWM runtime manifest changed during read")
    finally:
        os.close(descriptor)
    if hashlib.sha256(raw).hexdigest() != expected_sha256:
        raise ValueError("SAWM runtime manifest SHA256 differs from the external pin")
    manifest = json.loads(raw.decode("utf-8"))
    if (type(manifest) is not dict or set(manifest) - {"schema", "sql_resources", "files"}
            or manifest.get("schema") != "sawm/schema-compatibility-runtime@1"):
        raise ValueError("unsupported SAWM runtime manifest schema")
    resources = manifest.get("sql_resources")
    if (type(resources) is not dict or set(resources) != SAWM_SQL_RESOURCES
            or any(type(value) is not str or re.fullmatch(r"[0-9a-f]{64}", value) is None
                   for value in resources.values())
            or ("files" in manifest and type(manifest["files"]) is not dict)):
        raise ValueError("exact SAWM SQL resource commitments required")
    return manifest


def install_supervisor_heal_overlay(
    overlay: str,
    modules: tuple[tuple[str, str, str], ...] | None = None,
) -> None:
    """Pin supervisor heals into the sealed extra-gate package.

    Board scripts insert their own ``ipfs_accelerate_py`` first. Surgical
    ``sys.modules`` pins keep token republish and owner-side unstall on the
    one exclusive owner instead of launching a second extra-gate.
    """

    overlay_root = Path(_abspath(overlay))
    selected = modules if modules is not None else SUPERVISOR_HEAL_OVERLAY_MODULES
    for qualname, relative, package in selected:
        overlay_module(qualname, str(overlay_root / relative), package=package)


SEALED_SAWM_OBSERVATION_MODULES = (
    (
        "ipfs_accelerate_py.agent_supervisor.runtime.native_dispatch_drain",
        "ipfs_accelerate_py/agent_supervisor/runtime/native_dispatch_drain.py",
        "ipfs_accelerate_py.agent_supervisor.runtime",
    ),
    (
        "ipfs_accelerate_py.agent_supervisor.runtime.owner_status_observation",
        "ipfs_accelerate_py/agent_supervisor/runtime/owner_status_observation.py",
        "ipfs_accelerate_py.agent_supervisor.runtime",
    ),
)


def pin_sealed_sawm_observation_runtime(source_root: str) -> None:
    """Keep SAWM native observation on the sealed checkout file.

    Overlay-first extra-gate must bind typed-owner heals, but
    ``_owner_status_observation_runtime`` refuses any module whose
    ``__file__`` is not the sealed checkout path. Without this pin the
    exclusive owner never publishes ``owner-status-observation.json`` or
    the native drain listener, and the admitted coordinator sleeps on
    ``native_dispatch_observation_unavailable`` instead of spawning lanes.
    """

    root = Path(_abspath(source_root))
    for qualname, relative, package in SEALED_SAWM_OBSERVATION_MODULES:
        sys.modules.pop(qualname, None)
        overlay_module(qualname, str(root / relative), package=package)


def _is_spar_source(source_root: str, args: list[str]) -> bool:
    blob = f"{source_root} {' '.join(args)}"
    return "semantic_preserving" in blob or "materialize_semantic_preserving" in blob


def _is_sawm_source(source_root: str, args: list[str]) -> bool:
    blob = f"{source_root} {' '.join(args)}"
    return (
        "semantic-addressed-world-model" in blob
        or "semantic_addressed_world_model" in blob
    )


def _sawm_schema_start(args: list[str]) -> bool:
    """Recognize exact start argv; leave all other parsing to sealed SAWM."""
    operator_args = args[1:]
    while operator_args:
        if operator_args[0] == "--config":
            if len(operator_args) < 2 or operator_args[1].startswith("-"):
                return False
            operator_args = operator_args[2:]
        elif operator_args[0].startswith("--config="):
            if not operator_args[0].partition("=")[2]:
                return False
            operator_args = operator_args[1:]
        else:
            break
    return operator_args == ["quack-start"]


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    overlay = os.environ.get("IPFS_ACCELERATE_SUPERVISOR_OVERLAY", "")
    source_root = os.environ.get("IPFS_ACCELERATE_SEALED_SOURCE_ROOT", "")
    pin_only = False
    sealed_runtime = False
    runtime_manifest_sha256 = None
    while args:
        if args[0] == "--overlay":
            overlay = args[1]
            args = args[2:]
            continue
        if args[0] == "--source-root":
            source_root = args[1]
            args = args[2:]
            continue
        if args[0] == "--pin-only":
            pin_only = True
            args = args[1:]
            continue
        if args[0] == "--sealed-runtime":
            sealed_runtime = True
            args = args[1:]
            continue
        if args[0] == "--runtime-manifest-sha256":
            if len(args) < 2:
                raise SystemExit("--runtime-manifest-sha256 requires a digest")
            runtime_manifest_sha256 = args[1]
            args = args[2:]
            continue
        if args[0] == "--":
            args = args[1:]
            break
        break
    if not overlay or not source_root or not args:
        raise SystemExit(
            "usage: sealed_board_supervisor_launch.py "
            "--overlay DIR --source-root DIR [--pin-only | --sealed-runtime "
            "--runtime-manifest-sha256 SHA256] -- script.py [args...]"
        )
    if sealed_runtime:
        if pin_only:
            raise SystemExit("--sealed-runtime cannot be combined with --pin-only")
        if (not isinstance(runtime_manifest_sha256, str)
                or re.fullmatch(r"[0-9a-f]{64}", runtime_manifest_sha256) is None):
            raise SystemExit("--sealed-runtime requires --runtime-manifest-sha256 lowercase SHA256")
        root = Path(source_root).resolve()
        source_root = str(root)
        overlay = _abspath(overlay)
        script = Path(args[0])
        script = script.resolve() if script.is_absolute() else (root / script).resolve()
        expected = root / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py"
        if script != expected or not _is_sawm_source(source_root, args):
            raise SystemExit("--sealed-runtime requires the sealed SAWM operator")
        try:
            runtime_manifest = _sawm_runtime_manifest(overlay, runtime_manifest_sha256)
        except (OSError, UnicodeError, ValueError) as exc:
            raise SystemExit("invalid SAWM runtime manifest: " + str(exc)) from exc
    elif runtime_manifest_sha256 is not None:
        raise SystemExit("--runtime-manifest-sha256 requires --sealed-runtime")
    os.chdir(source_root)
    nested = Path(source_root) / "external" / "ipfs_accelerate"
    if pin_only or sealed_runtime:
        if sealed_runtime:
            roots = [str(Path(source_root).resolve()), str(Path(source_root).resolve() / "scripts")]
            if nested.is_dir():
                roots.append(str(nested.resolve()))
            sys.path = [*roots, *(item for item in sys.path if item not in roots)]
            prefer_sealed_scripts(source_root)
            if _sawm_schema_start(args):
                install_supervisor_heal_overlay(overlay, SAWM_SCHEMA_COMPATIBILITY_MODULES[:1])
                migrations = sys.modules[SAWM_SCHEMA_COMPATIBILITY_MODULES[0][0]]
                migrations.bind_bundled_sql_resources(
                    Path(overlay) / "ipfs_accelerate_py/agent_supervisor/task_sources/sql",
                    runtime_manifest["sql_resources"],
                )
                install_supervisor_heal_overlay(overlay, SAWM_SCHEMA_COMPATIBILITY_MODULES[1:])
                pin_sealed_sawm_observation_runtime(source_root)
        else:
            if nested.is_dir():
                sys.path.insert(0, str(nested.resolve()))
            if source_root not in sys.path:
                sys.path.insert(0, source_root)
            install_supervisor_heal_overlay(overlay, PIN_ONLY_EXTRA_GATE_MODULES)
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
    install_overlay(overlay, source_root)
    if nested.is_dir():
        sys.path.insert(0, str(nested.resolve()))
    if source_root not in sys.path:
        sys.path.insert(0, source_root)
    overlay_root = Path(overlay)
    if _is_sawm_source(source_root, args):
        pin_sealed_sawm_observation_runtime(source_root)
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
