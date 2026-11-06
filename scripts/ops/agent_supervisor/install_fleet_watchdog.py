#!/usr/bin/env python3
"""Install the fleet watchdog with an immutable, stdlib-only runtime release."""
from __future__ import annotations
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import stat
import sys
import tempfile
import time
from pathlib import Path


MODULES = ('fleet_watchdog', 'fleet_repair', 'fleet_completion', 'fleet_heals', 'fleet_holds', 'live_board_probe', 'canonical_writer_custody', 'diagnostic_handoff', 'storage_diagnostics', 'service_fleet_probe', 'native_catalogue_probe', 'recovery_policy', 'runtime_assets', 'systemd_heartbeat', 'repair_candidate', 'repair_candidate_runtime', 'native_executor_health', 'repair_authorization', 'repair_candidate_owners', 'repair_validation_worker')
REPAIR_PROFILE = "legal-ir"

PACKAGE_MODULES = {
    'agent_implementation_route': 'ipfs_accelerate_py/agent_implementation_route.py',
    'ipfs_accelerate_py.agent_supervisor.control': 'ipfs_accelerate_py/agent_supervisor/control/__init__.py',
    'ipfs_accelerate_py.agent_supervisor.control.control_contracts': 'ipfs_accelerate_py/agent_supervisor/control/control_contracts.py',
    'ipfs_accelerate_py.agent_supervisor.core': 'ipfs_accelerate_py/agent_supervisor/core/__init__.py',
    'ipfs_accelerate_py.agent_supervisor.core.wrapper_utils': 'ipfs_accelerate_py/agent_supervisor/core/wrapper_utils.py',
    'ipfs_accelerate_py.agent_supervisor.git_environment': 'ipfs_accelerate_py/agent_supervisor/git_environment.py',
    'ipfs_accelerate_py.agent_supervisor.merge': 'ipfs_accelerate_py/agent_supervisor/merge/__init__.py',
    'ipfs_accelerate_py.agent_supervisor.merge.checkout_lock': 'ipfs_accelerate_py/agent_supervisor/merge/checkout_lock.py',
    'ipfs_accelerate_py.agent_supervisor.merge.quarantine_validation': 'ipfs_accelerate_py/agent_supervisor/merge/quarantine_validation.py',
    'ipfs_accelerate_py.agent_supervisor.merge.workspace_quarantine': 'ipfs_accelerate_py/agent_supervisor/merge/workspace_quarantine.py',
    'ipfs_accelerate_py.agent_supervisor.merge.worktree_lifecycle': 'ipfs_accelerate_py/agent_supervisor/merge/worktree_lifecycle.py',
    'ipfs_accelerate_py.agent_supervisor.proof': 'ipfs_accelerate_py/agent_supervisor/proof/__init__.py',
    'ipfs_accelerate_py.agent_supervisor.proof.formal_verification_contracts': 'ipfs_accelerate_py/agent_supervisor/proof/formal_verification_contracts.py',
    'ipfs_accelerate_py.agent_supervisor.runtime': 'ipfs_accelerate_py/agent_supervisor/runtime/__init__.py',
    'ipfs_accelerate_py.agent_supervisor.runtime.event_log': 'ipfs_accelerate_py/agent_supervisor/runtime/event_log.py',
    'ipfs_accelerate_py.agent_supervisor.self_improvement': 'ipfs_accelerate_py/agent_supervisor/self_improvement/__init__.py',
    'ipfs_accelerate_py.agent_supervisor.self_improvement.supervisor_v2_contracts': 'ipfs_accelerate_py/agent_supervisor/self_improvement/supervisor_v2_contracts.py',
    'ipfs_accelerate_py.agent_supervisor.task_sources': 'ipfs_accelerate_py/agent_supervisor/task_sources/__init__.py',
    'ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_contracts': 'ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_contracts.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/__init__.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.app': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/app.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.artifacts': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/artifacts.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.auto_commit': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/auto_commit.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.cli': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/cli.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.context': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/context.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.core': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/core.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.deterministic_fallback': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/deterministic_fallback.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.diagnostics': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/diagnostics.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.engine': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/engine.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.file_replacement': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/file_replacement.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.git_utils': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/git_utils.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.history': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/history.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.lifecycle_wrapper': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/lifecycle_wrapper.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.llm': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/llm.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.llm_defaults': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/llm_defaults.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.plans': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/plans.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.registry': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/registry.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.runner': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/runner.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.specs': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/specs.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.status': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/status.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_loop': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_loop.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.supervisor_runtime': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/supervisor_runtime.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.task_board': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/task_board.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.typescript': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/typescript.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.worktrees': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/worktrees.py',
    'ipfs_accelerate_py.agent_supervisor.todo_daemon.wrapper': 'ipfs_accelerate_py/agent_supervisor/todo_daemon/wrapper.py',
    'ipfs_accelerate_py.agent_supervisor.validation': 'ipfs_accelerate_py/agent_supervisor/validation/__init__.py',
    'ipfs_accelerate_py.agent_supervisor.validation.validation_commands': 'ipfs_accelerate_py/agent_supervisor/validation/validation_commands.py',
    'ipfs_accelerate_py.agent_supervisor.validation.validation_runtime': 'ipfs_accelerate_py/agent_supervisor/validation/validation_runtime.py',
}


def unit_directory(value: str) -> str:
    # Unlike ExecStart and Environment, WorkingDirectory does not unquote its
    # value; surrounding quotes turn an absolute path into an invalid one.
    if not Path(value).is_absolute() or any(c in value for c in "\n\r\0"):
        raise ValueError("unit working directory must be a single absolute path")
    return value.replace("%", "%%")


def atomic_write(path: Path, data: bytes, *, mode: int = 0o600) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            os.fchmod(handle.fileno(), mode)
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def encoded_json(value: dict) -> bytes:
    return (json.dumps(value, indent=2) + "\n").encode()


def unit_quote(value: str) -> str:
    return '"' + value.replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%').replace('$', '$$').replace('\n', '\\n').replace('\r', '\\r') + '"'


def validate_inventory(inventory: dict, repair_cwd: Path) -> None:
    if not repair_cwd.is_absolute() or not repair_cwd.is_dir():
        raise ValueError("repair checkout must be an existing absolute directory")
    boards = inventory.get("boards")
    if not isinstance(boards, list) or not boards:
        raise ValueError("inventory requires a nonempty boards list")
    identifiers = set()
    for entry in boards:
        if not isinstance(entry, dict):
            raise ValueError("inventory boards must be objects")
        identifier = entry.get("id")
        if not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier) or identifier in identifiers:
            raise ValueError("inventory board ids must be unique safe names")
        identifiers.add(identifier)
        for field in ("cwd", "config_path", "runtime_root"):
            value = entry.get(field)
            if not isinstance(value, str) or not Path(value).is_absolute():
                raise ValueError(f"{identifier}: {field} must be an absolute path")
        holds = entry.get("hold_paths", [])
        if not isinstance(holds, list) or any(not isinstance(p, str) or not Path(p).is_absolute() for p in holds):
            raise ValueError(f"{identifier}: hold paths must be absolute paths")
        unavailable = not Path(entry["cwd"]).is_dir() or not Path(entry["config_path"]).is_file()
        if unavailable and not any(Path(hold).is_file() for hold in holds):
            raise ValueError(f"{identifier}: board checkout and configuration must exist unless explicitly held")
        for field in ("ensure_argv", "status_argv"):
            argv = entry.get(field, [])
            if not isinstance(argv, list) or any(not isinstance(arg, str) or not arg for arg in argv):
                raise ValueError(f"{identifier}: {field} must be a string list")


def generic_service_unit(python: str, source: Path, release: Path, config_path: Path,
                         suffix: str, module: str, mode: list[str], *,
                         description: str | None = None) -> bytes:
    """Render immutable-runtime prerequisites without requiring its old checkout."""
    module_path = release / "ipfs_accelerate_py/agent_supervisor/rescue" / f"{module}.py"
    argv = [python, "-P", "-m", f"ipfs_accelerate_py.agent_supervisor.rescue.{module}",
            *mode, "--config", str(config_path)]
    description = description or f"IPFS taskboard fleet {suffix} (SPAR SAWM ASEH PCTDD PCPR DOEP)"
    return f"""[Unit]
Description={description}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0
ConditionPathIsDirectory={unit_directory(str(release))}
ConditionFileNotEmpty={unit_directory(str(module_path))}
ConditionFileNotEmpty={unit_directory(str(config_path))}

[Service]
Type=simple
NotifyAccess=main
WatchdogSec=300
WorkingDirectory={unit_directory(str(release))}
Environment={unit_quote('PYTHONPATH=' + str(source.resolve()) + ':' + str(release))}
Environment=PYTHONUNBUFFERED=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
ExecStart={' '.join(unit_quote(arg) for arg in argv)}
Restart=always
RestartSec=30
TimeoutStopSec=330
KillMode=control-group
UMask=0077
Nice=10
CPUWeight=20
MemoryHigh=512M
MemoryMax=2G

[Install]
WantedBy=default.target
""".encode()


def _enrollment_boards(value: dict, label: str) -> dict[str, dict]:
    if not isinstance(value, dict) or not isinstance(value.get("boards"), list):
        raise ValueError(f"{label} must contain a boards list")
    result = {}
    for entry in value["boards"]:
        identifier = entry.get("id") if isinstance(entry, dict) else None
        if (not isinstance(identifier, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", identifier)
                or identifier in result):
            raise ValueError(f"{label} board ids must be unique safe names")
        result[identifier] = entry
    return result


def _enrollment_runtime(source: Path, home: Path) -> tuple[Path, dict[str, bytes]]:
    modules = source / "ipfs_accelerate_py/agent_supervisor/rescue"
    contents = {name: (modules / f"{name}.py").read_bytes() for name in MODULES}
    contents.update({name: (source / path).read_bytes()
                     for name, path in PACKAGE_MODULES.items()})
    for name, data in contents.items():
        compile(data, f"{name}.py", "exec")
    digest = hashlib.sha256(b"".join(contents.values())).hexdigest()[:20]
    release = home / ".local/lib/ipfs-taskboard-watchdog/releases" / digest
    return release, contents


def _materialize_enrollment_release(release: Path, contents: dict[str, bytes], source: Path) -> None:
    # Reuse the installer's immutable standalone-release protocol. This mode
    # never rewrites a global unit or switches its existing runtime binding.
    package = release
    package.mkdir(parents=True, exist_ok=True, mode=0o700)
    for component in ("ipfs_accelerate_py", "agent_supervisor", "rescue"):
        package /= component
        package.mkdir(exist_ok=True)
        initial = package / "__init__.py"
        if not initial.exists():
            atomic_write(initial, b'"""Standalone fleet watchdog runtime."""\n')
    for name, data in contents.items():
        target = release / PACKAGE_MODULES[name] if name in PACKAGE_MODULES else package / f"{name}.py"
        target.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"existing immutable release is inconsistent: {name}")
        if not target.exists():
            atomic_write(target, data)
    if not (release / "release.json").exists():
        atomic_write(release / "release.json", encoded_json(
            {"source": str(source), "sha256": release.name}))


def _service_fleet_enrollment(entry: dict, release: Path, config_path: Path) -> dict:
    if entry.get("probe_kind") != "service_fleet_report":
        raise ValueError("enroll-only requires the explicit service_fleet_report probe")
    if entry.get("ensure_argv"):
        raise ValueError("enroll-only preserves external launchers; ensure_argv must be empty")
    if "publication" in entry or "completion_gate" in entry:
        raise ValueError("a service-fleet diagnostic cannot acquire publication or completion authority")
    report = entry.get("report_path")
    namespace = entry.get("board_namespace")
    age = entry.get("report_max_age_seconds", 180)
    lanes = entry.get("expected_units")
    if (not isinstance(report, str) or not Path(report).is_absolute()
            or any(c in report for c in "\n\r\0")):
        raise ValueError("service-fleet report_path must be an absolute single path")
    if not isinstance(namespace, str) or not re.fullmatch(r"[a-z0-9][a-z0-9._-]{2,127}", namespace):
        raise ValueError("service-fleet board_namespace must be a canonical identifier")
    if not isinstance(age, int) or isinstance(age, bool) or not 1 <= age <= 3600:
        raise ValueError("service-fleet report_max_age_seconds must be in 1..3600")
    if (not isinstance(lanes, dict) or not lanes or len(lanes) > 32
            or any(not isinstance(k, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", k)
                   or not isinstance(v, str) or not re.fullmatch(r"[A-Za-z0-9_.@:-]+\.service", v)
                   for k, v in lanes.items())
            or len(set(lanes.values())) != len(lanes)):
        raise ValueError("service-fleet expected_units must bind explicit lane ids to service units")
    env = {"PYTHONPATH": str(release), "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    argv = [sys.executable, "-P", "-m",
            "ipfs_accelerate_py.agent_supervisor.rescue.service_fleet_probe",
            "--board", entry["id"], "--report", report, "--namespace", namespace,
            "--max-age-seconds", str(age)]
    for lane, unit in lanes.items():
        argv.extend(["--expected-unit", f"{lane}={unit}"])
    holds = set(entry.get("hold_paths", []))
    holds.update(str(Path(entry["runtime_root"]) / name) for name in ("OPERATOR_STOP", "HOLD"))
    return {"id": entry["id"], "cwd": entry["cwd"], "config": entry["config_path"],
            "hold_files": sorted(holds), "stall_seconds": 900,
            "failure_grace_seconds": 60, "blocked_grace_seconds": 300,
            "cooldown_seconds": 180, "max_backoff_seconds": 3600,
            "max_repair_attempts": 3,
            "service_fleet_binding": {"board_namespace": namespace, "max_age_seconds": age,
                                      "lanes": dict(lanes)},
            "probe": {"argv": argv, "env": env, "timeout_seconds": 90},
            "repair": {"argv": [sys.executable, "-P", "-m",
                                 "ipfs_accelerate_py.agent_supervisor.rescue.fleet_repair",
                                 "enqueue", "--config", str(config_path)],
                       "env": dict(env), "timeout_seconds": 15}}


def enroll(source: Path, inventory_path: Path, *, dry_run: bool = False,
           activate_runtime: bool = False) -> dict:
    """Append one diagnostic board without upgrading any existing fleet binding.

    The caller owns source custody. Both daemons reload this configuration, so
    enrollment never invokes systemctl or touches budgets or queue history.
    Units change only with explicit runtime activation, before root-owned cutover.
    Inventory is published first; a second-file failure leaves the prior fleet
    usable and is not an atomic two-file transaction or automatic retry grant.
    """
    home = Path.home()
    config_path = home / ".config/ipfs-taskboard-watchdog/fleet.json"
    installed_inventory = config_path.with_name("inventory.json")
    if (not config_path.is_file() or config_path.is_symlink()
            or not installed_inventory.is_file() or installed_inventory.is_symlink()):
        raise ValueError("enroll-only requires an existing regular fleet configuration and inventory")
    config_before = config_path.read_bytes()
    inventory_before = installed_inventory.read_bytes()
    old = json.loads(config_before)
    prior_inventory = json.loads(inventory_before)
    incoming = json.loads(inventory_path.read_bytes())
    prior = _enrollment_boards(prior_inventory, "existing inventory")
    fleet = _enrollment_boards(old, "existing fleet")
    selected = _enrollment_boards(incoming, "selected inventory")
    if (old.get("schema") != "agent-supervisor/fleet-watchdog-config@1"
            or not prior or set(prior) != set(fleet)
            or not isinstance(old.get("repair_worker"), dict)):
        raise ValueError("enroll-only requires consistent existing fleet/inventory bindings")
    for key, value in incoming.items():
        if key != "boards" and (key not in prior_inventory or prior_inventory[key] != value):
            raise ValueError(f"enroll-only cannot change inventory field: {key}")
    for identifier, entry in selected.items():
        if identifier in prior and entry != prior[identifier]:
            raise ValueError(f"enroll-only cannot change existing inventory board: {identifier}")
    added = [entry for identifier, entry in selected.items() if identifier not in prior]
    if len(added) != 1:
        raise ValueError("enroll-only requires exactly one genuinely new board")
    entry = added[0]
    repair_cwd = old["repair_worker"].get("cwd")
    if not isinstance(repair_cwd, str):
        raise ValueError("existing repair worker requires its original checkout")
    validate_inventory({"boards": [entry]}, Path(repair_cwd))
    release, contents = _enrollment_runtime(source, home)
    board = _service_fleet_enrollment(entry, release, config_path)
    inventory_after = dict(prior_inventory, boards=[*prior_inventory["boards"], entry])
    config_after = dict(old, boards=[*old["boards"], board])
    if activate_runtime:
        config_after["runtime_release"] = str(release)
    inventory_bytes = encoded_json(inventory_after)
    config_bytes = encoded_json(config_after)

    def descriptor(path: Path, payload: bytes) -> dict:
        return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    unit_updates = []
    if activate_runtime:
        for suffix, module, mode in (("watchdog", "fleet_watchdog", ["--apply"]),
                                     ("repair", "fleet_repair", ["run"])):
            target = home / ".config/systemd/user" / f"ipfs-taskboard-{suffix}.service"
            if target.is_symlink() or not target.is_file():
                raise ValueError("runtime activation requires existing regular generic service units")
            unit_updates.append((target, target.read_bytes(), generic_service_unit(
                sys.executable, source, release, config_path, suffix, module, mode)))

    plan = {"schema": "agent-supervisor/fleet-watchdog-enrollment-plan@1",
            "dry_run": dry_run, "applied": False, "board_id": entry["id"],
            "release": str(release), "activate_runtime": activate_runtime,
            "global_runtime_release_before": old.get("runtime_release"),
            "global_runtime_release_after": config_after.get("runtime_release"),
            "global_runtime_release_preserved": not activate_runtime,
            "files": [{"before": descriptor(installed_inventory, inventory_before),
                       "after": descriptor(installed_inventory, inventory_bytes),
                       "append": {"path": "/boards/-", "value": entry}},
                      {"before": descriptor(config_path, config_before),
                       "after": descriptor(config_path, config_bytes),
                       "append": {"path": "/boards/-", "value": board}}],
            "unrelated_board_ids_preserved": list(fleet),
            "repair_worker_preserved": True,
            "all_existing_top_level_values_preserved": not activate_runtime,
            "explicit_top_level_changes": ["runtime_release"] if activate_runtime else [],
            "unit_changes": [{"before": descriptor(path, before),
                              "after": descriptor(path, after), "postimage_utf8": after.decode()}
                             for path, before, after in unit_updates],
            "service_commands": [], "state_history_changes": [],
            "two_file_atomicity_claimed": False,
            "runtime_modules": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                                for name, data in contents.items()}}
    if dry_run:
        return plan

    def compare(path: Path, expected: bytes) -> None:
        if path.is_symlink() or path.read_bytes() != expected:
            raise ValueError(f"enrollment preimage changed: {path}")

    compare(installed_inventory, inventory_before)
    compare(config_path, config_before)
    for path, before, _after in unit_updates:
        compare(path, before)
    _materialize_enrollment_release(release, contents, source)
    compare(installed_inventory, inventory_before)
    compare(config_path, config_before)
    for path, before, after in unit_updates:
        compare(path, before)
        atomic_write(path, after)
    atomic_write(installed_inventory, inventory_bytes)
    compare(config_path, config_before)
    compare(installed_inventory, inventory_bytes)
    atomic_write(config_path, config_bytes)
    compare(installed_inventory, inventory_bytes)
    compare(config_path, config_bytes)
    for path, _before, after in unit_updates:
        compare(path, after)
    plan["applied"] = True
    return plan


def install_profile(source: Path, inventory_path: Path, repair_cwd: Path, *,
                    profile: str = "legal-ir", dry_run: bool = False,
                    enable: bool = False) -> dict:
    """Stage a separately named diagnostic fleet, never the existing global one.

    Even --enable only stages these unit postimages. The operator performs
    reviewed service activation; no daemon or repair process is touched here.
    Existing profiles must be byte-identical, otherwise a separate CAS upgrade
    is required. Existing global state, units and inventory are not rewritten.
    """
    if profile != "legal-ir":
        raise ValueError("only the explicit legal-ir profile is supported")
    home = Path.home()
    config_dir = home / ".config/ipfs-taskboard-watchdog" / profile
    config_path = config_dir / "fleet.json"
    installed_inventory = config_dir / "inventory.json"
    state_dir = home / ".local/state/ipfs-taskboard-watchdog" / profile
    if config_dir.is_symlink() or state_dir.is_symlink():
        raise ValueError("profile configuration and history must not be aliased")
    incoming = json.loads(inventory_path.read_bytes())
    selected = _enrollment_boards(incoming, "profile inventory")
    if len(selected) != 1:
        raise ValueError("the legal-ir profile requires exactly one explicit diagnostic board")
    validate_inventory(incoming, repair_cwd)
    entry = next(iter(selected.values()))
    if entry.get("publication") not in (None, False) or entry.get("completion_gate") not in (None, False):
        raise ValueError("a diagnostic profile cannot acquire publication or completion authority")
    diagnostic_entry = dict(entry)
    diagnostic_entry.pop("publication", None)
    diagnostic_entry.pop("completion_gate", None)

    global_config = home / ".config/ipfs-taskboard-watchdog/fleet.json"
    global_before = None
    previous_policy = {}
    if global_config.exists():
        if global_config.is_symlink() or not global_config.is_file():
            raise ValueError("the global repair policy must come from a regular configuration")
        global_before = global_config.read_bytes()
        existing = json.loads(global_before)
        if not isinstance(existing, dict) or not isinstance(existing.get("repair_worker", {}), dict):
            raise ValueError("the existing global repair worker policy is invalid")
        previous_policy = existing.get("repair_worker", {})
    coder = shutil.which("codex")
    repair_argv = previous_policy.get("argv") or ([coder, "exec", "--sandbox", "danger-full-access",
                                                 "-c", 'approval_policy="never"', "--json"] if coder else [])
    if (not isinstance(repair_argv, list) or not repair_argv
            or any(not isinstance(arg, str) or not arg for arg in repair_argv)
            or not shutil.which(repair_argv[0])):
        raise ValueError("a working codex repair executable is required before profile installation")
    release, contents = _enrollment_runtime(source, home)
    board = _service_fleet_enrollment(diagnostic_entry, release, config_path)
    config = {"schema": "agent-supervisor/fleet-watchdog-config@1", "poll_seconds": 60,
              "state_dir": str(state_dir), "boards": [board], "runtime_release": str(release),
              "repair_worker": {"timeout_seconds": 2400, "retry_seconds": 1800,
                                "max_backoff_seconds": 21600, **previous_policy,
                                "cwd": str(repair_cwd), "argv": repair_argv,
                                "job_unit": f"ipfs-taskboard-{profile}-repair-job.service"}}
    updates = [(installed_inventory, encoded_json(incoming)), (config_path, encoded_json(config))]
    for suffix, module, mode in (("watchdog", "fleet_watchdog", ["--apply"]),
                                 ("repair", "fleet_repair", ["run"])):
        unit = home / ".config/systemd/user" / f"ipfs-taskboard-{profile}-{suffix}.service"
        updates.append((unit, generic_service_unit(
            sys.executable, source, release, config_path, f"{profile}-{suffix}", module, mode,
            description=f"IPFS {profile} diagnostic fleet {suffix}")))

    present = []
    for path, payload in updates:
        if path.is_symlink():
            raise ValueError(f"profile target must not be aliased: {path}")
        exists = path.exists()
        present.append(exists)
        if exists and (not path.is_file() or path.read_bytes() != payload):
            raise ValueError(f"existing profile differs; explicit CAS upgrade required: {path}")
    if any(present) and not all(present):
        raise ValueError("partial existing profile requires explicit reconciliation")
    idempotent = all(present)
    if not idempotent:
        if config_dir.exists() and any(config_dir.iterdir()):
            raise ValueError("unrecognized existing profile configuration requires reconciliation")
        if state_dir.exists() and any(state_dir.iterdir()):
            raise ValueError("existing profile history requires explicit reconciliation")

    def descriptor(path: Path, payload: bytes) -> dict:
        return {"path": str(path), "bytes": len(payload), "sha256": hashlib.sha256(payload).hexdigest()}

    plan = {"schema": "agent-supervisor/fleet-watchdog-profile-plan@1", "profile": profile,
            "dry_run": dry_run, "applied": False, "existing_identical_profile": idempotent,
            "enable_requested": enable, "manual_service_activation_required": True,
            "config": str(config_path), "state": str(state_dir), "release": str(release),
            "repair_job_unit": config["repair_worker"]["job_unit"],
            "global_policy_source": descriptor(global_config, global_before) if global_before is not None else None,
            "global_configuration_inventory_units_and_history_untouched": True,
            "repair_model_and_argv_preserved": bool(previous_policy.get("argv")),
            "files": [{"before": descriptor(path, payload) if idempotent else None,
                       "after": descriptor(path, payload), "postimage_utf8": payload.decode()}
                      for path, payload in updates],
            "service_commands": [], "two_file_atomicity_claimed": False,
            "runtime_modules": {name: {"bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()}
                                for name, data in contents.items()}}
    if dry_run:
        return plan

    def compare_targets() -> None:
        for path, payload in updates:
            if path.is_symlink() or (path.exists() and (not idempotent or path.read_bytes() != payload)):
                raise ValueError(f"profile target changed during preparation: {path}")
            if idempotent and not path.exists():
                raise ValueError(f"profile target disappeared during preparation: {path}")
        if global_before is None:
            if global_config.exists() or global_config.is_symlink():
                raise ValueError("global repair policy appeared during profile preparation")
        elif global_config.is_symlink() or global_config.read_bytes() != global_before:
            raise ValueError("global repair policy changed during profile preparation")

    compare_targets()
    _materialize_enrollment_release(release, contents, source)
    compare_targets()
    if not idempotent:
        config_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        state_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        (home / ".config/systemd/user").mkdir(parents=True, exist_ok=True)
        # Configuration is staged last. No service action accompanies this
        # non-atomic first installation; partial files refuse a blind retry.
        for path, payload in [*updates[2:], updates[0], updates[1]]:
            if path.exists() or path.is_symlink():
                raise ValueError(f"profile target appeared before publication: {path}")
            atomic_write(path, payload)
    for path, payload in updates:
        if path.is_symlink() or path.read_bytes() != payload:
            raise ValueError(f"profile postimage mismatch: {path}")
    plan["applied"] = True
    return plan


def install(source: Path, inventory_path: Path, repair_cwd: Path, *, enable: bool,
            defer_repair_restart: bool = False) -> dict:
    home = Path.home()
    config_dir = home / ".config/ipfs-taskboard-watchdog"
    state_dir = home / ".local/state/ipfs-taskboard-watchdog"
    library = home / ".local/lib/ipfs-taskboard-watchdog"
    config_path = config_dir / "fleet.json"
    inventory = json.loads(inventory_path.read_text())
    if not isinstance(inventory, dict):
        raise ValueError("inventory must be a JSON object")
    validate_inventory(inventory, repair_cwd)
    old = json.loads(config_path.read_text()) if config_path.exists() else {}
    if not isinstance(old, dict) or not isinstance(old.get("boards", []), list):
        raise ValueError("existing fleet configuration is invalid")
    prior = {b["id"]: b for b in old.get("boards", [])}
    previous_policy = old.get("repair_worker", {})
    if not isinstance(previous_policy, dict):
        raise ValueError("existing repair worker policy must be an object")
    coder = shutil.which("codex")
    repair_argv = previous_policy.get("argv") or ([coder, "exec", "--sandbox", "danger-full-access",
                                                 "-c", 'approval_policy="never"', "--json"] if coder else [])
    if (not isinstance(repair_argv, list) or not repair_argv
            or any(not isinstance(arg, str) or not arg for arg in repair_argv)
            or not shutil.which(repair_argv[0])):
        raise ValueError("a working codex repair executable is required before installation")
    release, contents = _enrollment_runtime(source, home)
    for directory in (config_dir, state_dir, release):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    _materialize_enrollment_release(release, contents, source)
    installed_inventory = config_dir / "inventory.json"
    atomic_write(installed_inventory, encoded_json(inventory))
    python = sys.executable
    env = {"PYTHONPATH": str(release), "PYTHONUNBUFFERED": "1",
           "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    unit_dir = home / ".config/systemd/user"
    unit_dir.mkdir(parents=True, exist_ok=True)
    ensure_units = []
    boards = []
    for entry in inventory["boards"]:
        board_id = entry["id"]
        holds = set(entry.get("hold_paths", []))
        holds.update(prior.get(board_id, {}).get("hold_files", []))
        holds.update(str(Path(entry["runtime_root"]) / name) for name in ("OPERATOR_STOP", "HOLD"))
        board = {"id": board_id, "cwd": entry["cwd"], "config": entry["config_path"],
                 "hold_files": sorted(holds), "stall_seconds": 900,
                 "failure_grace_seconds": 60, "blocked_grace_seconds": 300,
                 "cooldown_seconds": 180, "max_backoff_seconds": 3600,
                 "probe": {"argv": [python, "-P", "-m", "ipfs_accelerate_py.agent_supervisor.rescue.live_board_probe",
                                    "--inventory", str(installed_inventory), "--board", board_id],
                           "env": env, "timeout_seconds": 90},
                 "repair": {"argv": [python, "-P", "-m", "ipfs_accelerate_py.agent_supervisor.rescue.fleet_repair",
                                     "enqueue", "--config", str(config_path)],
                            "env": env, "timeout_seconds": 15}}
        if entry.get("ensure_argv"):
            ensure_argv = entry["ensure_argv"]
            if Path(ensure_argv[0]).name == "systemctl":
                board["ensure"] = {"argv": ensure_argv, "timeout_seconds": 30}
            else:
                # A detached owner inherits its launcher's cgroup even after
                # setsid(). Run native ensure outside the watchdog cgroup so
                # upgrading/restarting the watchdog cannot kill board workers.
                ensure_unit = f"ipfs-taskboard-{board_id}-ensure.service"
                native_unit = f"""[Unit]
Description=IPFS {board_id} native supervisor ensure
After=network-online.target
Wants=network-online.target
ConditionPathIsDirectory={unit_directory(entry['cwd'])}
ConditionFileNotEmpty={unit_directory(entry['config_path'])}

[Service]
Type=oneshot
WorkingDirectory={unit_directory(entry['cwd'])}
ExecStart={' '.join(unit_quote(arg) for arg in ensure_argv)}
TimeoutStartSec=360
TimeoutStopSec=30
KillMode=process
UMask=0077
Nice=10
CPUWeight=20
Environment=OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
"""
                atomic_write(unit_dir / ensure_unit, native_unit.encode())
                ensure_units.append(ensure_unit)
                board["ensure"] = {"argv": ["systemctl", "--user", "start", "--no-block", ensure_unit],
                                   "timeout_seconds": 15}
        boards.append(board)
    config = {"schema": "agent-supervisor/fleet-watchdog-config@1", "poll_seconds": 60,
              "state_dir": str(state_dir), "boards": boards, "runtime_release": str(release),
              "repair_worker": {"timeout_seconds": 2400,
                                "retry_seconds": 1800, "max_backoff_seconds": 21600,
                                **previous_policy, "cwd": str(repair_cwd), "argv": repair_argv}}
    if config_path.exists():
        # Preserve publication manifests and operator tuning on runtime upgrades.
        for board in boards:
            for key in ("publication", "stall_seconds", "blocked_grace_seconds", "failure_grace_seconds",
                        "cooldown_seconds", "max_backoff_seconds", "max_ensure_attempts",
                        "launch_only_hold_files", "diagnostic_handoff", "storage_checks"):
                if key in prior.get(board["id"], {}):
                    board[key] = prior[board["id"]][key]
        shutil.copy2(config_path, config_path.with_suffix(".json.previous"))
    if "poll_seconds" in old:
        config["poll_seconds"] = old["poll_seconds"]
    atomic_write(config_path, encoded_json(config))
    installed_units = []
    for suffix, module, mode in (("watchdog", "fleet_watchdog", ["--apply"]),
                                 ("repair", "fleet_repair", ["run"])):
        unit_name = f"ipfs-taskboard-{suffix}.service"
        atomic_write(unit_dir / unit_name, generic_service_unit(
            python, source, release, config_path, suffix, module, mode))
        installed_units.append(unit_name)
    if enable:
        subprocess.run(["systemctl", "--user", "daemon-reload"], check=True)
        subprocess.run(["systemctl", "--user", "enable", *installed_units], check=True)
        if defer_repair_restart:
            subprocess.run(["systemctl", "--user", "restart", installed_units[0]], check=True)
            subprocess.run(["systemctl", "--user", "start", installed_units[1]], check=True)
        else:
            subprocess.run(["systemctl", "--user", "restart", *installed_units], check=True)
    return {"config": str(config_path), "state": str(state_dir), "release": str(release),
            "services": installed_units, "ensure_services": ensure_units, "enabled": enable}



def _finite_repair_release(source: Path, home: Path, relatives: list[str]):
    """Populate the exact selected normal package, including real initializers.

    Reuses original release writes; adds no synthetic namespace/initializer and
    never falls back to the previous mutable rescue tree for selected modules.
    """
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import ordinary_bytes
    _, original = _enrollment_runtime(source, home)
    files = {PACKAGE_MODULES[name] if name in PACKAGE_MODULES else
             "ipfs_accelerate_py/agent_supervisor/rescue/" + name + ".py": data
             for name, data in original.items()}
    if type(relatives) is not list or not 0 < len(relatives) <= 4096:
        raise ValueError("finite selected source population required")
    for rel in relatives:
        path = Path(rel)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError("relative selected source path required")
        files[rel] = ordinary_bytes(source / rel, 4 * 1024**2)
    for rel in ("ipfs_accelerate_py/__init__.py", "ipfs_accelerate_py/agent_supervisor/__init__.py",
                "ipfs_accelerate_py/agent_supervisor/rescue/__init__.py"):
        if rel not in files:
            raise ValueError("genuine selected package initializers required")
    material = b"".join(name.encode() + b"\0" + data for name, data in sorted(files.items()))
    release = home / ".local/lib/ipfs-taskboard-watchdog/releases" / hashlib.sha256(material).hexdigest()[:20]
    fresh = not release.exists()
    _materialize_enrollment_release(release, original, source)
    for rel, data in files.items():
        target = release / rel
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if fresh:
            # Original materializer creates its generic initializer before the
            # unpublished release is complete. Replace only in this NEW release
            # with the exact real selected initializer, before any config CAS.
            atomic_write(target, data, mode=0o400)
        elif ordinary_bytes(target, 4 * 1024**2) != data:
            raise ValueError("existing immutable selected release differs")
    expected = set(files) | {"release.json", "repair-source-commitment.json"}
    observed_entries = 0
    for directory, names, leaves in os.walk(release, followlinks=False):
        for name in (*names, *leaves):
            observed_entries += 1
            path = Path(directory) / name
            info = path.lstat()
            if (observed_entries > len(files) * 8 + 128 or info.st_uid != os.getuid()
                    or not (stat.S_ISDIR(info.st_mode) or stat.S_ISREG(info.st_mode))):
                raise ValueError("immutable repair release contains an unowned/aliased/unbounded entry")
            if stat.S_ISREG(info.st_mode) and (info.st_nlink != 1 or path.relative_to(release).as_posix() not in expected):
                raise ValueError("immutable repair release contains an undeclared file or bytecode")
    return release, original, files


def render_control_owner_commands(policy_pin: dict, *, validation_timeout_seconds: int,
                                  adoption_timeout_seconds: int) -> dict:
    """Render fixed nested owners from a separately authenticated control root.

    Top-level runtime sources may advance through the same-epoch successor
    chain. These original owner controls and policy deliberately remain exact.
    This is a pure renderer: no release materialization, grants, or commands.
    """
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners import (
        checked_descriptor, checked_control_release, checked_policy, descriptor, need)
    policy = checked_policy(Path(policy_pin["path"]), policy_pin["sha256"], current_owner=False)
    need(descriptor(Path(policy_pin["path"])) == policy_pin, "exact original owner policy descriptor required")
    control = checked_control_release(policy, current_owner=False)
    pins = {pin["path"]: pin for pin in [*policy["source_descriptors"], policy_pin, control["manifest"]]}
    need(len(pins) <= 128, "nested owner control population exceeds its unchanged finite bound")
    for pin in pins.values():
        checked_descriptor(pin)
    arguments = ["--policy", policy_pin["path"], "--policy-sha256", policy_pin["sha256"],
        "--candidate-root", "{candidate_root}", "--freeze-path", "{freeze_path}",
        "--receipt-path", "{receipt_path}", "--source-commitment", "{source_commitment}",
        "--invocation-root", "{invocation_root}"]
    prefix = [policy["python"], "-I", "-B", str(control["owner"])]
    result = {}
    for name, action, seconds in (("validation_owner", "validate", validation_timeout_seconds),
            ("adoption_owner", "adopt", adoption_timeout_seconds)):
        need(type(seconds) is int and 0 < seconds <= 1200, "finite original nested owner deadline required")
        selected = [*arguments, *(["--validation-path", "{validation_path}"] if action == "adopt" else [])]
        result[name] = {"argv": [*prefix, action, *selected],
            "verify_argv": [*prefix, "verify-validation" if action == "validate" else "verify-adoption", *selected],
            "source_descriptors": [pins[key] for key in sorted(pins)], "timeout_seconds": seconds}
    if "validation_admission_retry" in policy["validation"]:
        profile = policy["validation"]["validation_admission_retry"]
        need(validation_timeout_seconds >= profile["source_probe_timeout_seconds"] + policy["validation"]["timeout_seconds"] + 60,
             "original owner deadline cannot accommodate the explicit pre-D source probe and cleanup")
        result["validation_admission_retry"] = profile
    return result


SCIENTIFIC_GIT_BINDING_SCHEMA = "agent-supervisor/scientific-git-executor-binding@1"
SCIENTIFIC_PRODUCER_GIT_BINDING_SCHEMA = "agent-supervisor/scientific-git-executor-binding@2"


def _repair_scientific_producer_git_binding(typed: dict, profile: dict, native_root: Path):
    """Authenticate one separate producer Git generation; preserve native roots.

    Only the producer entrypoint and its three genuine package initializer
    descriptors may belong to the second root. These descriptors are metadata
    evidence; this reader never imports or executes a producer body.
    """
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import checked_descriptor, json_object
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners import need, git_command
    def document(ref):
        raw = checked_descriptor(ref)
        need(len(raw) <= 4 * 1024**2, "finite producer Git metadata required")
        return json_object(raw)
    ref = typed["producer_source_commitment"]
    commitment = document(ref)
    need(set(commitment) == {"schema", "source_root", "source_commit", "source_tree", "source_inventory",
             "source_operation_plan", "source_qualification", "runtime_activation_or_task_authority"}
         and commitment["schema"] == "agent-supervisor/scientific-producer-source-commitment@1"
         and commitment["runtime_activation_or_task_authority"] is False,
         "closed source-only producer Git commitment required")
    root = Path(commitment["source_root"])
    head, tree = commitment["source_commit"], commitment["source_tree"]
    need(root.is_absolute() and ".." not in root.parts and root.resolve() == root
         and root != native_root and root not in native_root.parents and native_root not in root.parents
         and type(head) is str and re.fullmatch(r"[0-9a-f]{40}", head)
         and type(tree) is str and re.fullmatch(r"[0-9a-f]{40}", tree),
         "separate genuine producer Git root/HEAD/tree required")
    inventory = document(commitment["source_inventory"])
    document(commitment["source_operation_plan"])
    document(commitment["source_qualification"])
    need(set(inventory) == {"schema", "source_root", "source_commit", "source_tree", "file_count", "source_bytes", "files"}
         and inventory["schema"] == "legal-ir-scientific-source-inventory/v1"
         and all(inventory[key] == commitment[key] for key in ("source_root", "source_commit", "source_tree"))
         and type(inventory["files"]) is list and 0 < len(inventory["files"]) <= 8192,
         "producer inventory generation differs")
    rows = {}
    for row in inventory["files"]:
        need(type(row) is dict and set(row) == {"path", "relative_path", "bytes", "sha256", "mode", "git_blob"},
             "closed producer inventory row required")
        rel = row["relative_path"]
        need(type(rel) is str and rel and not Path(rel).is_absolute() and ".." not in Path(rel).parts
             and Path(rel).as_posix() == rel and rel not in rows and row["path"] == str(root / rel)
             and type(row["bytes"]) is int and 0 <= row["bytes"] <= 4 * 1024**2
             and type(row["mode"]) is int and 0 <= row["mode"] <= 0o777 and not row["mode"] & 0o002
             and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
             and type(row["git_blob"]) is str and re.fullmatch(r"[0-9a-f]{40}", row["git_blob"]),
             "producer inventory path/body/mode identity differs")
        rows[rel] = row
    need(type(inventory["file_count"]) is int and inventory["file_count"] == len(rows)
         and type(inventory["source_bytes"]) is int and inventory["source_bytes"] == sum(row["bytes"] for row in rows.values()),
         "producer complete inventory population differs")
    relatives = {"ipfs_accelerate_py/__init__.py", "ipfs_accelerate_py/agent_supervisor/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/rescue/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/rescue/scientific_executor.py"}
    pins = profile["source_descriptors"]
    selected = [pin for pin in pins if Path(pin["path"]).is_relative_to(root)]
    need(len(selected) == len(relatives) and {Path(pin["path"]).relative_to(root).as_posix() for pin in selected} == relatives
         and profile["main_entrypoint"] in selected
         and profile["main_entrypoint"]["path"] == str(root / "ipfs_accelerate_py/agent_supervisor/rescue/scientific_executor.py")
         and all(Path(profile[role]["path"]).is_relative_to(native_root) for role in ("health_module", "owner_source", "driver_source")),
         "exact separate producer entrypoint/parents and retained native roles required")
    need(profile["unit"].get("environment", {}).get("PYTHONPATH") == str(native_root) + os.pathsep + profile["unit"]["working_directory"]
         and profile["unit"]["argv"].count(profile["main_entrypoint"]["path"]) == 1,
         "separate producer must preserve native-only Python search roots and exact entrypoint")
    for pin in selected:
        row = rows.get(Path(pin["path"]).relative_to(root).as_posix())
        need(row is not None and pin == {key: row[key] for key in ("path", "bytes", "sha256")},
             "producer profile source snapshot differs")
        raw = checked_descriptor(pin)
        need(stat.S_IMODE(Path(pin["path"]).lstat().st_mode) == row["mode"]
             and hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest() == row["git_blob"],
             "producer physical source/mode differs from its committed Git blob")
    environment = {key: value for key, value in os.environ.items() if not key.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_OPTIONAL_LOCKS="0",
                       GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1")
    def git(*args):
        return git_command(root, ["-c", "core.fsmonitor=false", "-c", "core.hooksPath=" + os.devnull, *args],
                           timeout=3, environment=environment)
    def generation():
        need(git("rev-parse", "--show-toplevel").strip() == str(root)
             and git("rev-parse", "--verify", "HEAD").strip() == head
             and git("rev-parse", head + "^{tree}").strip() == tree
             and git("status", "--porcelain=v1", "-z", "--untracked-files=all") == "",
             "producer genuine Git root/HEAD/tree/clean state differs")
    generation()
    entries = git("ls-tree", "-r", "-z", "--full-tree", head)
    need(len(entries.encode()) <= 4 * 1024**2, "finite producer Git tree required")
    actual = {}
    for entry in entries.split("\0"):
        if entry:
            header, rel = entry.split("\t", 1); mode, kind, blob = header.split(" ")
            need(kind in ("blob", "commit") and rel not in actual, "bounded producer Git tree required")
            actual[rel] = (kind, int(mode, 8) & 0o777, blob)
    need(all(actual.get(rel) == ("blob", 0o755 if row["mode"] & 0o111 else 0o644, row["git_blob"])
             for rel, row in rows.items()), "producer selected genuine Git inventory differs")
    def close():
        for pin in selected:
            checked_descriptor(pin)
            row = rows[Path(pin["path"]).relative_to(root).as_posix()]
            need(stat.S_IMODE(Path(pin["path"]).lstat().st_mode) == row["mode"], "producer source mode changed during read")
        for pin in (ref, commitment["source_inventory"], commitment["source_operation_plan"], commitment["source_qualification"]):
            checked_descriptor(pin)
        generation()
    return close


def _repair_scientific_git_binding(binding: dict, git_binding: dict, profile: dict, release: Path) -> None:
    """Authenticate an explicit existing Git executor; grant no source repair authority.

    The unchanged Release parser executes below. This function reads metadata,
    hashes sources and issues fixed read-only Git commands through the existing
    owner, reusing only the trusted original metadata parser without importing
    the external SDK, invoking main or executing an observer.
    """
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import checked_descriptor, json_object
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners import need, git_command
    def document(ref):
        raw = checked_descriptor(ref)
        need(len(raw) <= 4 * 1024**2, "finite scientific Git metadata required")
        return json_object(raw)
    typed = document(git_binding)
    separate_producer = (set(typed) == {"schema", "profile", "source_commitment", "observer_plan", "producer_source_commitment"}
                         and typed["schema"] == SCIENTIFIC_PRODUCER_GIT_BINDING_SCHEMA)
    need(((set(typed) == {"schema", "profile", "source_commitment", "observer_plan"}
           and typed["schema"] == SCIENTIFIC_GIT_BINDING_SCHEMA) or separate_producer)
         and typed["profile"] == binding,
         "closed explicit scientific Git/profile binding required")
    commitment = document(typed["source_commitment"])
    need(set(commitment) == {"schema", "source_root", "source_commit", "source_tree", "source_inventory",
         "original_import_Git_binding", "capsule_DATA", "source_operation_plan", "source_qualification",
         "source_qualification_successors", "runtime_activation_or_task_authority"}
         and commitment["schema"] == "legal-ir-scientific-source-commitment/v1"
         and commitment["runtime_activation_or_task_authority"] is False,
         "scientific source-only Git commitment required")
    root = Path(commitment["source_root"])
    head, tree = commitment["source_commit"], commitment["source_tree"]
    need(root.is_absolute() and ".." not in root.parts and root.resolve() == root
         and type(head) is str and re.fullmatch(r"[0-9a-f]{40}", head)
         and type(tree) is str and re.fullmatch(r"[0-9a-f]{40}", tree),
         "exact genuine scientific Git root/HEAD/tree required")
    producer_close = _repair_scientific_producer_git_binding(typed, profile, root) if separate_producer else None
    inventory = document(commitment["source_inventory"])
    capsule = document(commitment["capsule_DATA"])
    original = document(commitment["original_import_Git_binding"])
    document(commitment["source_operation_plan"])
    need(set(inventory) == {"schema", "source_root", "source_commit", "source_tree", "file_count", "source_bytes", "files"}
         and inventory["schema"] == "legal-ir-scientific-source-inventory/v1"
         and set(capsule) == {"schema", "source_root", "source_commit", "source_tree", "source_count", "source_bytes", "sources", "DATA_not_import_or_execution_credit"}
         and capsule["schema"] == "legal-ir-scientific-capsule-DATA-scope/v1"
         and capsule["DATA_not_import_or_execution_credit"] is True
         and all(value[key] == commitment[key] for value in (inventory, capsule)
                 for key in ("source_root", "source_commit", "source_tree")),
         "scientific inventory/capsule generation differs")
    def population(rows):
        need(type(rows) is list and 0 < len(rows) <= 8192, "finite scientific source population required")
        result = {}
        for row in rows:
            need(type(row) is dict and set(row) == {"path", "relative_path", "bytes", "sha256", "mode", "git_blob"},
                 "closed scientific inventory row required")
            rel = row["relative_path"]
            need(type(rel) is str and rel and not Path(rel).is_absolute() and ".." not in Path(rel).parts
                 and Path(rel).as_posix() == rel and rel not in result and row["path"] == str(root / rel)
                 and type(row["bytes"]) is int and 0 <= row["bytes"] <= 4 * 1024**2
                 and type(row["mode"]) is int and row["mode"] in (0o644, 0o755)
                 and type(row["sha256"]) is str and re.fullmatch(r"[0-9a-f]{64}", row["sha256"])
                 and type(row["git_blob"]) is str and re.fullmatch(r"[0-9a-f]{40}", row["git_blob"]),
                 "scientific inventory path/body identity differs")
            result[rel] = row
        return result
    rows = population(inventory["files"]); archived = population(capsule["sources"])
    need(type(inventory["file_count"]) is int and inventory["file_count"] == len(rows)
         and type(inventory["source_bytes"]) is int and inventory["source_bytes"] == sum(r["bytes"] for r in rows.values())
         and type(capsule["source_count"]) is int and capsule["source_count"] == len(archived)
         and type(capsule["source_bytes"]) is int and capsule["source_bytes"] == sum(r["bytes"] for r in archived.values())
         and all(rows.get(rel) == row for rel, row in archived.items()),
         "scientific complete inventory/capsule population differs")
    # The qualified original driver owns closed CLI2/capsule parsing below.
    # Here the archived selection is joined to the explicit inventory only.
    need(original.get("source_root") == str(root) and original.get("source_head") == head,
         "scientific original capsule generation differs")
    need(type(original.get("blobs")) is list and len(original["blobs"]) == len(archived)
         and {row["relative_path"] for row in original["blobs"]} == set(archived)
         and all(row == {"relative_path": rel, "bytes": archived[rel]["bytes"],
                        "object_id": archived[rel]["git_blob"]}
                 for row in original["blobs"] for rel in (row["relative_path"],)),
         "scientific original capsule/inventory selection differs")
    plan = document(typed["observer_plan"])
    need(plan.get("accelerate_root") == str(root) and plan.get("task_scope") == profile["task_scope"],
         "scientific observer plan/root/scope differs")
    cli = plan["native_metadata_cli"]
    need(cli.get("runtime_root") == str(root) and cli.get("import_head_capsule_Git") == original,
         "scientific CLI2 source/capsule differs")
    project = cli["project_import_plan"]; observation = cli["observation_binding"]
    need(type(project) is dict and project.get("accelerate_root") == str(root)
         and project.get("datasets_root") == cli["cwd"] == profile["unit"]["working_directory"]
         and project.get("publication_root") == cli["cwd"]
         and type(observation) is dict and observation.get("task_scope") == profile["task_scope"]
         and all(observation.get(k) == profile["catalogue"][k] for k in ("plan_root_cid", "repository_tree_id")),
         "scientific producer cwd/catalogue/observation scope differs")
    roles = {"main_entrypoint": "rescue/scientific_executor.py", "health_module": "rescue/native_executor_health.py",
             "owner_source": "rescue/original_embedded_observation_owner.py", "driver_source": "rescue/native_embedded_observation_driver.py"}
    if separate_producer:
        del roles["main_entrypoint"]
    pins = {pin["path"]: pin for pin in profile["source_descriptors"]}
    critical = [(profile[role], "ipfs_accelerate_py/agent_supervisor/" + rel) for role, rel in roles.items()]
    critical += [(cli["pre_body_guard"], "ipfs_accelerate_py/agent_supervisor/rescue/project_import_source_guard.py"),
                 (observation["observer_source"], "ipfs_accelerate_py/agent_supervisor/todo_daemon/native_owner_bootstrap.py"),
                 (original["route_source"], "ipfs_accelerate_py/agent_implementation_route.py")]
    def body_matches_Git(pin, row):
        raw = checked_descriptor(pin)
        need(hashlib.sha1(b"blob " + str(len(raw)).encode("ascii") + b"\0" + raw).hexdigest() == row["git_blob"],
             "scientific physical source differs from its committed Git blob")
    for pin, rel in critical:
        row = rows.get(rel)
        need(type(pin) is dict and row is not None and pin == {key: row[key] for key in ("path", "bytes", "sha256")}
             and pins.get(pin["path"]) == pin and pin in project["source_pins"]
             and cli["source_sha256"].get(pin["path"]) == pin["sha256"],
             "scientific source role/inventory/import-plan differs")
        body_matches_Git(pin, row)
    for pin in profile["source_descriptors"]:
        if Path(pin["path"]).is_relative_to(root):
            row = rows.get(Path(pin["path"]).relative_to(root).as_posix())
            need(row is not None and pin == {key: row[key] for key in ("path", "bytes", "sha256")},
                 "scientific profile source snapshot differs")
            body_matches_Git(pin, row)
    argv = profile["unit"]["argv"]
    need(argv.count("--config") == argv.count("--config-sha256") == 1,
         "scientific producer service configuration required")
    config_path = argv[argv.index("--config") + 1]; config_hash = argv[argv.index("--config-sha256") + 1]
    need(config_path in pins and pins[config_path]["sha256"] == config_hash,
         "scientific producer configuration source differs")
    config = document(pins[config_path]); template = document(config["idle_template_ref"])
    need(config["schema"] == "agent-supervisor/scientific-executor-service@1"
         and config["profile_path"] == binding["path"]
         and config["catalogue"].get("board_id") == "legal-ir-scientific"
         and all(config["catalogue"].get(k) == profile["catalogue"][k] for k in
                 ("board_namespace", "plan_root_cid", "repository_tree_id", "population_sha256"))
         and template["schema"] == "agent-supervisor/scientific-executor-cycle-template@1"
         and template["kind"] == "genuine_scoped_idle" and template["plan_ref"] == typed["observer_plan"],
         "scientific profile/config/template/observer-plan chain differs")
    request = document(template["request_ref"])
    need(request["schema"] == "legal-ir-original-native-control-request/v1" and request["cwd"] == cli["cwd"]
         and request["source_roots"][0] == cli["cwd"]
         and all(request["source_sha256"].get(pin["path"]) == pin["sha256"] for pin, _ in critical),
         "scientific original observer request source/cwd differs")
    environment = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    environment.update(GIT_CONFIG_NOSYSTEM="1", GIT_CONFIG_GLOBAL=os.devnull, GIT_OPTIONAL_LOCKS="0",
                       GIT_TERMINAL_PROMPT="0", GIT_NO_LAZY_FETCH="1")
    def git(*args):
        return git_command(root, ["-c", "core.fsmonitor=false", "-c", "core.hooksPath=" + os.devnull, *args],
                           timeout=3, environment=environment)
    def generation():
        need(git("rev-parse", "--show-toplevel").strip() == str(root)
             and git("rev-parse", "--verify", "HEAD").strip() == head
             and git("rev-parse", head + "^{tree}").strip() == tree
             and git("status", "--porcelain=v1", "-z", "--untracked-files=all") == "",
             "scientific genuine Git root/HEAD/tree/clean state differs")
    generation()
    entries = git("ls-tree", "-r", "-z", "--full-tree", head)
    need(len(entries.encode()) <= 4 * 1024**2, "finite scientific Git tree required")
    actual = {}
    for entry in entries.split("\0"):
        if not entry:
            continue
        header, rel = entry.split("\t", 1); mode, kind, blob = header.split(" ")
        need(kind in ("blob", "commit") and rel not in actual, "bounded scientific Git tree required")
        actual[rel] = (kind, int(mode, 8) & 0o777, blob)
    # The public inventory is a selected source population, not the full repo.
    # Unselected README/docs/gitlinks grant no import or execution authority.
    need(all(actual.get(rel) == ("blob", row["mode"], row["git_blob"]) for rel, row in rows.items()),
         "scientific selected genuine Git inventory differs")
    # Reuse the already qualified original metadata API at its genuine role
    # path, after authenticating all declared physical bodies against Git. Its
    # stdlib-only module body does not call main, import the SDK, open the DB,
    # create a process, reserve resources or execute an observation.
    import types
    driver = profile["driver_source"]; driver_raw = checked_descriptor(driver)
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import ordinary_bytes
    trusted_driver = release / "ipfs_accelerate_py/agent_supervisor/rescue/native_embedded_observation_driver.py"
    need(ordinary_bytes(trusted_driver, 4 * 1024**2) == driver_raw,
         "scientific Git parser differs from the trusted selected immutable component")
    name = "_ipfs_repair_Git_driver_" + driver["sha256"]
    need(name not in sys.modules, "scientific driver metadata namespace is already occupied")
    module = types.ModuleType(name); module.__file__ = driver["path"]; module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(driver_raw, driver["path"], "exec", dont_inherit=True), module.__dict__)
        parsed = module.configuration(plan)
        need(type(parsed) is dict and parsed == cli, "original scientific CLI2 metadata parser differs")
    except RuntimeError as error:
        raise ValueError("original scientific CLI2 metadata parser rejected plan") from error
    finally:
        if sys.modules.get(name) is module:
            del sys.modules[name]
    # Close metadata and live generation after all reads; no database is opened.
    for ref in (binding, git_binding, typed["source_commitment"], typed["observer_plan"], commitment["source_inventory"],
                commitment["capsule_DATA"], commitment["original_import_Git_binding"], pins[config_path],
                config["idle_template_ref"], template["request_ref"]):
        checked_descriptor(ref)
    generation()
    if producer_close is not None:
        producer_close()



def _repair_scientific_profile(binding: dict, release: Path, git_binding: dict | None = None) -> dict:
    """Call the existing metadata parser from its actual immutable release file."""
    import hashlib
    import types
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import checked_descriptor, json_object
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners import need
    profile = json_object(checked_descriptor(binding))
    roles = {"main_entrypoint": "scientific_executor.py", "health_module": "native_executor_health.py",
             "owner_source": "original_embedded_observation_owner.py",
             "driver_source": "native_embedded_observation_driver.py"}
    if git_binding is not None:
        _repair_scientific_git_binding(binding, git_binding, profile, release)
    for role, name in roles.items():
        pin = profile.get(role)
        need(type(pin) is dict and (git_binding is not None or pin.get("path") == str(
            release / "ipfs_accelerate_py/agent_supervisor/rescue" / name)),
            "scientific profile role must use the selected immutable release")
        raw = checked_descriptor(pin)
        need(len(raw) <= 4 * 1024**2, "finite scientific Python source required")
    pin = profile["health_module"]
    if git_binding is not None:
        from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import descriptor
        verifier = descriptor(release / "ipfs_accelerate_py/agent_supervisor/rescue/native_executor_health.py")
        need((verifier["bytes"], verifier["sha256"]) == (pin["bytes"], pin["sha256"]),
             "selected immutable scientific verifier differs from the pinned producer")
        pin = verifier
    raw = checked_descriptor(pin)
    name = "_ipfs_repair_release_health_" + hashlib.sha256(raw).hexdigest()
    need(name not in sys.modules, "scientific metadata parser namespace is already occupied")
    module = types.ModuleType(name)
    module.__file__ = pin["path"]
    module.__package__ = ""
    sys.modules[name] = module
    try:
        exec(compile(raw, pin["path"], "exec", dont_inherit=True), module.__dict__)
        parsed = module._profile(binding)
        need(type(parsed) is dict and parsed == profile, "canonical scientific profile parser differs")
        if git_binding is not None:
            _repair_scientific_git_binding(binding, git_binding, parsed, release)
        return parsed
    finally:
        if sys.modules.get(name) is module:
            del sys.modules[name]



def _repair_scientific_configuration(config: dict, binding: dict, release: Path, git_binding: dict | None = None) -> dict:
    """Compose one explicit scientific profile without granting runtime authority."""
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners import need
    profile = _repair_scientific_profile(binding, release, git_binding)
    value = json.loads(json.dumps(config))
    boards = value.get("boards")
    need(type(boards) is list and len(boards) == 2
         and all(type(board) is dict and type(board.get("id")) is str for board in boards)
         and {board.get("id") for board in boards} == {"legal-ir", "legal-ir-scientific"},
         "the two exact existing boards are required for a scientific profile")
    board = next(board for board in boards if board["id"] == "legal-ir-scientific")
    declared_git = board.get("native_executor_git_binding")
    need(declared_git is None or declared_git == git_binding,
         "existing scientific Git binding cannot be silently changed or removed")
    if git_binding is not None:
        need(board.get("native_executor_binding") in (None, binding),
             "scientific Git binding cannot replace another producer profile")
        board["native_executor_git_binding"] = json.loads(json.dumps(git_binding))
    catalogue = board.get("native_catalogue_binding")
    fields = ("board_namespace", "plan_root_cid", "repository_tree_id", "population_sha256")
    need(type(catalogue) is dict and all(profile["catalogue"].get(k) == catalogue.get(k)
         and k in catalogue for k in fields), "scientific profile belongs to another catalogue")
    probe = board.get("probe")
    need(type(probe) is dict and type(probe.get("argv")) is list
         and all(type(arg) is str for arg in probe["argv"]), "declared scientific probe required")
    argv = probe["argv"]
    need(argv.count("-m") == 1 and argv[argv.index("-m") + 1:] and
         argv[argv.index("-m") + 1] == "ipfs_accelerate_py.agent_supervisor.rescue.native_catalogue_probe",
         "existing canonical scientific probe required")
    for flag, expected in (("--board", board["id"]), ("--namespace", catalogue["board_namespace"]),
            ("--plan-root-cid", catalogue["plan_root_cid"]),
            ("--repository-tree-id", catalogue["repository_tree_id"]),
            ("--population-sha256", catalogue["population_sha256"])):
        need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
             and argv[argv.index(flag) + 1] == expected, "scientific probe catalogue binding differs")
    options = ("--executor-profile", "--executor-profile-bytes", "--executor-profile-sha256")
    old = board.get("native_executor_binding")
    remove = set()
    if old is not None:
        need(type(old) is dict and set(old) == {"path", "bytes", "sha256"}
             and type(old["path"]) is str and type(old["bytes"]) is int
             and type(old["sha256"]) is str, "existing scientific descriptor must remain closed")
        expected = (old["path"], str(old["bytes"]), old["sha256"])
        for flag, argument in zip(options, expected):
            need(argv.count(flag) == 1 and argv.index(flag) + 1 < len(argv)
                 and argv[argv.index(flag) + 1] == argument, "existing scientific probe/profile join differs")
            index = argv.index(flag)
            remove.update((index, index + 1))
    else:
        need(not any(flag in argv for flag in options), "unbound scientific profile arguments refused")
    probe["argv"] = [arg for index, arg in enumerate(argv) if index not in remove]
    for flag, argument in zip(options, (binding["path"], str(binding["bytes"]), binding["sha256"])):
        probe["argv"].extend((flag, argument))
    board["native_executor_binding"] = json.loads(json.dumps(binding))
    return value



def prepare_existing_repair_configuration(source: Path, contract: dict, bindings: dict, *, scientific_executor_binding: dict | None = None, scientific_executor_git_binding: dict | None = None) -> dict:
    """Explicit operator grant for the existing profile; never mint an epoch.

    The caller provides the complete finite epoch and policy. This function
    proves the real selected commit/tree/files under the public source owner,
    then reuses the same exact profile CAS renderer as autonomous source-only
    successors. It does not touch the legacy job or start/stop a service.
    """
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners import (
        checked_policy, public_source_lease, git_command, need)
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import descriptor, digest, ordinary_bytes
    need(type(bindings) is dict and set(bindings) == {"candidate_policy", "candidate_runtime", "authorization_epoch", "source_snapshot"}
         and bindings["source_snapshot"] is None, "explicit initial bindings and no supplied snapshot required")
    selected = bindings["candidate_policy"]
    need(Path(selected["repository_root"]) == source, "initial source root differs from the explicit policy")
    argv = bindings["candidate_runtime"]["validation_owner"]["argv"]
    need(argv.count("--policy") == argv.count("--policy-sha256") == 1, "maintained public owner profile binding required")
    owner_policy = checked_policy(Path(argv[argv.index("--policy") + 1]), argv[argv.index("--policy-sha256") + 1], current_owner=False)
    owner_pin = {"path": str(Path(argv[argv.index("--policy") + 1])),
        "bytes": len(ordinary_bytes(Path(argv[argv.index("--policy") + 1]))), "sha256": argv[argv.index("--policy-sha256") + 1]}
    nested = render_control_owner_commands(owner_pin,
        validation_timeout_seconds=bindings["candidate_runtime"]["validation_owner"]["timeout_seconds"],
        adoption_timeout_seconds=bindings["candidate_runtime"]["adoption_owner"]["timeout_seconds"])
    need(("validation_admission_retry" in bindings["candidate_runtime"]) == ("validation_admission_retry" in nested),
         "initial validation retry owner/runtime authority differs")
    need(all(bindings["candidate_runtime"][key] == value for key, value in nested.items()),
         "initial nested owners differ from the immutable control renderer")
    control_root = Path(owner_policy["control_release"]["root"])
    need(source != control_root and source not in control_root.parents and control_root not in source.parents,
         "mutable selected source and trusted control release must be separate")
    if "external_fixture_profile" in owner_policy["validation"]:
        control_root = Path(owner_policy["control_release"]["root"])
        for trusted in owner_policy["validation"]["trusted_tests"]:
            relative = Path(trusted["path"]).relative_to(control_root).as_posix()
            need(relative in selected["source_paths"] and relative not in selected["allowed_paths"],
                 "initial Git source policy must already include unchanged trusted tests outside provider writes")
            actual = descriptor(source / relative)
            need((actual["bytes"], actual["sha256"]) == (trusted["bytes"], trusted["sha256"]),
                 "initial active test generation differs from immutable control inputs")
    with public_source_lease(owner_policy)(source, mode="read", timeout_seconds=selected["git_timeout_seconds"]):
        need(git_command(source, ["rev-parse", "--verify", "HEAD"], timeout=selected["git_timeout_seconds"]).strip()
             == selected["base_commit"], "initial source HEAD is not the declared genuine snapshot")
        tree = git_command(source, ["rev-parse", selected["base_commit"] + "^{tree}"], timeout=selected["git_timeout_seconds"]).strip()
        manifest = {}
        for rel in selected["source_paths"]:
            pin = descriptor(source / rel)
            raw = git_command(source, ["show", selected["base_commit"] + ":" + rel], timeout=selected["git_timeout_seconds"]).encode("utf-8")
            need(raw == ordinary_bytes(source / rel, 4 * 1024**2), "initial selected source differs from its real Git object")
            manifest[rel] = {"bytes": pin["bytes"], "sha256": pin["sha256"],
                             "executable": bool((source / rel).lstat().st_mode & 0o111)}
        commitment = digest({"base_commit": selected["base_commit"], "manifest": manifest})
    snapshot = {"prior_policy": selected, "new_commit": selected["base_commit"], "new_tree": tree,
                "source_commitment": commitment, "manifest": manifest}
    concrete = {**bindings, "source_snapshot": snapshot}
    optional = {}
    if scientific_executor_binding is not None or scientific_executor_git_binding is not None:
        optional = {"scientific_executor_binding": scientific_executor_binding,
                    "scientific_executor_git_binding": scientific_executor_git_binding}
    return prepare_repair_upgrade(source, contract, commitment, initial_bindings=concrete, **optional)


def prepare_repair_upgrade(source: Path, contract: dict, source_commitment: str, *, initial_bindings: dict | None = None, scientific_executor_binding: dict | None = None, scientific_executor_git_binding: dict | None = None) -> dict:
    """One existing profile, exact beforeimages, immutable source population.

    This only stages files. The separate activation owner, after the original
    repair job closes, applies the reviewed CAS and bounded service commands.
    """
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import (
        checked_descriptor, descriptor, digest, json_object, ordinary_bytes)
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners import need
    fields = {"schema", "profile", "config", "inventory", "units", "prior_job",
              "expected_loaded_before", "job_unit", "source_successor", "initial_release"}
    need(type(contract) is dict and set(contract) == fields
         and contract["schema"] == "agent-supervisor/repair-profile-upgrade-contract@1"
         and contract["profile"] == REPAIR_PROFILE, "one explicit existing profile contract required")
    raw = checked_descriptor(contract["config"]); old = json_object(raw)
    checked_descriptor(contract["inventory"]); checked_descriptor(contract["prior_job"])
    boards = old["boards"]
    need(type(boards) is list and bool(boards) and all(type(board) is dict and type(board.get("id")) is str for board in boards)
         and len({board["id"] for board in boards}) == len(boards), "existing uniquely named fleet boards required")
    selected = [index for index, board in enumerate(boards) if board["id"] == REPAIR_PROFILE]
    need(len(selected) == 1, "one existing legal-ir board must be selected")
    board_index = selected[0]
    state = Path(old["state_dir"])
    need(contract["prior_job"]["path"] == str(state / "repairs/legal-ir/job.json"), "legacy history binding differs")
    need(set(contract["units"]) == {"watchdog", "repair"}, "two exact existing units required")
    for suffix, pin in contract["units"].items():
        checked_descriptor(pin)
        need(Path(pin["path"]).name == "ipfs-taskboard-" + REPAIR_PROFILE + "-" + suffix + ".service", "foreign unit target")
    successor = contract["source_successor"]
    config = json.loads(json.dumps(old))
    if initial_bindings is None:
        need(successor is not None and successor["source_commitment"] == source_commitment,
             "source successor belongs to another freeze")
        need(old["repair_worker"]["candidate_policy"] == successor["prior_policy"], "profile source preimage differs")
        config["repair_worker"]["candidate_policy"] = successor["successor_policy"]
        config["repair_worker"]["candidate_runtime"]["source_descriptors"] = successor["successor_source_descriptors"]
        need(old["boards"] == config["boards"], "source upgrade cannot mint an epoch")
        render = successor
    else:
        need(successor is None and set(initial_bindings) == {"candidate_policy", "candidate_runtime", "authorization_epoch", "source_snapshot"},
             "one explicit initial policy/runtime/epoch binding required")
        need("candidate_policy" not in old["repair_worker"] and "candidate_runtime" not in old["repair_worker"]
             and "repair_authorization_epoch" not in old["boards"][board_index], "existing productive authority requires a separate reviewed transition")
        config["repair_worker"]["candidate_policy"] = initial_bindings["candidate_policy"]
        config["repair_worker"]["candidate_runtime"] = initial_bindings["candidate_runtime"]
        config["boards"][board_index]["repair_authorization_epoch"] = initial_bindings["authorization_epoch"]
        render = initial_bindings["source_snapshot"]
        need(render["source_commitment"] == source_commitment and render["prior_policy"] == initial_bindings["candidate_policy"],
             "initial source snapshot differs from explicit source policy")
        from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import require_current_authorization
        from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate import parse_candidate_policy
        from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_runtime import parse_runtime_policy
        require_current_authorization(config, config["boards"][board_index])
        parsed = parse_candidate_policy(config["repair_worker"])
        need(parsed is not None, "initial selected candidate policy missing")
        parse_runtime_policy(config["repair_worker"], parsed)
    need(len(config["boards"]) == len(boards)
         and [board["id"] for board in config["boards"]] == [board["id"] for board in boards]
         and all(config["boards"][index] == board for index, board in enumerate(boards) if index != board_index),
         "every other existing board must remain exactly unchanged")
    expected_board = boards[board_index] if initial_bindings is None else {
        **boards[board_index], "repair_authorization_epoch": initial_bindings["authorization_epoch"]}
    need(config["boards"][board_index] == expected_board, "selected board changes exceed the explicit finite epoch grant")
    need(old["repair_worker"]["job_unit"] == contract["job_unit"], "original job unit differs")
    home = Path(contract["config"]["path"]).parents[3]
    need(home / ".config/ipfs-taskboard-watchdog" / REPAIR_PROFILE / "fleet.json" == Path(contract["config"]["path"]), "canonical existing profile layout required")
    science = next((board for board in config["boards"] if board["id"] == "legal-ir-scientific"), {})
    if scientific_executor_binding is None and science.get("native_executor_git_binding") is not None:
        need(scientific_executor_git_binding in (None, science["native_executor_git_binding"]),
             "explicit scientific Git binding differs from the existing producer")
        scientific_executor_binding = science.get("native_executor_binding")
        scientific_executor_git_binding = science["native_executor_git_binding"]
    need(scientific_executor_git_binding is None or scientific_executor_binding is not None,
         "scientific Git binding requires its exact profile")
    release, contents, selected_files = _finite_repair_release(source, home, render["prior_policy"]["source_paths"])
    config["runtime_release"] = str(release)
    if scientific_executor_binding is not None:
        config = _repair_scientific_configuration(config, scientific_executor_binding, release, scientific_executor_git_binding)
    sources = {name: descriptor(release / PACKAGE_MODULES[name] if name in PACKAGE_MODULES else
        release / "ipfs_accelerate_py/agent_supervisor/rescue" / (name + ".py")) for name in contents}
    commitment_path = release / "repair-source-commitment.json"
    commitment = encoded_json({"schema": "agent-supervisor/repair-source-release@1",
        "source_commitment": source_commitment, "sources": sources,
        "snapshot_commit": render["new_commit"], "snapshot_tree": render["new_tree"],
        "selected_sources": {rel: descriptor(release / rel) for rel in selected_files}})
    if commitment_path.exists():
        need(ordinary_bytes(commitment_path) == commitment, "immutable source commitment differs")
    else:
        atomic_write(commitment_path, commitment, mode=0o400)
    if initial_bindings is not None:
        need(descriptor(commitment_path) == contract["initial_release"],
             "initial immutable release commitment differs from the explicit grant")
    updates = []
    for suffix, module, mode in (("watchdog", "fleet_watchdog", ["--apply"]), ("repair", "fleet_repair", ["run"])):
        pin = contract["units"][suffix]
        data = generic_service_unit(sys.executable, Path(render["prior_policy"]["repository_root"]), release,
            Path(contract["config"]["path"]), REPAIR_PROFILE + "-" + suffix, module, mode)
        text = data.decode()
        lines = text.splitlines()
        argv = [sys.executable, "-P", "-B", str(release / "ipfs_accelerate_py/agent_supervisor/rescue/repair_candidate_owners.py"),
                "serve", "--service", suffix, "--config", contract["config"]["path"]]
        lines = [("ExecStart=" + " ".join(unit_quote(a) for a in argv)) if line.startswith("ExecStart=") else
                 ("Environment=" + unit_quote("PYTHONPATH=" + str(release) + ":" + render["prior_policy"]["repository_root"]))
                 if line.startswith("Environment=") and "PYTHONPATH=" in line else line for line in lines]
        updates.append({"before": pin, "postimage_utf8": "\n".join(lines) + "\n",
                        "mode": stat.S_IMODE(Path(pin["path"]).lstat().st_mode)})
    updates.append({"before": contract["config"], "postimage_utf8": encoded_json(config).decode(),
                    "mode": stat.S_IMODE(Path(contract["config"]["path"]).lstat().st_mode)})
    plan = {"schema": "agent-supervisor/repair-profile-upgrade-stage@1", "source_commitment": source_commitment,
        "contract": contract, "files": updates, "release": str(release), "release_sources": sources,
        "release_commitment": descriptor(commitment_path), "selected_release_sources": {rel: descriptor(release / rel) for rel in selected_files}, "state_dir": str(state),
        "source_successor": successor, "initial_bindings": initial_bindings, "manual_activation_required": True,
        "staged_only": True, "service_actions": 0, "legacy_history_preserved": True}
    if scientific_executor_binding is not None:
        plan["scientific_executor_binding"] = json.loads(json.dumps(scientific_executor_binding))
    if scientific_executor_git_binding is not None:
        plan["scientific_executor_git_binding"] = json.loads(json.dumps(scientific_executor_git_binding))
    verify_repair_upgrade(plan)
    return plan


def verify_repair_upgrade(plan: dict, *, installed: bool = False) -> None:
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import checked_descriptor, descriptor, json_object
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners import need, proc_identity
    need(plan["schema"] == "agent-supervisor/repair-profile-upgrade-stage@1" and len(plan["files"]) == 3,
         "exact two-unit and configuration CAS required")
    checked_descriptor(plan["contract"]["prior_job"]); checked_descriptor(plan["contract"]["inventory"])
    checked_descriptor(plan["contract"]["initial_release"])
    checked_descriptor(plan["release_commitment"])
    for pin in [*plan["release_sources"].values(), *plan["selected_release_sources"].values()]:
        checked_descriptor(pin)
    post = json_object(plan["files"][-1]["postimage_utf8"].encode())
    science = next((board for board in post["boards"] if board["id"] == "legal-ir-scientific"), {})
    need(science.get("native_executor_git_binding") == plan.get("scientific_executor_git_binding")
         and (plan.get("scientific_executor_git_binding") is None or
              science.get("native_executor_binding") == plan.get("scientific_executor_binding")),
         "staged scientific Git/profile binding differs from the full configuration")
    if plan.get("scientific_executor_binding") is not None:
        need(_repair_scientific_configuration(post, plan["scientific_executor_binding"], Path(plan["release"]),
             plan.get("scientific_executor_git_binding")) == post, "staged scientific profile/probe join differs")
    for row in plan["files"]:
        path = Path(row["before"]["path"])
        if installed:
            need(path.read_bytes() == row["postimage_utf8"].encode()
                 and stat.S_IMODE(path.lstat().st_mode) == row["mode"], "profile postimage differs")
            descriptor(path)  # Ordinary identity/noalias check, not just equal bytes.
        else:
            checked_descriptor(row["before"])
            need(stat.S_IMODE(path.lstat().st_mode) == row["mode"], "profile mode preimage differs")
    if not installed:
        for identity in plan["contract"]["expected_loaded_before"].values():
            need(proc_identity(identity["pid"]) == identity, "original live fleet identity changed")


def apply_repair_upgrade(plan: dict, runner, *, deadline=None) -> dict:
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners import (
        need, install_source_successor, verify_original_job_closed, cutover_remaining)
    deadline = time.monotonic() + 120 if deadline is None else deadline
    cutover_remaining(deadline)
    verify_repair_upgrade(plan)
    verify_original_job_closed(plan["contract"]["job_unit"], runner, deadline=deadline)
    source_result = install_source_successor(plan["source_successor"], deadline=deadline) if plan["source_successor"] is not None else {
        "completed": True, "source_WRITE_entered": False, "source_WRITE_closed": True, "postwrite_source_READ_closed": True}
    # The old owner job is now closed. This external process is neither of the
    # two units below and cannot stop its own supervising repair invocation.
    for row in plan["files"]:
        cutover_remaining(deadline)
        from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import checked_descriptor
        checked_descriptor(row["before"])
        if plan.get("scientific_executor_binding") is not None:
            _repair_scientific_profile(plan["scientific_executor_binding"], Path(plan["release"]),
                plan.get("scientific_executor_git_binding"))
        cutover_remaining(deadline)
        atomic_write(Path(row["before"]["path"]), row["postimage_utf8"].encode(), mode=row["mode"])
    verify_repair_upgrade(plan, installed=True)
    commands = []
    for argv in (["systemctl", "--user", "daemon-reload"],
                 ["systemctl", "--user", "restart", "ipfs-taskboard-" + REPAIR_PROFILE + "-watchdog.service"],
                 ["systemctl", "--user", "restart", "ipfs-taskboard-" + REPAIR_PROFILE + "-repair.service"]):
        result = runner({"argv": argv}, cwd="/", timeout=min(45, cutover_remaining(deadline)))
        commands.append({"argv": argv, "returncode": result["returncode"]})
        need(type(result["returncode"]) is int and result["returncode"] == 0, "profile service action failed; no replay")
    verify_repair_upgrade(plan, installed=True)
    return {"schema": "agent-supervisor/repair-profile-upgrade-installation@1", "completed": True,
        "commands": commands, "source_successor": source_result, "source_WRITE_closed": True,
        "postwrite_source_READ_closed": True, "config_units_CAS_closed": True}


def observe_repair_upgrade(plan: dict, runner, *, deadline=None) -> dict:
    """Bounded read-only readiness; restarting never repeats during this wait."""
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_authorization import descriptor, json_object, ordinary_bytes
    from ipfs_accelerate_py.agent_supervisor.rescue.repair_candidate_owners import need, proc_identity, cutover_remaining
    deadline = time.monotonic() + 10 if deadline is None else deadline
    cutover_remaining(deadline)
    verify_repair_upgrade(plan, installed=True)
    observed = {}; identities = {}
    for suffix in ("watchdog", "repair"):
        unit = "ipfs-taskboard-" + REPAIR_PROFILE + "-" + suffix + ".service"
        while True:
            left = cutover_remaining(deadline)
            result = runner({"argv": ["systemctl", "--user", "show", unit,
                "--property=MainPID,InvocationID,ActiveState,Result"]}, cwd="/", timeout=min(10, left))
            need(type(result["returncode"]) is int and result["returncode"] == 0, "actual fleet unit observation failed")
            props = dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)
            need(props.get("ActiveState") not in {"failed", "inactive"}
                 and props.get("Result") == "success", "new fleet unit failed while awaiting source readiness")
            ready = (props.get("ActiveState") == "active" and props.get("MainPID", "").isdecimal()
                and int(props["MainPID"]) > 0 and re.fullmatch(r"[0-9a-f]{32}", props.get("InvocationID", "")))
            if ready:
                owner = proc_identity(int(props["MainPID"]))
                need(owner is not None, "new fleet PID disappeared during readiness")
                identity = {"owner": owner, "invocation_id": props["InvocationID"]}
                if suffix in identities:
                    need(identities[suffix] == identity, "fleet owner/invocation changed during readiness; no restart replay")
                else:
                    identities[suffix] = identity
                path = Path(plan["state_dir"]) / "runtime-loaded" / (props["InvocationID"] + "-" + suffix + ".json")
                try:
                    info = path.lstat()
                except FileNotFoundError:
                    pass  # Type=simple may be active before serve publishes its attestation.
                else:
                    need(stat.S_ISREG(info.st_mode) and info.st_uid == os.getuid() and info.st_nlink == 1
                         and not info.st_mode & 0o222, "source attestation has an unsafe file identity")
                    try:
                        value = json_object(ordinary_bytes(path)); pin = descriptor(path)
                    except json.JSONDecodeError:
                        # The original O_EXCL writer may still be completing this
                        # exact invocation's first file; do not accept partial JSON.
                        value = None
                    except ValueError as error:
                        if str(error) != "repair metadata changed while read":
                            raise
                        value = None
                    if value is not None:
                        need(value["owner"] == owner and value["invocation_id"] == props["InvocationID"]
                             and value["source_commitment"] == plan["source_commitment"]
                             and value["release"] == plan["release"]
                             and value["loaded_module"] == plan["release_sources"]["fleet_" + suffix]
                             and value["config"] == descriptor(Path(plan["contract"]["config"]["path"]))
                             and proc_identity(owner["pid"]) == owner,
                             "actual loaded source/config/PID birth differs")
                        observed[suffix] = pin
                        break
            time.sleep(min(0.05, cutover_remaining(deadline)))
    # One service cannot restart unnoticed while the other finishes readiness.
    for suffix, identity in identities.items():
        unit = "ipfs-taskboard-" + REPAIR_PROFILE + "-" + suffix + ".service"
        result = runner({"argv": ["systemctl", "--user", "show", unit,
            "--property=MainPID,InvocationID,ActiveState,Result"]}, cwd="/", timeout=min(10, cutover_remaining(deadline)))
        props = dict(line.split("=", 1) for line in result["stdout"].splitlines() if "=" in line)
        need(type(result["returncode"]) is int and result["returncode"] == 0
             and props.get("MainPID") == str(identity["owner"]["pid"])
             and props.get("InvocationID") == identity["invocation_id"] and props.get("ActiveState") == "active"
             and props.get("Result") == "success" and proc_identity(identity["owner"]["pid"]) == identity["owner"],
             "actual fleet identity changed before readiness closed")
        need(descriptor(Path(observed[suffix]["path"])) == observed[suffix], "readiness attestation changed")
    return {"loaded_runtime": observed, "runtime_unit_config_CAS_closed": True,
            "read_only_readiness_wait_closed": True, "readiness_deadline_monotonic": deadline}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--repair-cwd", type=Path)
    parser.add_argument("--profile", choices=["legal-ir"],
                        help="Stage an isolated LegalIR diagnostic fleet without changing the global fleet")
    parser.add_argument("--enroll-only", action="store_true",
                        help="Append one new diagnostic board, preserving the existing fleet and units")
    parser.add_argument("--activate-runtime", action="store_true",
                        help="With --enroll-only, stage only the two generic units and new runtime binding")
    parser.add_argument("--dry-run", action="store_true",
                        help="With --enroll-only or --profile, print the plan without any home mutations")
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--defer-repair-restart", action="store_true",
                        help="Let a running repair finish before its dispatcher adopts this release")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3]
    if args.profile:
        if args.enroll_only or args.activate_runtime or args.defer_repair_restart:
            parser.error("--profile cannot enroll or upgrade the global fleet")
        if args.repair_cwd is None:
            parser.error("--profile requires an explicit --repair-cwd")
        result = install_profile(source, args.inventory.resolve(), args.repair_cwd.resolve(),
                                 profile=args.profile, dry_run=args.dry_run, enable=args.enable)
    elif args.enroll_only:
        if args.enable or args.defer_repair_restart or args.repair_cwd is not None:
            parser.error("--enroll-only cannot change services or the repair checkout")
        result = enroll(source, args.inventory.resolve(), dry_run=args.dry_run,
                        activate_runtime=args.activate_runtime)
    else:
        if args.dry_run or args.activate_runtime:
            parser.error("--dry-run and --activate-runtime require --enroll-only")
        if args.repair_cwd is None:
            parser.error("--repair-cwd is required for a full installation")
        result = install(source, args.inventory.resolve(), args.repair_cwd.resolve(),
                         enable=args.enable, defer_repair_restart=args.defer_repair_restart)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
