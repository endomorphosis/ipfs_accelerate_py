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
import sys
import tempfile
from pathlib import Path


MODULES = ("fleet_watchdog", "fleet_repair", "fleet_completion", "fleet_heals",
           "fleet_holds", "live_board_probe", "canonical_writer_custody",
           "diagnostic_handoff", "storage_diagnostics", "service_fleet_probe", "native_catalogue_probe")
PACKAGE_MODULES = {"agent_implementation_route": "ipfs_accelerate_py/agent_implementation_route.py"}


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
    """The existing generic watchdog/repair unit format, shared by both modes."""
    argv = [python, "-P", "-m", f"ipfs_accelerate_py.agent_supervisor.rescue.{module}",
            *mode, "--config", str(config_path)]
    description = description or f"IPFS taskboard fleet {suffix} (SPAR SAWM ASEH PCTDD PCPR DOEP)"
    return f"""[Unit]
Description={description}
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
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
