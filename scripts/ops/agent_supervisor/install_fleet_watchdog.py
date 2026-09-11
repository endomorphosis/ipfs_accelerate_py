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


MODULES = ("fleet_watchdog", "fleet_repair", "fleet_completion", "live_board_probe",
           "canonical_writer_custody", "diagnostic_handoff")


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
    modules_dir = source / "ipfs_accelerate_py/agent_supervisor/rescue"
    contents = {name: (modules_dir / f"{name}.py").read_bytes() for name in MODULES}
    for name, data in contents.items():
        compile(data, f"{name}.py", "exec")
    digest = hashlib.sha256(b"".join(contents.values())).hexdigest()[:20]
    release = library / "releases" / digest
    for directory in (config_dir, state_dir, release):
        directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    package = release
    for component in ("ipfs_accelerate_py", "agent_supervisor", "rescue"):
        package /= component
        package.mkdir(exist_ok=True)
        initial = package / "__init__.py"
        if not initial.exists():
            atomic_write(initial, b'"""Standalone fleet watchdog runtime."""\n')
    for name, data in contents.items():
        target = package / f"{name}.py"
        if target.exists() and target.read_bytes() != data:
            raise ValueError(f"existing immutable release is inconsistent: {name}")
        if not target.exists():
            atomic_write(target, data)
    if not (release / "release.json").exists():
        atomic_write(release / "release.json", encoded_json({"source": str(source), "sha256": digest}))
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
                        "launch_only_hold_files", "diagnostic_handoff"):
                if key in prior.get(board["id"], {}):
                    board[key] = prior[board["id"]][key]
        shutil.copy2(config_path, config_path.with_suffix(".json.previous"))
    if "poll_seconds" in old:
        config["poll_seconds"] = old["poll_seconds"]
    atomic_write(config_path, encoded_json(config))
    installed_units = []
    for suffix, module, mode in (("watchdog", "fleet_watchdog", ["--apply"]),
                                 ("repair", "fleet_repair", ["run"])):
        argv = [python, "-P", "-m", f"ipfs_accelerate_py.agent_supervisor.rescue.{module}", *mode,
                "--config", str(config_path)]
        unit_name = f"ipfs-taskboard-{suffix}.service"
        unit = f"""[Unit]
Description=IPFS taskboard fleet {suffix} (SPAR SAWM ASEH PCTDD PCPR DOEP)
After=network-online.target
Wants=network-online.target
StartLimitIntervalSec=0

[Service]
Type=simple
WorkingDirectory={unit_directory(str(release))}
Environment={unit_quote('PYTHONPATH=' + str(release))}
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
"""
        atomic_write(unit_dir / unit_name, unit.encode())
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
    parser.add_argument("--repair-cwd", type=Path, required=True)
    parser.add_argument("--enable", action="store_true")
    parser.add_argument("--defer-repair-restart", action="store_true",
                        help="Let a running repair finish before its dispatcher adopts this release")
    args = parser.parse_args()
    source = Path(__file__).resolve().parents[3]
    print(json.dumps(install(source, args.inventory.resolve(), args.repair_cwd.resolve(),
                            enable=args.enable, defer_repair_restart=args.defer_repair_restart), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
