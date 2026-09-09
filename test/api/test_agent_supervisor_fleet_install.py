"""Installation preserves controls and launches outside live checkout imports."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import shutil

import pytest


SCRIPT = Path(__file__).resolve().parents[2] / "scripts/ops/agent_supervisor/install_fleet_watchdog.py"
spec = importlib.util.spec_from_file_location("fleet_installer", SCRIPT)
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


@pytest.fixture
def inputs(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    original_which = installer.shutil.which
    monkeypatch.setattr(installer.shutil, "which",
        lambda name: sys.executable if name == "codex" else original_which(name))
    source = tmp_path / "source"
    modules = source / "ipfs_accelerate_py/agent_supervisor/rescue"
    modules.mkdir(parents=True)
    for name in installer.MODULES:
        (modules / f"{name}.py").write_text("print(__file__)\n")
    board_root = tmp_path / "board"
    board_root.mkdir()
    config = board_root / "board.json"
    config.write_text("{}")
    inventory = tmp_path / "inventory.json"
    inventory.write_text(json.dumps({"boards": [{
        "id": "spar", "cwd": str(board_root), "config_path": str(config),
        "runtime_root": str(board_root / "state"), "hold_paths": [],
        "ensure_argv": [sys.executable, "native.py", "ensure"],
    }]}))
    return source, inventory, board_root, home


def test_install_uses_safe_python_path_and_separate_native_ensure_unit(inputs):
    source, inventory, board_root, home = inputs
    shadow = board_root / "ipfs_accelerate_py"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise RuntimeError('live checkout shadow')\n")
    result = installer.install(source, inventory, board_root, enable=False)
    config = json.loads(Path(result["config"]).read_text())
    board = config["boards"][0]
    process = subprocess.run(board["probe"]["argv"], cwd=board_root,
        env={**os.environ, **board["probe"]["env"]}, capture_output=True, text=True, check=True)
    assert result["release"] in process.stdout
    assert "-P" in board["repair"]["argv"]
    assert board["ensure"]["argv"] == ["systemctl", "--user", "start", "--no-block", "ipfs-taskboard-spar-ensure.service"]
    native = (home / ".config/systemd/user/ipfs-taskboard-spar-ensure.service").read_text()
    assert "Type=oneshot" in native
    assert "KillMode=process" in native
    assert "TimeoutStartSec=360" in native
    assert "native.py" in native
    assert result["ensure_services"] == ["ipfs-taskboard-spar-ensure.service"]


@pytest.mark.skipif(shutil.which("systemd-analyze") is None, reason="systemd is unavailable")
def test_generated_service_files_pass_systemd_validation(inputs):
    source, inventory, board_root, home = inputs
    result = installer.install(source, inventory, board_root, enable=False)
    unit_dir = home / ".config/systemd/user"
    paths = [str(unit_dir / name) for name in result["services"] + result["ensure_services"]]
    process = subprocess.run(["systemd-analyze", "--user", "--man=no", "verify", *paths],
                             capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr


def test_upgrade_preserves_holds_backoff_and_restarts_only_fleet_services(inputs, monkeypatch):
    source, inventory, board_root, home = inputs
    first = installer.install(source, inventory, board_root, enable=False)
    config_path = Path(first["config"])
    config = json.loads(config_path.read_text())
    board = config["boards"][0]
    hold = str(board_root / "custom.hold")
    board.update(hold_files=[hold], cooldown_seconds=777, max_backoff_seconds=999,
                 max_ensure_attempts=4, publication={"board_id": "spar"})
    config["poll_seconds"] = 75
    config["repair_worker"]["retry_seconds"] = 1000
    config_path.write_text(json.dumps(config))
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    upgraded = installer.install(source, inventory, board_root, enable=True)
    current = json.loads(config_path.read_text())
    assert hold in current["boards"][0]["hold_files"]
    assert current["boards"][0]["cooldown_seconds"] == 777
    assert current["boards"][0]["max_ensure_attempts"] == 4
    assert current["boards"][0]["publication"] == {"board_id": "spar"}
    assert current["repair_worker"]["retry_seconds"] == 1000
    assert current["poll_seconds"] == 75
    assert upgraded["release"] == first["release"]
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", *upgraded["services"]],
        ["systemctl", "--user", "restart", *upgraded["services"]],
    ]
    assert "ipfs-taskboard-spar-ensure.service" not in calls[-1]
    assert config_path.with_suffix(".json.previous").exists()


@pytest.mark.parametrize("change", ["missing_config", "duplicate_id", "relative_runtime"])
def test_invalid_inventory_fails_before_configuration_or_units_change(inputs, change):
    source, inventory, board_root, home = inputs
    data = json.loads(inventory.read_text())
    if change == "missing_config":
        data["boards"][0]["config_path"] = str(board_root / "missing.json")
    elif change == "duplicate_id":
        data["boards"].append(dict(data["boards"][0]))
    else:
        data["boards"][0]["runtime_root"] = "state"
    inventory.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        installer.install(source, inventory, board_root, enable=True)
    assert not (home / ".config").exists()


def test_missing_coder_fails_before_enabling_service(inputs, monkeypatch):
    source, inventory, board_root, home = inputs
    monkeypatch.setattr(installer.shutil, "which", lambda name: None)
    with pytest.raises(ValueError, match="executable is required"):
        installer.install(source, inventory, board_root, enable=True)
    assert not (home / ".config").exists()


def test_existing_service_ensure_stays_direct(inputs):
    source, inventory, board_root, _home = inputs
    data = json.loads(inventory.read_text())
    argv = ["systemctl", "--user", "start", "board.service"]
    data["boards"][0]["ensure_argv"] = argv
    inventory.write_text(json.dumps(data))
    result = installer.install(source, inventory, board_root, enable=False)
    config = json.loads(Path(result["config"]).read_text())
    assert config["boards"][0]["ensure"]["argv"] == argv
    assert result["ensure_services"] == []
