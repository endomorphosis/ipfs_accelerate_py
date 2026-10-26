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


@pytest.mark.parametrize("standalone", [False, True])
def test_real_installed_probe_imports_its_dependencies_outside_source(inputs, standalone):
    _source, inventory, board_root, _home = inputs
    source = SCRIPT.parents[3]
    shadow = board_root / "ipfs_accelerate_py"
    shadow.mkdir()
    (shadow / "__init__.py").write_text("raise RuntimeError('live checkout shadow')\n")
    result = installer.install(source, inventory, board_root, enable=False)
    release = Path(result["release"])
    config = json.loads(Path(result["config"]).read_text())
    probe = config["boards"][0]["probe"]
    if standalone:
        argv = [sys.executable, "-P", str(release / "ipfs_accelerate_py/agent_supervisor/rescue/live_board_probe.py")]
    else:
        argv = probe["argv"][:4]
    process = subprocess.run([*argv, "--help"], cwd=board_root,
        env={**os.environ, **probe["env"]}, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert "--inventory" in process.stdout


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
    board.update(hold_files=[hold], launch_only_hold_files=[hold], cooldown_seconds=777, max_backoff_seconds=999,
                 max_ensure_attempts=4, publication={"board_id": "spar"},
                 diagnostic_handoff="monitoring-handoff.json",
                 storage_checks={"git_worktrees": [str(board_root)]})
    config["poll_seconds"] = 75
    config["repair_worker"]["retry_seconds"] = 1000
    config_path.write_text(json.dumps(config))
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    upgraded = installer.install(source, inventory, board_root, enable=True)
    current = json.loads(config_path.read_text())
    assert hold in current["boards"][0]["hold_files"]
    assert current["boards"][0]["cooldown_seconds"] == 777
    assert current["boards"][0]["launch_only_hold_files"] == [hold]
    assert current["boards"][0]["max_ensure_attempts"] == 4
    assert current["boards"][0]["publication"] == {"board_id": "spar"}
    assert current["boards"][0]["diagnostic_handoff"] == "monitoring-handoff.json"
    assert current["boards"][0]["storage_checks"] == {"git_worktrees": [str(board_root)]}
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


def test_real_installed_repair_imports_configured_handoff_loader(inputs):
    _source, inventory, board_root, _home = inputs
    result = installer.install(SCRIPT.parents[3], inventory, board_root, enable=False)
    release = Path(result["release"])
    code = ("from ipfs_accelerate_py.agent_supervisor.rescue.diagnostic_handoff "
            "import load_diagnostic_handoff; "
            "from ipfs_accelerate_py.agent_supervisor.rescue.storage_diagnostics "
            "import observe_storage; "
            "from pathlib import Path; "
            "assert load_diagnostic_handoff({}, Path('/unused'), {}, observed_at=0) == {}; "
            "assert observe_storage({})['handling'] == 'diagnostic_only'; "
            "print(load_diagnostic_handoff.__module__)")
    process = subprocess.run([sys.executable, "-P", "-c", code], cwd=board_root,
        env={**os.environ, "PYTHONPATH": str(release)}, capture_output=True, text=True, timeout=15)
    assert process.returncode == 0, process.stderr
    assert "diagnostic_handoff" in process.stdout


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


def test_job_can_stage_upgrade_without_restarting_its_dispatcher(inputs, monkeypatch):
    source, inventory, board_root, _home = inputs
    calls = []
    monkeypatch.setattr(installer.subprocess, "run", lambda argv, **kwargs: calls.append(argv))
    result = installer.install(source, inventory, board_root, enable=True, defer_repair_restart=True)
    config = json.loads(Path(result["config"]).read_text())
    assert config["runtime_release"] == result["release"]
    assert calls == [
        ["systemctl", "--user", "daemon-reload"],
        ["systemctl", "--user", "enable", *result["services"]],
        ["systemctl", "--user", "restart", "ipfs-taskboard-watchdog.service"],
        ["systemctl", "--user", "start", "ipfs-taskboard-repair.service"],
    ]


def test_external_owner_inventory_does_not_rewrite_native_unit(inputs):
    source, inventory, board_root, home = inputs
    result = installer.install(source, inventory, board_root, enable=False)
    unit = home / ".config/systemd/user/ipfs-taskboard-spar-ensure.service"
    unit.write_text("external owner unit\n")
    data = json.loads(inventory.read_text())
    data["boards"][0]["ensure_argv"] = []
    inventory.write_text(json.dumps(data))
    installer.install(source, inventory, board_root, enable=False)
    assert unit.read_text() == "external owner unit\n"
    assert "ensure" not in json.loads(Path(result["config"]).read_text())["boards"][0]


def test_missing_checkout_can_be_retained_only_under_existing_explicit_hold(inputs):
    source, inventory, board_root, _home = inputs
    data = json.loads(inventory.read_text())
    missing = board_root / 'lost-family'
    data['boards'][0]['cwd'] = str(missing)
    data['boards'][0]['config_path'] = str(missing / 'board.json')
    hold = board_root / 'external-incident.hold'
    data['boards'][0]['hold_paths'] = [str(hold)]
    inventory.write_text(json.dumps(data))
    with pytest.raises(ValueError, match='explicitly held'):
        installer.install(source, inventory, board_root, enable=False)
    hold.write_text('Preserve missing authority; no rematerialization')
    result = installer.install(source, inventory, board_root, enable=False)
    installed = json.loads(Path(result['config']).read_text())
    assert str(hold) in installed['boards'][0]['hold_files']
    assert not missing.exists()


def _enrollment_case(inputs):
    source, inventory, board_root, home = inputs
    original = json.loads(inventory.read_text())["boards"][0]
    ids = ["spar", "sawm", "aseh", "pctdd", "pcpr", "doep"]
    prior_inventory = {"schema": "operator-inventory@1", "operator_metadata": {"keep": True},
                       "boards": [dict(original, id=identifier) for identifier in ids]}
    inventory.write_text(json.dumps(prior_inventory))
    installed = installer.install(source, inventory, board_root, enable=False)
    config_path = Path(installed["config"])
    previous = json.loads(config_path.read_text())
    for board in previous["boards"]:
        board.update(probe={"argv": [sys.executable, f"/operator/{board['id']}/probe.py"],
                            "env": {"PYTHONPATH": f"/operator/{board['id']}:original-release"},
                            "timeout_seconds": 33},
                     repair={"argv": [sys.executable, "operator-repair.py"],
                             "env": {"PYTHONPATH": "/operator/repair-overlay"}, "timeout_seconds": 17},
                     ensure={"argv": ["systemctl", "--user", "start", f"{board['id']}.service"]},
                     publication={"source": board["id"]}, hold_files=[f"/operator/{board['id']}.hold"],
                     launch_only_hold_files=[f"/operator/{board['id']}.launch-hold"],
                     max_ensure_attempts=2, max_repair_attempts=7, cooldown_seconds=777,
                     operator_extension={"preserve": [board["id"], 9]})
    previous["operator_extension"] = {"preserve": "unknown top-level policy"}
    previous["poll_seconds"] = 91
    previous["repair_worker"].update(argv=[sys.executable, "operator-code-worker.py", "--model", "grok"],
                                     model="grok-operator-selection", max_attempts=2,
                                     env={"PYTHONPATH": "/operator/worker-overlay"})
    config_path.write_text(json.dumps(previous))
    history = home / ".local/state/ipfs-taskboard-watchdog/repairs/spar/history.json"
    history.parent.mkdir(parents=True)
    history.write_text('{"status":"running","attempts":3,"hold":"retain"}\n')
    # A new source generation should produce a new release only for the added
    # probe/enqueue bindings unless activation is explicitly requested.
    (source / "ipfs_accelerate_py/agent_supervisor/rescue/service_fleet_probe.py").write_text(
        "print('new service-fleet probe generation')\n")
    legal = board_root / "legal"
    legal.mkdir()
    config = legal / "board.json"
    config.write_text("{}")
    entry = {"id": "legal-ir", "cwd": str(legal), "config_path": str(config),
             "runtime_root": str(legal / "runtime"), "ensure_argv": [],
             "probe_kind": "service_fleet_report", "report_path": str(legal / "fleet-health.json"),
             "board_namespace": "legal-ir-autoencoder-v1", "report_max_age_seconds": 180,
             "expected_units": {"lane-0": "legal-ir-autoencoder-v1.service",
                                "lane-1": "legal-ir-autoencoder-v1-lane-1.service"}}
    incoming = dict(prior_inventory, boards=[*prior_inventory["boards"], entry])
    inventory.write_text(json.dumps(incoming))
    return {"source": source, "inventory": inventory, "home": home, "entry": entry,
            "previous": previous, "prior_inventory": prior_inventory,
            "config_path": config_path, "history": history}


def _home_files(home):
    return {str(path.relative_to(home)): path.read_bytes()
            for path in home.rglob("*") if path.is_file()}


@pytest.mark.parametrize("activate", [False, True])
def test_enroll_dry_run_has_no_home_or_service_mutations(inputs, monkeypatch, activate):
    case = _enrollment_case(inputs)
    before = _home_files(case["home"])
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: pytest.fail("service command"))
    monkeypatch.setattr(installer, "atomic_write", lambda *a, **k: pytest.fail("dry-run write"))
    plan = installer.enroll(case["source"], case["inventory"], dry_run=True,
                            activate_runtime=activate)
    assert not plan["applied"] and plan["dry_run"]
    assert plan["board_id"] == "legal-ir"
    assert plan["unrelated_board_ids_preserved"] == [b["id"] for b in case["previous"]["boards"]]
    assert plan["service_commands"] == plan["state_history_changes"] == []
    assert len(plan["files"]) == 2
    assert [change["append"]["path"] for change in plan["files"]] == ["/boards/-", "/boards/-"]
    assert len(plan["unit_changes"]) == (2 if activate else 0)
    assert plan["explicit_top_level_changes"] == (["runtime_release"] if activate else [])
    assert _home_files(case["home"]) == before
    assert not Path(plan["release"]).exists()


@pytest.mark.parametrize("activate", [False, True])
def test_enroll_preserves_all_six_boards_worker_policy_history_and_units(inputs, monkeypatch, activate):
    case = _enrollment_case(inputs)
    home = case["home"]
    units = home / ".config/systemd/user"
    units_before = {p.name: p.read_bytes() for p in units.iterdir() if p.is_file()}
    history_before = case["history"].read_bytes()
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: pytest.fail("service command"))
    result = installer.enroll(case["source"], case["inventory"], activate_runtime=activate)
    current = json.loads(case["config_path"].read_text())
    installed_inventory = json.loads(case["config_path"].with_name("inventory.json").read_text())
    assert result["applied"] and not result["dry_run"]
    assert current["boards"][:-1] == case["previous"]["boards"]
    assert installed_inventory["boards"][:-1] == case["prior_inventory"]["boards"]
    assert {k: v for k, v in installed_inventory.items() if k != "boards"} == {
        k: v for k, v in case["prior_inventory"].items() if k != "boards"}
    expected_top = {k: v for k, v in case["previous"].items() if k != "boards"}
    if activate:
        expected_top["runtime_release"] = result["release"]
    assert {k: v for k, v in current.items() if k != "boards"} == expected_top
    assert current["repair_worker"] == case["previous"]["repair_worker"]
    assert case["history"].read_bytes() == history_before
    units_after = {p.name: p.read_bytes() for p in units.iterdir() if p.is_file()}
    changed_units = {name for name in units_before if units_before[name] != units_after[name]}
    assert changed_units == ({"ipfs-taskboard-watchdog.service", "ipfs-taskboard-repair.service"}
                             if activate else set())
    assert set(units_after) == set(units_before)
    if activate:
        assert {Path(x["after"]["path"]).name for x in result["unit_changes"]} == changed_units
        for change in result["unit_changes"]:
            assert Path(change["after"]["path"]).read_text() == change["postimage_utf8"]
    board = current["boards"][-1]
    assert board["id"] == "legal-ir" and "ensure" not in board
    assert "publication" not in board and "completion_gate" not in board
    assert board["max_repair_attempts"] == 3
    assert board["service_fleet_binding"] == {
        "board_namespace": "legal-ir-autoencoder-v1", "max_age_seconds": 180,
        "lanes": case["entry"]["expected_units"]}
    assert board["probe"]["env"]["PYTHONPATH"] == result["release"]
    assert board["repair"]["env"]["PYTHONPATH"] == result["release"]
    argv = board["probe"]["argv"]
    assert "ipfs_accelerate_py.agent_supervisor.rescue.service_fleet_probe" in argv
    assert argv[-4:] == ["--expected-unit", "lane-0=legal-ir-autoencoder-v1.service",
                         "--expected-unit", "lane-1=legal-ir-autoencoder-v1-lane-1.service"]
    assert (Path(result["release"]) / "ipfs_accelerate_py/agent_supervisor/rescue/service_fleet_probe.py").is_file()


@pytest.mark.parametrize("change", ["existing_changed", "duplicate", "multiple_new", "probe_kind",
                                     "ensure", "unit", "age", "publication"])
def test_enroll_rejects_conflicts_and_unsupported_authority_before_any_home_write(inputs, change):
    case = _enrollment_case(inputs)
    incoming = json.loads(case["inventory"].read_text())
    entry = incoming["boards"][-1]
    if change == "existing_changed":
        incoming["boards"][0]["status_argv"] = ["changed-probe"]
    elif change == "duplicate":
        incoming["boards"].append(dict(entry))
    elif change == "multiple_new":
        incoming["boards"].append(dict(entry, id="another-board"))
    elif change == "probe_kind":
        entry["probe_kind"] = "unknown"
    elif change == "ensure":
        entry["ensure_argv"] = ["systemctl", "--user", "restart", "native.service"]
    elif change == "unit":
        entry["expected_units"]["lane-1"] = "*.service"
    elif change == "age":
        entry["report_max_age_seconds"] = True
    else:
        entry["publication"] = {"completion_gate": "invented"}
    case["inventory"].write_text(json.dumps(incoming))
    before = _home_files(case["home"])
    with pytest.raises(ValueError):
        installer.enroll(case["source"], case["inventory"])
    assert _home_files(case["home"]) == before


def test_enroll_refuses_missing_existing_fleet_without_creating_home_configuration(inputs):
    source, inventory, _board_root, home = inputs
    before = _home_files(home)
    with pytest.raises(ValueError, match="existing regular fleet"):
        installer.enroll(source, inventory, dry_run=True)
    assert _home_files(home) == before


def test_enroll_rechecks_preimages_after_release_preparation(inputs, monkeypatch):
    case = _enrollment_case(inputs)
    installed_inventory = case["config_path"].with_name("inventory.json")
    prior_inventory_bytes = installed_inventory.read_bytes()
    materialize = installer._materialize_enrollment_release
    foreign = dict(case["previous"], competing_operator_revision="preserve")
    foreign_bytes = json.dumps(foreign).encode()

    def competing_update(*args):
        materialize(*args)
        case["config_path"].write_bytes(foreign_bytes)

    monkeypatch.setattr(installer, "_materialize_enrollment_release", competing_update)
    with pytest.raises(ValueError, match="preimage changed"):
        installer.enroll(case["source"], case["inventory"])
    assert case["config_path"].read_bytes() == foreign_bytes
    assert installed_inventory.read_bytes() == prior_inventory_bytes


def test_enroll_cli_dry_run_does_not_enter_full_install_or_service_activation(inputs, monkeypatch, capsys):
    _source, inventory, _board_root, _home = inputs
    calls = []
    monkeypatch.setattr(installer, "enroll", lambda *a, **k: calls.append(k) or {"dry_run": True})
    monkeypatch.setattr(installer, "install", lambda *a, **k: pytest.fail("full installation"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--inventory", str(inventory),
                                     "--enroll-only", "--dry-run", "--activate-runtime"])
    assert installer.main() == 0
    assert calls == [{"dry_run": True, "activate_runtime": True}]
    assert json.loads(capsys.readouterr().out) == {"dry_run": True}


@pytest.mark.parametrize("fault", ["oversize", "duplicate_service"])
def test_service_enrollment_bounds_match_probe_contract(tmp_path, fault):
    units = {"lane-0": "legal-ir-autoencoder-v1.service",
             "lane-1": "legal-ir-autoencoder-v1-lane-1.service"}
    if fault == "oversize":
        units = {f"lane-{n}": f"test-{n}.service" for n in range(33)}
    else:
        units["lane-1"] = units["lane-0"]
    entry = dict(id="legal-ir", cwd=str(tmp_path), config_path=str(tmp_path/"config.json"),
                 runtime_root=str(tmp_path/"state"), probe_kind="service_fleet_report",
                 report_path=str(tmp_path/"health.json"), board_namespace="legal-ir-autoencoder-v1",
                 expected_units=units)
    with pytest.raises(ValueError, match="expected_units"):
        installer._service_fleet_enrollment(entry, tmp_path/"release", tmp_path/"fleet.json")


def _profile_case(inputs):
    case = _enrollment_case(inputs)
    case["inventory"].write_text(json.dumps({"schema": "service-diagnostic-inventory@1",
                                           "boards": [dict(case["entry"], publication=False)]}))
    repair = inputs[2] / "isolated-repair"
    repair.mkdir()
    case["repair_cwd"] = repair
    return case


@pytest.mark.parametrize("enable", [False, True])
def test_legal_ir_profile_dry_run_is_pure_and_never_changes_global_bindings(inputs, monkeypatch, enable):
    case = _profile_case(inputs)
    before = _home_files(case["home"])
    monkeypatch.setattr(installer, "atomic_write", lambda *a, **k: pytest.fail("dry-run write"))
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: pytest.fail("service command"))
    plan = installer.install_profile(case["source"], case["inventory"], case["repair_cwd"],
                                     profile="legal-ir", dry_run=True, enable=enable)
    assert not plan["applied"] and plan["dry_run"]
    assert plan["service_commands"] == [] and plan["manual_service_activation_required"]
    assert len(plan["files"]) == 4
    assert {Path(x["after"]["path"]).name for x in plan["files"]} == {
        "fleet.json", "inventory.json", "ipfs-taskboard-legal-ir-watchdog.service",
        "ipfs-taskboard-legal-ir-repair.service"}
    assert plan["config"] == str(case["home"] / ".config/ipfs-taskboard-watchdog/legal-ir/fleet.json")
    assert plan["state"] == str(case["home"] / ".local/state/ipfs-taskboard-watchdog/legal-ir")
    assert plan["repair_job_unit"] == "ipfs-taskboard-legal-ir-repair-job.service"
    assert _home_files(case["home"]) == before
    assert not Path(plan["release"]).exists()


@pytest.mark.parametrize("enable", [False, True])
def test_legal_ir_profile_stages_only_isolated_configuration_and_units(inputs, monkeypatch, enable):
    case = _profile_case(inputs)
    before = _home_files(case["home"])
    monkeypatch.setattr(installer.subprocess, "run", lambda *a, **k: pytest.fail("service command"))
    plan = installer.install_profile(case["source"], case["inventory"], case["repair_cwd"],
                                     profile="legal-ir", enable=enable)
    after = _home_files(case["home"])
    assert all(after[path] == data for path, data in before.items())
    config = json.loads(Path(plan["config"]).read_text())
    assert config["state_dir"] == plan["state"]
    assert config["runtime_release"] == plan["release"]
    assert len(config["boards"]) == 1
    board = config["boards"][0]
    assert board["service_fleet_binding"]["board_namespace"] == "legal-ir-autoencoder-v1"
    assert board["max_repair_attempts"] == 3 and "ensure" not in board
    assert "publication" not in board and "completion_gate" not in board
    expected_worker = dict(case["previous"]["repair_worker"], cwd=str(case["repair_cwd"]),
                           job_unit="ipfs-taskboard-legal-ir-repair-job.service")
    assert config["repair_worker"] == expected_worker
    assert config["repair_worker"]["model"] == "grok-operator-selection"
    for suffix in ("watchdog", "repair"):
        unit = case["home"] / ".config/systemd/user" / f"ipfs-taskboard-legal-ir-{suffix}.service"
        text = unit.read_text()
        assert str(Path(plan["config"])) in text
        assert f"IPFS legal-ir diagnostic fleet {suffix}" in text
    allowed = {str(Path(x["after"]["path"]).relative_to(case["home"])) for x in plan["files"]}
    release = str(Path(plan["release"]).relative_to(case["home"])) + "/"
    assert all(path in allowed or path.startswith(release) for path in set(after) - set(before))
    snapshot = _home_files(case["home"])
    monkeypatch.setattr(installer, "atomic_write", lambda *a, **k: pytest.fail("idempotent rewrite"))
    again = installer.install_profile(case["source"], case["inventory"], case["repair_cwd"],
                                      profile="legal-ir", enable=enable)
    assert again["existing_identical_profile"] and again["applied"]
    assert _home_files(case["home"]) == snapshot


@pytest.mark.parametrize("change", ["config", "unit", "partial"])
def test_legal_ir_profile_refuses_existing_mismatched_or_partial_installation(inputs, change):
    case = _profile_case(inputs)
    plan = installer.install_profile(case["source"], case["inventory"], case["repair_cwd"])
    config = Path(plan["config"])
    if change == "config":
        config.write_text('{"operator_revision":"retain"}')
    elif change == "unit":
        (case["home"] / ".config/systemd/user/ipfs-taskboard-legal-ir-repair.service").write_text(
            "External profile owner\n")
    else:
        config.unlink()
    before = _home_files(case["home"])
    with pytest.raises(ValueError, match="profile differs|partial existing profile"):
        installer.install_profile(case["source"], case["inventory"], case["repair_cwd"])
    assert _home_files(case["home"]) == before


@pytest.mark.parametrize("change", ["history", "ensure", "publication", "repair_cwd"])
def test_legal_ir_profile_refuses_unknown_history_or_authority_before_writing(inputs, change):
    case = _profile_case(inputs)
    if change == "history":
        history = case["home"] / ".local/state/ipfs-taskboard-watchdog/legal-ir/unknown.json"
        history.parent.mkdir(parents=True)
        history.write_text("retained prior incident\n")
    elif change == "repair_cwd":
        case["repair_cwd"] = case["repair_cwd"] / "missing"
    else:
        data = json.loads(case["inventory"].read_text())
        if change == "ensure":
            data["boards"][0]["ensure_argv"] = ["systemctl", "--user", "restart", "native.service"]
        else:
            data["boards"][0]["publication"] = {"completion_gate": "invented"}
        case["inventory"].write_text(json.dumps(data))
    before = _home_files(case["home"])
    with pytest.raises(ValueError):
        installer.install_profile(case["source"], case["inventory"], case["repair_cwd"])
    assert _home_files(case["home"]) == before


def test_legal_ir_profile_rechecks_copied_global_repair_policy_without_overwriting_it(inputs, monkeypatch):
    case = _profile_case(inputs)
    materialize = installer._materialize_enrollment_release
    foreign = dict(case["previous"], new_operator_policy="preserve")
    foreign_bytes = json.dumps(foreign).encode()

    def update_policy(*args):
        materialize(*args)
        case["config_path"].write_bytes(foreign_bytes)

    monkeypatch.setattr(installer, "_materialize_enrollment_release", update_policy)
    with pytest.raises(ValueError, match="global repair policy changed"):
        installer.install_profile(case["source"], case["inventory"], case["repair_cwd"])
    assert case["config_path"].read_bytes() == foreign_bytes
    assert not (case["home"] / ".config/ipfs-taskboard-watchdog/legal-ir").exists()


def test_legal_ir_profile_cli_plans_without_entering_global_install_or_enrollment(inputs, monkeypatch, capsys):
    _source, inventory, board_root, _home = inputs
    calls = []
    monkeypatch.setattr(installer, "install_profile", lambda *a, **k: calls.append(k) or {"profile": "legal-ir"})
    monkeypatch.setattr(installer, "install", lambda *a, **k: pytest.fail("global install"))
    monkeypatch.setattr(installer, "enroll", lambda *a, **k: pytest.fail("global enrollment"))
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), "--inventory", str(inventory), "--repair-cwd",
                                     str(board_root), "--profile", "legal-ir", "--dry-run", "--enable"])
    assert installer.main() == 0
    assert calls == [{"profile": "legal-ir", "dry_run": True, "enable": True}]
    assert json.loads(capsys.readouterr().out) == {"profile": "legal-ir"}
