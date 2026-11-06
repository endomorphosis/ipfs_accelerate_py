import json
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.rescue.sealed_board_supervisor_launch import (
    hold_retain_owner_until_operator_stop,
    install_overlay,
    overlay_module,
    overlay_supervise_exit_code,
    prefer_sealed_scripts,
)


def test_source_root_insert_cannot_hide_overlay(tmp_path, monkeypatch):
    overlay = tmp_path / "overlay"
    source = tmp_path / "sealed"
    overlay.mkdir()
    source.mkdir()
    monkeypatch.setattr("sys.path", ["/usr/lib/python3"])
    install_overlay(str(overlay), str(source))
    import sys

    assert sys.path[0] == str(overlay.resolve())
    sys.path.insert(0, str(source.resolve()))
    assert sys.path[0] == str(overlay.resolve())
    assert sys.path[1] == str(source.resolve())


def test_overlay_module_keeps_relative_imports_on_sealed_package():
    import ipfs_accelerate_py.agent_supervisor.semantic_state.spar_accepted_root as loaded

    path = Path(loaded.__file__).resolve()
    overlay_module(
        "ipfs_accelerate_py.agent_supervisor.semantic_state.spar_accepted_root",
        str(path),
        package="ipfs_accelerate_py.agent_supervisor.semantic_state",
    )
    import ipfs_accelerate_py.agent_supervisor.semantic_state.spar_accepted_root as again

    assert hasattr(again, "admit_current_bound_clause_records")
    assert again.REQUIRED_CLAUSES == loaded.REQUIRED_CLAUSES


def test_prefer_sealed_scripts_drops_kit_shadow(tmp_path):
    import sys
    import types

    sealed = tmp_path / "sealed" / "scripts"
    sealed.mkdir(parents=True)
    (sealed / "__init__.py").write_text("")
    shadow = types.ModuleType("scripts")
    shadow.__file__ = str(tmp_path / "ipfs_kit_py" / "scripts" / "__init__.py")
    shadow.__path__ = [str(tmp_path / "ipfs_kit_py" / "scripts")]
    sys.modules["scripts"] = shadow
    prefer_sealed_scripts(str(tmp_path / "sealed"))
    assert "scripts" not in sys.modules


def test_overlay_supervise_exit_restarts_when_owner_already_stopped():
    assert (
        overlay_supervise_exit_code(
            "/unused",
            ["supervise", "--implement"],
            0,
            owner_lifecycle="stopped",
        )
        == 1
    )


def test_overlay_supervise_exit_keeps_zero_when_owner_ready():
    assert (
        overlay_supervise_exit_code(
            "/unused",
            ["supervise", "--implement"],
            0,
            owner_lifecycle="ready",
        )
        == 0
    )


def test_hold_retain_owner_does_not_return_on_native_complete():
    import ipfs_accelerate_py.agent_supervisor.runtime.terminal_closeout as terminal

    original = terminal.retain_owner_for_closeout
    hold_retain_owner_until_operator_stop()
    calls = {"n": 0}

    def wait(_seconds: float) -> bool:
        calls["n"] += 1
        return calls["n"] >= 2

    result = terminal.retain_owner_for_closeout(
        observe=lambda: {"completion_authority": True, "complete": True},
        wait=wait,
        stopped=lambda: False,
        check_owner=lambda: None,
        output=lambda _m: None,
        interval_seconds=1,
    )
    terminal.retain_owner_for_closeout = original
    assert result == "stopped"
    assert calls["n"] >= 2


def test_overlay_supervise_exit_preserves_nonzero():
    assert (
        overlay_supervise_exit_code("/unused", ["supervise"], 2, owner_lifecycle="stopped")
        == 2
    )


def test_nested_extra_gate_insert_cannot_hide_overlay(tmp_path, monkeypatch):
    from ipfs_accelerate_py.agent_supervisor.rescue.sealed_board_supervisor_launch import (
        install_overlay,
    )

    overlay = tmp_path / "overlay"
    source = tmp_path / "sealed"
    nested = source / "external" / "ipfs_accelerate"
    overlay.mkdir()
    nested.mkdir(parents=True)
    monkeypatch.setattr("sys.path", ["/usr/lib/python3"])
    install_overlay(str(overlay), str(source))
    import sys

    sys.path.insert(0, str(nested.resolve()))
    assert sys.path[0] == str(overlay.resolve())
    assert sys.path[1] == str(nested.resolve())


def test_install_supervisor_heal_overlay_pins_quack_state_server():
    from ipfs_accelerate_py.agent_supervisor.rescue.sealed_board_supervisor_launch import (
        install_supervisor_heal_overlay,
    )
    import ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server as server

    overlay = Path(server.__file__).resolve().parents[3]
    install_supervisor_heal_overlay(str(overlay))
    import ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server as again

    assert callable(getattr(again.QuackStateServer, "_ensure_client_token_handoff"))
    assert callable(getattr(again.QuackStateServer, "_unstall_false_terminal_blocked"))
    import ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema as schema
    import ipfs_accelerate_py.agent_supervisor.todo_daemon.retained_attempt_fairness as fairness

    assert callable(getattr(schema, "install_datasets_authoritative_operational_schema"))
    assert hasattr(fairness, "RetainedAttemptFairness")


def test_sawm_overlay_supervise_uses_sealed_launch():
    from ipfs_accelerate_py.agent_supervisor.rescue.overlay_sys_path import overlay_root

    script = (
        Path(overlay_root())
        / "ipfs_accelerate_py/agent_supervisor/rescue/sawm_supervise_with_heal_overlay.sh"
    )
    text = script.read_text(encoding="utf-8")
    assert "SEALED=" in text
    assert '"${LAUNCH[@]}" quack-start' in text
    assert "owner_ready()" in text
    assert "live_owner_ready.py" in text
    assert "STATUS_JSON" not in text
    assert '"${SEALED[@]}" launch &' in text
    assert '"${LAUNCH[@]}" launch' not in text
    assert "lanes_attached" in text
    assert "retrying without stopping extra-gate" in text
    assert "client vault rearmed, retrying without stopping extra-gate" in text


def test_pin_sealed_sawm_observation_runtime_uses_checkout_file(tmp_path):
    import ipfs_accelerate_py.agent_supervisor.runtime.owner_status_observation as observation

    from ipfs_accelerate_py.agent_supervisor.rescue.sealed_board_supervisor_launch import (
        _is_sawm_source,
        pin_sealed_sawm_observation_runtime,
    )

    original = Path(observation.__file__).resolve()
    sealed = tmp_path / "semantic-addressed-world-model-r2"
    relative = Path("ipfs_accelerate_py/agent_supervisor/runtime")
    (sealed / relative).mkdir(parents=True)
    for name in ("owner_status_observation.py", "native_dispatch_drain.py"):
        (sealed / relative / name).write_text(
            (original.parent / name).read_text(encoding="utf-8"),
            encoding="utf-8",
        )
    try:
        pin_sealed_sawm_observation_runtime(str(sealed))
        import ipfs_accelerate_py.agent_supervisor.runtime.owner_status_observation as pinned

        assert Path(pinned.__file__).resolve() == (sealed / relative / "owner_status_observation.py").resolve()
        assert _is_sawm_source(
            str(sealed),
            ["scripts/ops/agent_supervisor/semantic_addressed_world_model.py", "quack-start"],
        )
    finally:
        pin_sealed_sawm_observation_runtime(str(original.parents[3]))


def test_live_owner_ready_skips_stale_first_run_status(tmp_path):
    import os

    from ipfs_accelerate_py.agent_supervisor.rescue.live_owner_ready import (
        find_ready_owner_status,
        owner_is_ready,
        state_root_for_ready_owner,
    )

    def write_status(run: str, *, lifecycle: str, pid: int) -> Path:
        path = (
            tmp_path
            / "data/agent_supervisor/semantic_addressed_world_model"
            / run
            / "quack-owner/quack-state-server.status.json"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        (path.parent.parent / "state" / "lane-0").mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({
            "lifecycle": lifecycle,
            "identity": {
                "status": lifecycle,
                "process_birth": {"pid": pid},
            },
        }))
        return path

    stale = write_status("run-r2-m10", lifecycle="stopped", pid=1)
    live = write_status("run-r2-m27", lifecycle="ready", pid=os.getpid())
    assert stale.exists()
    assert find_ready_owner_status(tmp_path) == live.resolve()
    assert owner_is_ready(tmp_path, os.getpid()) is True
    assert owner_is_ready(tmp_path, 1) is False
    assert state_root_for_ready_owner(tmp_path, quack_pid=os.getpid()) == (
        tmp_path
        / "data/agent_supervisor/semantic_addressed_world_model/run-r2-m27/state"
    ).resolve()


import pytest


def _cold_sawm_runtime_fixture(tmp_path, *, require_binding=False):
    import hashlib

    from ipfs_accelerate_py.agent_supervisor.rescue import sealed_board_supervisor_launch as launcher

    source = tmp_path / "semantic-addressed-world-model-r2"
    overlay = tmp_path / "compatibility-runtime"
    packages = ("ipfs_accelerate_py", "ipfs_accelerate_py/agent_supervisor",
                "ipfs_accelerate_py/agent_supervisor/task_sources",
                "ipfs_accelerate_py/agent_supervisor/runtime")
    modules = {
        "ipfs_accelerate_py/agent_implementation_route.py": "provider",
        "ipfs_accelerate_py/agent_supervisor/runtime/configured_board_scheduler.py": "scheduler",
        "ipfs_accelerate_py/agent_supervisor/runtime/quack_state_server.py": "owner",
        "ipfs_accelerate_py/agent_supervisor/task_sources/compatibility_neighbor.py": "neighbor",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_migrations.py": "migrations",
        "ipfs_accelerate_py/agent_supervisor/task_sources/control_plane_schema.py": "schema",
        "ipfs_accelerate_py/agent_supervisor/runtime/native_dispatch_drain.py": "drain",
        "ipfs_accelerate_py/agent_supervisor/runtime/owner_status_observation.py": "observation",
    }
    for root, label in ((source, "sealed"), (overlay, "overlay")):
        for relative in packages:
            directory = root / relative
            directory.mkdir(parents=True, exist_ok=True)
            (directory / "__init__.py").write_text("")
        for relative, role in modules.items():
            body = f"ORIGIN = {label + '-' + role!r}\n"
            if role in {"schema", "migrations"}:
                body = "from . import compatibility_neighbor\n" + body
            if label == "overlay" and role == "migrations":
                body += '''BOUND_RESOURCES = None
def bind_bundled_sql_resources(directory, expected_sha256):
    from pathlib import Path
    from hashlib import sha256
    global BOUND_RESOURCES
    assert {path.name for path in Path(directory).iterdir()} == set(expected_sha256)
    assert all(sha256((Path(directory) / name).read_bytes()).hexdigest() == digest
               for name, digest in expected_sha256.items())
    BOUND_RESOURCES = dict(expected_sha256)
'''
            if label == "overlay" and role == "schema" and require_binding:
                body += '''from . import control_plane_migrations
assert control_plane_migrations.BOUND_RESOURCES is not None, "schema loaded before resource binding"
'''
            (root / relative).write_text(body)
    sql_directory = overlay / "ipfs_accelerate_py/agent_supervisor/task_sources/sql"
    sql_directory.mkdir()
    resources = {}
    for name in launcher.SAWM_SQL_RESOURCES:
        raw = f"-- private authored resource {name}\nSELECT 1;\n".encode()
        (sql_directory / name).write_bytes(raw)
        resources[name] = hashlib.sha256(raw).hexdigest()
    manifest = overlay / "sawm-runtime-manifest.json"
    manifest.write_text(json.dumps({"schema": "sawm/schema-compatibility-runtime@1",
                                   "sql_resources": resources, "files": {}}, sort_keys=True))
    manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    operator = source / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py"
    operator.parent.mkdir(parents=True)
    operator.write_text('''import importlib, json, sys
preloaded_package_modules = sorted(name for name in sys.modules
    if name == "ipfs_accelerate_py" or name.startswith("ipfs_accelerate_py."))
import ipfs_accelerate_py
names = {
    "provider": "ipfs_accelerate_py.agent_implementation_route",
    "scheduler": "ipfs_accelerate_py.agent_supervisor.runtime.configured_board_scheduler",
    "owner": "ipfs_accelerate_py.agent_supervisor.runtime.quack_state_server",
    "migrations": "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_migrations",
    "schema": "ipfs_accelerate_py.agent_supervisor.task_sources.control_plane_schema",
    "drain": "ipfs_accelerate_py.agent_supervisor.runtime.native_dispatch_drain",
    "observation": "ipfs_accelerate_py.agent_supervisor.runtime.owner_status_observation",
}
loaded = {role: importlib.import_module(name) for role, name in names.items()}
print(json.dumps({"origins": {role: module.ORIGIN for role, module in loaded.items()},
    "files": {role: module.__file__ for role, module in loaded.items()},
    "neighbor": loaded["schema"].compatibility_neighbor.ORIGIN,
    "resources_bound": sorted(getattr(loaded["migrations"], "BOUND_RESOURCES", None) or {}),
    "preloaded_package_modules": preloaded_package_modules,
    "package_path": list(ipfs_accelerate_py.__path__),
    "path_class": type(sys.path).__name__, "first_path": sys.path[0]}))
''')
    return Path(launcher.__file__).resolve(), source, overlay, operator, manifest_sha256


@pytest.mark.parametrize("mode,config_args", [
    ("full", []), ("pin_only", []), ("sealed_runtime", []),
    ("sealed_runtime", ["--config", "quack-start"]),
    ("sealed_runtime", ["--config=private.json"]),
    ("sealed_runtime", ["--config", "first.json", "--config", "last.json"]),
])
def test_sealed_runtime_cold_mode_keeps_native_package_and_legacy_modes(tmp_path, mode, config_args):
    import subprocess
    import sys

    launcher, source, overlay, operator, manifest_sha256 = _cold_sawm_runtime_fixture(
        tmp_path, require_binding=mode == "sealed_runtime")
    flags = {"full": [], "pin_only": ["--pin-only"],
             "sealed_runtime": ["--sealed-runtime", "--runtime-manifest-sha256", manifest_sha256]}[mode]
    if mode != "sealed_runtime":
        (overlay / "sawm-runtime-manifest.json").unlink()
    result = subprocess.run([sys.executable, "-I", "-B", str(launcher),
        "--overlay", str(overlay), "--source-root", str(source), *flags, "--", str(operator),
        *config_args, "quack-start"],
        cwd=tmp_path, check=True, capture_output=True, text=True, timeout=10)
    observed = json.loads(result.stdout)
    native = "overlay" if mode == "full" else "sealed"
    assert {role: observed["origins"][role] for role in ("provider", "scheduler", "owner")} == {
        role: f"{native}-{role}" for role in ("provider", "scheduler", "owner")}
    assert observed["origins"]["migrations"] == "overlay-migrations"
    assert observed["origins"]["schema"] == (
        "sealed-schema" if mode == "pin_only" else "overlay-schema")
    assert observed["neighbor"] == f"{native}-neighbor"
    for role, name in (("drain", "native_dispatch_drain"),
                       ("observation", "owner_status_observation")):
        assert observed["origins"][role] == f"sealed-{role}"
        assert Path(observed["files"][role]) == (
            source / f"ipfs_accelerate_py/agent_supervisor/runtime/{name}.py")
    assert observed["package_path"] == [str((overlay if mode == "full" else source) / "ipfs_accelerate_py")]
    assert observed["path_class"] == ("_OverlayPath" if mode == "full" else "list")
    if mode == "sealed_runtime":
        assert observed["first_path"] == str(source)
        from ipfs_accelerate_py.agent_supervisor.rescue import sealed_board_supervisor_launch as module
        assert observed["resources_bound"] == sorted(module.SAWM_SQL_RESOURCES)
    else:
        assert observed["resources_bound"] == []


@pytest.mark.parametrize("operator_args", [
    ["launch"], ["status"], ["--help"], ["--config", "quack-start", "launch"],
    ["--config=quack-start", "launch"], ["quack-start", "--unknown"],
    ["--config"], ["--config", "--help", "quack-start"], ["--unknown", "quack-start"],
])
def test_sealed_runtime_clients_import_only_sealed_package_after_manifest_gate(tmp_path, operator_args):
    import subprocess
    import sys

    launcher, source, overlay, operator, manifest_sha256 = _cold_sawm_runtime_fixture(
        tmp_path, require_binding=True)
    for name in ("control_plane_migrations.py", "control_plane_schema.py"):
        (overlay / "ipfs_accelerate_py/agent_supervisor/task_sources" / name).write_text(
            'raise RuntimeError("client_crossed_compatibility_module_boundary")\n')
    result = subprocess.run([sys.executable, "-I", "-B", str(launcher),
        "--overlay", str(overlay), "--source-root", str(source), "--sealed-runtime",
        "--runtime-manifest-sha256", manifest_sha256, "--", str(operator), *operator_args],
        cwd=tmp_path, check=True, capture_output=True, text=True, timeout=10)
    observed = json.loads(result.stdout)
    assert observed["preloaded_package_modules"] == []
    assert observed["resources_bound"] == []
    assert observed["origins"] == {role: f"sealed-{role}" for role in (
        "provider", "scheduler", "owner", "migrations", "schema", "drain", "observation")}
    assert observed["neighbor"] == "sealed-neighbor"
    assert observed["package_path"] == [str(source / "ipfs_accelerate_py")]
    assert observed["path_class"] == "list" and observed["first_path"] == str(source)


@pytest.mark.parametrize("flags,other_operator,reason", [
    (["--pin-only", "--sealed-runtime"], False, "cannot be combined"),
    (["--sealed-runtime", "--pin-only"], False, "cannot be combined"),
    (["--sealed-runtime"], True, "requires the sealed SAWM operator"),
])
def test_sealed_runtime_invalid_modes_refuse_before_any_package_import(
        tmp_path, flags, other_operator, reason):
    import subprocess
    import sys

    launcher, source, overlay, operator, manifest_sha256 = _cold_sawm_runtime_fixture(tmp_path)
    for root in (source, overlay):
        (root / "ipfs_accelerate_py/__init__.py").write_text(
            'raise RuntimeError("package_imported_before_mode_validation")\n')
    if other_operator:
        operator = operator.with_name("other_operator.py")
    result = subprocess.run([sys.executable, "-I", "-B", str(launcher),
        "--overlay", str(overlay), "--source-root", str(source), *flags,
        "--runtime-manifest-sha256", manifest_sha256, "--", str(operator)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and reason in result.stderr
    assert "package_imported_before_mode_validation" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("fault", [
    "missing_pin", "wrong_pin", "invalid_pin", "missing_manifest", "foreign_schema",
    "foreign_resources", "pin_without_mode",
])
def test_sealed_runtime_manifest_and_external_pin_refuse_before_package_imports(tmp_path, fault):
    import hashlib
    import subprocess
    import sys

    launcher, source, overlay, operator, manifest_sha256 = _cold_sawm_runtime_fixture(tmp_path)
    manifest = overlay / "sawm-runtime-manifest.json"
    for root in (source, overlay):
        (root / "ipfs_accelerate_py/__init__.py").write_text(
            'raise RuntimeError("package_imported_before_manifest_validation")\n')
    flags = ["--sealed-runtime"]
    if fault == "wrong_pin":
        manifest_sha256 = "0" * 64
    elif fault == "invalid_pin":
        manifest_sha256 = "X" * 64
    elif fault == "missing_manifest":
        manifest.unlink()
    elif fault in {"foreign_schema", "foreign_resources"}:
        value = json.loads(manifest.read_bytes())
        if fault == "foreign_schema":
            value["schema"] = "foreign"
        else:
            value["sql_resources"]["9999_foreign.sql"] = "0" * 64
        manifest.write_text(json.dumps(value))
        manifest_sha256 = hashlib.sha256(manifest.read_bytes()).hexdigest()
    elif fault == "pin_without_mode":
        flags = []
    if fault != "missing_pin":
        flags.extend(["--runtime-manifest-sha256", manifest_sha256])
    result = subprocess.run([sys.executable, "-I", "-B", str(launcher),
        "--overlay", str(overlay), "--source-root", str(source), *flags, "--", str(operator)],
        cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode != 0 and "manifest" in result.stderr
    assert "package_imported_before_manifest_validation" not in result.stderr
    assert result.stdout == ""


@pytest.mark.parametrize("invalid_mode,digest", [
    ("--pin-only", ""), ("--unknown", ""), ("--sealed-runtime", ""),
    ("--sealed-runtime", "X" * 64), ("", "0" * 64),
])
def test_sealed_runtime_shell_rejects_invalid_mode_before_touching_source(tmp_path, invalid_mode, digest):
    import subprocess

    from ipfs_accelerate_py.agent_supervisor.rescue import sealed_board_supervisor_launch as launcher

    script = Path(launcher.__file__).with_name("sawm_supervise_with_heal_overlay.sh")
    result = subprocess.run(["/bin/bash", str(script), str(tmp_path / "absent-overlay"),
        str(tmp_path / "absent-source"), str(tmp_path / "absent-launcher.py"), invalid_mode, digest],
        cwd=tmp_path, capture_output=True, text=True, timeout=5)
    assert result.returncode == 2 and "usage:" in result.stderr
    assert "No such file" not in result.stderr
