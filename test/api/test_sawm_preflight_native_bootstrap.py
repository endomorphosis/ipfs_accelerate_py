"""The public SAWM client commands must hold the admitted native image."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def operator():
    path = ROOT / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py"
    spec = importlib.util.spec_from_file_location("sawm_client_bootstrap_test", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    yield module
    sys.modules.pop(spec.name, None)


@pytest.fixture
def admitted_native(operator, monkeypatch):
    from ipfs_accelerate_py import agent_implementation_route as route
    from ipfs_accelerate_py.agent_supervisor.runtime import (
        configured_board_scheduler as scheduler,
    )

    for alias in ("duckdb", "_duckdb"):
        monkeypatch.delitem(sys.modules, alias, raising=False)
    for name in tuple(os.environ):
        if name.startswith("LD_"):
            monkeypatch.delenv(name)
    descriptor = os.memfd_create("sawm-test-native-lifetime", os.MFD_CLOEXEC)
    module = SimpleNamespace(
        __file__=f"/proc/self/fd/{descriptor}", __version__="1.5.5"
    )
    launch = SimpleNamespace(
        descriptor=SimpleNamespace(descriptor=descriptor),
        pin=SimpleNamespace(distribution_version="1.5.5"),
    )
    events = []
    board = object()
    monkeypatch.setattr(operator, "_config", lambda path: {})
    monkeypatch.setattr(operator, "_delegate_repair_service_launch", lambda args: None)
    monkeypatch.setattr(scheduler, "load_configured_board", lambda *a, **k: board)
    monkeypatch.setattr(
        scheduler, "_configured_board_dependency_seal_snapshot", lambda b: b
    )
    monkeypatch.setattr(
        scheduler, "_seal_configured_board_native_dependency", lambda *a, **k: launch
    )

    def preload(value):
        assert value is launch
        events.append("preload")
        for alias in ("duckdb", "_duckdb"):
            monkeypatch.setitem(sys.modules, alias, module)
        return module

    def verify(value):
        assert value is launch
        os.fstat(descriptor)
        events.append("verify")
        return module.__file__

    def held():
        assert sys.modules.get("duckdb") is module
        assert sys.modules.get("_duckdb") is module
        os.fstat(descriptor)

    monkeypatch.setattr(route, "preload_agent_supervisor_native_dependency", preload)
    monkeypatch.setattr(
        route, "verify_agent_supervisor_native_dependency_sealed_fd", verify
    )
    yield SimpleNamespace(
        held=held, events=events, descriptor=descriptor, scheduler=scheduler
    )
    try:
        os.close(descriptor)
    except OSError:
        pass


@pytest.mark.parametrize("command", ["preflight", "dry-run", "launch"])
@pytest.mark.parametrize("outcome", ["accepted", "rejected", "raised", "interrupted"])
def test_public_client_holds_native_through_preflight_scheduler_and_cleanup(
    operator, admitted_native, monkeypatch, command, outcome, capsys
):
    native = admitted_native
    events = native.events

    class Interrupted(BaseException):
        pass

    class Transaction:
        state = "begun"

        def commit(self):
            native.held()
            self.state = "committed"
            events.append("commit")
            return {"retired": True}

        def rollback(self):
            native.held()
            self.state = "rolled_back"
            events.append("rollback")
            return {"rolled_back": True}

    transaction = Transaction()

    def preflight(config, **options):
        native.held()
        events.append("preflight")
        real_launch = command == "launch"
        assert options["probe_provider"] is real_launch
        assert options["retire_provider_token_handoff"] is real_launch
        if real_launch:
            options["token_handoff_transaction_sink"](transaction)
        return {"quack": {}}

    def schedule(arguments, **options):
        native.held()
        events.append("scheduler")
        if outcome == "raised":
            raise RuntimeError("scheduler rejected test launch")
        if outcome == "interrupted":
            raise Interrupted("test interruption")
        return 0 if outcome == "accepted" else 2

    monkeypatch.setattr(operator, "_live_preflight", preflight)
    monkeypatch.setattr(native.scheduler, "main", schedule)
    arguments = [command]
    if command == "launch":
        arguments.append("--foreground")
    if outcome == "interrupted":
        with pytest.raises(Interrupted):
            operator.main(arguments)
        result = 2
    else:
        result = operator.main(arguments)
    assert events[:3] == ["preload", "verify", "preflight"]
    assert "scheduler" in events
    assert result == (0 if outcome == "accepted" else 2)
    assert events[-1] == "verify"
    if command == "launch":
        assert transaction.state == (
            "committed" if outcome == "accepted" else "rolled_back"
        )
    with pytest.raises(OSError):
        os.fstat(native.descriptor)
    if outcome == "raised":
        assert json.loads(capsys.readouterr().out)["valid"] is False


@pytest.mark.parametrize("denial", ["ambient", "loader", "seal"])
def test_public_preflight_native_denial_prevents_admission_and_dispatch(
    operator, admitted_native, monkeypatch, denial, capsys
):
    native = admitted_native
    reached = []
    monkeypatch.setattr(
        operator, "_live_preflight", lambda *a, **k: reached.append("preflight") or {}
    )
    monkeypatch.setattr(
        native.scheduler, "main", lambda *a, **k: reached.append("scheduler") or 0
    )
    if denial == "ambient":
        monkeypatch.setitem(sys.modules, "duckdb", object())
    elif denial == "loader":
        monkeypatch.setenv("LD_LIBRARY_PATH", "/untrusted-test")
    else:

        def refuse(*args, **kwargs):
            raise ValueError("test seal mismatch")

        monkeypatch.setattr(
            native.scheduler, "_seal_configured_board_native_dependency", refuse
        )
    assert operator.main(["preflight"]) == 2
    assert reached == []
    assert json.loads(capsys.readouterr().out)["valid"] is False


def test_live_check_uses_same_native_preflight_scope(
    operator, admitted_native, monkeypatch
):
    from ipfs_accelerate_py.agent_supervisor.task_sources import duckdb_state

    native = admitted_native
    monkeypatch.setattr(
        operator,
        "_config",
        lambda p: {"database_program": {"store_id": "disposable.duckdb"}},
    )
    monkeypatch.setattr(
        operator,
        "_materializer",
        lambda: SimpleNamespace(build_population=lambda root: {}),
    )
    monkeypatch.setattr(
        duckdb_state,
        "discover_live_quack_endpoint",
        lambda store: SimpleNamespace(uri="quack:127.0.0.1:1"),
    )

    def preflight(config, **options):
        native.held()
        native.events.append("preflight")
        assert options == {"probe_provider": False}
        return {"valid": True}

    monkeypatch.setattr(operator, "_live_preflight", preflight)
    assert operator.main(["check"]) == 0
    assert native.events == ["preload", "verify", "preflight", "verify"]
    with pytest.raises(OSError):
        os.fstat(native.descriptor)


@pytest.mark.parametrize("failed_validator", ["dependencies", "board"])
def test_actual_preflight_still_requires_both_native_validators(
    operator, admitted_native, monkeypatch, failed_validator, capsys
):
    native = admitted_native
    calls = []

    def validate(path, function):
        native.held()
        calls.append(path)
        return {"valid": failed_validator not in path}

    def forbidden(*args, **kwargs):
        pytest.fail("invalid board must not dispatch or construct authority")

    monkeypatch.setattr(operator, "_validator", validate)
    monkeypatch.setattr(operator, "_materializer", forbidden)
    monkeypatch.setattr(native.scheduler, "main", forbidden)
    assert operator.main(["preflight"]) == 2
    assert len(calls) == 2
    assert (
        "sealed dependency or board validation failed"
        in json.loads(capsys.readouterr().out)["error"]
    )
    with pytest.raises(OSError):
        os.fstat(native.descriptor)


def test_isolated_public_preflight_loads_actual_admitted_native_and_quack(tmp_path):
    """Only the admission continuation is a fixture; native sealing/SQL are real.

    No owner or live database is touched. A separate isolated process uses the
    committed board's actual native authorization and a disposable local store.
    """
    probe = r"""
import importlib.util, json, os, socket, sys
from pathlib import Path
root, temporary = map(Path, sys.argv[1:])
spec = importlib.util.spec_from_file_location("sawm_disposable_client", root / "scripts/ops/agent_supervisor/semantic_addressed_world_model.py")
operator = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = operator
spec.loader.exec_module(operator)
scheduler = operator._configured_board_scheduler_runtime()
observed = {}
def client_admission(config, **options):
    import duckdb
    from ipfs_accelerate_py.agent_supervisor.task_sources.duckdb_state import open_quack_transport_connection
    path = temporary / "disposable.duckdb"
    descriptor = int(duckdb.__file__.removeprefix("/proc/self/fd/"))
    os.fstat(descriptor)
    observed.update(descriptor=descriptor, version=duckdb.__version__)
    with duckdb.connect(str(path)) as created:
        created.execute("CREATE TABLE tasks (value INTEGER)")
        created.execute("INSERT INTO tasks VALUES (17)")
    transport = operator._SawmQuackTransport(config["quack_owner"])
    transport.prepare_extension_custody()
    replica = transport._open_replica_connection(path)
    with socket.socket() as address:
        address.bind(("127.0.0.1", 0))
        port = address.getsockname()[1]
    uri = "quack:127.0.0.1:" + str(port)
    try:
        replica.execute("SELECT * FROM quack_serve(?, token := ?, allow_other_hostname := false, disable_ssl := true)", [uri, "disposable-preflight-token"])
        client = open_quack_transport_connection(uri, token="disposable-preflight-token")
        try:
            observed["row"] = client.execute("SELECT value FROM tasks").fetchone()[0]
        finally:
            client.close()
    finally:
        replica.execute("SELECT * FROM quack_stop(?)", [uri])
        replica.close()
        transport._remove_extension_projection()
    return {"valid": True, "quack": {}}
def schedule(args):
    os.fstat(observed["descriptor"])
    observed["scheduler_fd_held"] = True
    return 0
operator._live_preflight = client_admission
scheduler.main = schedule
result = operator.main(["preflight"])
if result:
    raise SystemExit(result)
try:
    os.fstat(observed["descriptor"])
except OSError:
    observed["descriptor_closed"] = True
print(json.dumps(observed, sort_keys=True))
"""
    completed = subprocess.run(
        ["/usr/bin/python3", "-I", "-S", "-B", "-c", probe, str(ROOT), str(tmp_path)],
        cwd=tmp_path,
        env={"PATH": "/usr/bin:/bin", "LANG": "C.UTF-8", "HOME": str(Path.home())},
        capture_output=True,
        text=True,
        timeout=90,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    observed = json.loads(completed.stdout.splitlines()[-1])
    assert observed["row"] == 17
    assert observed["version"] == "1.5.5"
    assert observed["scheduler_fd_held"] is True
    assert observed["descriptor_closed"] is True
