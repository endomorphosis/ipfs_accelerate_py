"""Read-only closeout admission preserves sealed task population and owner birth."""

import json
import sys
from types import SimpleNamespace

import pytest

from test.api.semantic_refactoring.test_bootstrap_controls import _materializer


@pytest.mark.parametrize("bad_seal", [False, True])
def test_start_binds_readonly_population_and_never_rewrites_authority(
    tmp_path, monkeypatch, bad_seal
):
    m = _materializer()
    monkeypatch.setattr(m, "_assert_start_not_held", lambda _: None)
    receipt = {
        "plan_root_cid": "plan:sealed",
        "repository_tree_id": "tree:sealed",
        "database_task_source_receipt": {"task_cids": ["task:one"]},
    }
    receipt["bootstrap_receipt_id"] = m._identity(receipt)
    if bad_seal:
        receipt["bootstrap_receipt_id"] = "wrong"
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(receipt))
    calls = {}

    class Server:
        def start(self):
            return "identity"

        def ready(self):
            return {"ready": True}

        def bind_database_status_scope(self, **binding):
            calls["binding"] = binding

        def recover_legacy_completion_projections(self):
            calls["recovered"] = True
            return []

        def publish_spar_source_forest(self):
            assert calls.get("recovered") is True
            calls["kit_persistence"] = True
            return {"admitted": False, "completion_authority": False}

        def stop(self):
            calls["stopped"] = True

    server = Server()
    monkeypatch.setattr(
        m,
        "_build_state_owner",
        lambda _: (server, {"bootstrap_receipt": path}, "program"),
    )
    monkeypatch.setattr(
        m,
        "_load_config",
        lambda _: (SimpleNamespace(board_namespace="board:sealed"), {}),
    )
    profile = object()
    monkeypatch.setattr(m, "_native_closeout_profile", lambda *a: profile)
    before = path.read_bytes()
    if bad_seal:
        with pytest.raises(m.OperatorError):
            m._start_state_owner(tmp_path / "config")
        assert calls == {"stopped": True}
    else:
        m._start_state_owner(tmp_path / "config")
        assert calls["kit_persistence"] is True
        assert calls["binding"] == {
            "board_namespace": "board:sealed",
            "plan_root_cid": "plan:sealed",
            "repository_tree_id": "tree:sealed",
            "task_cids": ["task:one"],
            "closeout_profile": profile,
        }
    assert path.read_bytes() == before


@pytest.mark.parametrize("failure", ["", "owner", "truncated"])
def test_native_status_preserves_unsettled_goals_and_owner_identity(
    tmp_path, monkeypatch, failure
):
    m = _materializer()
    identity = {
        "server_id": "owner:one",
        "process_birth_id": "birth:one",
        "store_id": "control.duckdb",
        "database_uuid": "uuid:one",
        "generation": 2,
        "fence_epoch": 2,
    }
    tasks = [
        {
            "task_cid": "task:one",
            "task_alias": "SPAR-050",
            "status": "todo",
            "revision": 1,
        }
    ]
    goals = [{"goal_alias": "SPAR-G000", "status": "active"}]
    snapshot = {
        "completion_snapshot": {"completion_projection": {"task_states": tasks}},
        "closeout_facts": {
            "truncated": failure == "truncated",
            "relations": {
                "tasks": {"available": True, "rows": tasks},
                "goals": {"available": True, "rows": goals},
                "leases": {"available": True, "rows": []},
            },
        },
    }
    paths = {
        "owner": tmp_path,
        "database": tmp_path / "control.duckdb",
        "bootstrap_receipt": tmp_path / "bootstrap.json",
    }
    (tmp_path / "typed-state-owner.token").write_text("local-read-token")

    class Client:
        def __init__(self, **kwargs):
            self.identity = {**identity, "generation": 9 if failure == "owner" else 2}

        def completion_closeout_snapshot(self, cids):
            assert cids == ["task:one"]
            return snapshot

        def close(self):
            pass

    module = SimpleNamespace(
        STATUS_BOOTSTRAP_CLIENT_ID="reader",
        TYPED_STATE_OWNER_SOCKET_FILENAME="owner.sock",
        TYPED_STATE_OWNER_TOKEN_FILENAME="typed-state-owner.token",
        TypedStateOwnerConnection=Client,
        compact_default_owner_socket_path=lambda *a, **k: tmp_path / "owner.sock",
    )
    monkeypatch.setitem(
        sys.modules,
        "ipfs_accelerate_py.agent_supervisor.task_sources.typed_state_owner",
        module,
    )
    board = SimpleNamespace(
        board_namespace="board:sealed",
        resolved_database_program=lambda: SimpleNamespace(store_id="control.duckdb"),
    )
    monkeypatch.setattr(
        m, "_load_config", lambda _: (board, {"initial_projection": {"goal_count": 32}})
    )
    monkeypatch.setattr(m, "_runtime_paths", lambda _: paths)
    monkeypatch.setattr(m, "_owner_liveness", lambda _: "alive")
    monkeypatch.setattr(
        m,
        "_json_object",
        lambda path: (
            {"database_task_source_receipt": {"task_cids": ["task:one"]}}
            if path == paths["bootstrap_receipt"]
            else {"lifecycle": "ready", "identity": identity}
        ),
    )
    if failure:
        with pytest.raises(m.OperatorError):
            m.authoritative_status(tmp_path / "config")
    else:
        result = m.authoritative_status(tmp_path / "config")
        assert result["tasks"] == tasks
        assert json.loads(result["control"]["goals_json"]) == goals
        assert result["required_goal_count"] == 32
        assert result["completion_authority"] is False


@pytest.mark.parametrize("name", ["HOLD", "OPERATOR_STOP", "watchdog.disabled", "watchdog.hold"])
def test_native_start_honors_operator_hold_before_opening_authority(tmp_path, monkeypatch, name):
    m = _materializer()
    board = SimpleNamespace(runtime_paths={"root": "runtime"}, path=lambda _: tmp_path)
    (tmp_path / name).write_text("owned operator hold")
    monkeypatch.setattr(m, "_load_config", lambda _: (board, {}))
    monkeypatch.setattr(m, "_build_state_owner", lambda _: pytest.fail("held startup opened authority"))
    with pytest.raises(m.OperatorError, match="startup held"):
        m._start_state_owner(tmp_path / "config")
    assert (tmp_path / name).read_text() == "owned operator hold"


def test_native_start_without_operator_hold_is_admitted(tmp_path):
    m = _materializer()
    m._assert_start_not_held(SimpleNamespace(runtime_paths={"root": "runtime"}, path=lambda _: tmp_path))


def test_native_closeout_honors_hold_without_reading_state(tmp_path, monkeypatch):
    m = _materializer()
    board = SimpleNamespace(runtime_paths={"root": "runtime"}, path=lambda _: tmp_path)
    (tmp_path / "OPERATOR_STOP").write_text("operator")
    monkeypatch.setattr(m, "authoritative_status", lambda _: pytest.fail("read while held"))
    healthy = SimpleNamespace(failure="")
    assert m._retain_closeout_owner(tmp_path / "config", board=board,
                                   broker=healthy, monitor=healthy) == "stopped"


def test_native_closeout_signal_stops_and_restores_handlers(tmp_path, monkeypatch):
    import signal
    m = _materializer()
    board = SimpleNamespace(runtime_paths={"root": "runtime"}, path=lambda _: tmp_path)
    previous = {sig: signal.getsignal(sig) for sig in (signal.SIGTERM, signal.SIGINT)}
    def observe(_):
        signal.raise_signal(signal.SIGTERM)
        return {"completion_authority": False, "task_count": 51}
    monkeypatch.setattr(m, "authoritative_status", observe)
    healthy = SimpleNamespace(failure="")
    assert m._retain_closeout_owner(tmp_path / "config", board=board,
                                   broker=healthy, monitor=healthy) == "stopped"
    assert {sig: signal.getsignal(sig) for sig in previous} == previous


def test_native_closeout_owner_fault_propagates(tmp_path, monkeypatch):
    m = _materializer()
    board = SimpleNamespace(runtime_paths={"root": "runtime"}, path=lambda _: tmp_path)
    monkeypatch.setattr(m, "authoritative_status", lambda _: pytest.fail("read after fault"))
    with pytest.raises(m.OperatorError, match="monitor failed during closeout"):
        m._retain_closeout_owner(tmp_path / "config", board=board,
                                broker=SimpleNamespace(failure="lost fence"),
                                monitor=SimpleNamespace(failure=""))
