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
