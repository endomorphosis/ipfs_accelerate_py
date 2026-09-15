from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.rescue.sealed_board_supervisor_launch import (
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


def test_overlay_supervise_exit_preserves_nonzero():
    assert (
        overlay_supervise_exit_code("/unused", ["supervise"], 2, owner_lifecycle="stopped")
        == 2
    )
