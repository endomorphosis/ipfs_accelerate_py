from ipfs_accelerate_py.agent_supervisor.rescue.sealed_board_supervisor_launch import (
    install_overlay,
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
