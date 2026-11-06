"""A deleted overlay must not overwrite a working systemd runtime binding."""
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.rescue import fleet_heals as heals
from ipfs_accelerate_py.agent_supervisor.rescue.runtime_assets import schema_runtime_assets


def populate(root):
    assets = schema_runtime_assets(root, board_id="sawm")["missing_assets"]
    for relative in assets:
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("# pinned test asset\n")
    return assets


def test_runtime_preflight_requires_selected_assets(tmp_path):
    root = tmp_path / "runtime"
    assert schema_runtime_assets(root, board_id="sawm")["valid"] is False
    names = populate(root)
    assert len(names) == 5
    assert schema_runtime_assets(root, board_id="sawm")["valid"] is True
    (root / names[-1]).unlink()
    assert schema_runtime_assets(root, board_id="sawm")["missing_assets"] == [names[-1]]


@pytest.mark.parametrize("kind", ["symlink", "directory", "empty"])
def test_runtime_preflight_rejects_unusable_selected_asset(tmp_path, kind):
    root = tmp_path / "runtime"
    name = populate(root)[0]
    path = root / name
    path.unlink()
    if kind == "symlink":
        target = tmp_path / "external.py"
        target.write_text("print('unexpected')")
        path.symlink_to(target)
    elif kind == "directory":
        path.mkdir()
    else:
        path.touch()
    assert schema_runtime_assets(root, board_id="sawm")["missing_assets"] == [name]


def test_missing_runtime_preserves_existing_dropins(tmp_path, monkeypatch):
    unit = "test-retained-owner.service"
    user_dir = tmp_path / "units"
    dropins = user_dir / (unit + ".d")
    dropins.mkdir(parents=True)
    old = dropins / "81-schema-serve-in-place.conf"
    selected = dropins / "90-sawm-runtime-path.conf"
    old.write_text("[Service]\n# old observed path\n")
    selected.write_text("[Service]\n# accepted stable runtime\n")
    before = {p.name: p.read_bytes() for p in dropins.iterdir()}
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.setattr(heals, "_exclusive_owner_unit", lambda *a: unit)
    monkeypatch.setattr(heals, "_inventory_board", lambda *a: {"cwd": str(source)})
    monkeypatch.setattr(heals, "supervisor_overlay_root", lambda: str(tmp_path / "deleted"))
    def forbidden(*a, **kw):
        pytest.fail("missing runtime reached daemon reload")
    result = heals.bind_schema_serve_in_place({"id": "sawm"}, {},
        systemd_user_dir=user_dir, daemon_reload=forbidden)
    assert result["status"] == "skip"
    assert result["reason"] == "supervisor_runtime_assets_unavailable"
    assert {p.name: p.read_bytes() for p in dropins.iterdir()} == before
