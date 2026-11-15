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



def _populate_sawm_source(root):
    """Real files for availability only; no fixture code is imported or run."""
    names = (
        "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
        "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
        "ipfs_accelerate_py/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/task_sources/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/__init__.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/native_dispatch_drain.py",
        "ipfs_accelerate_py/agent_supervisor/runtime/owner_status_observation.py",
    )
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        body = ("" if path.name == "__init__.py" else
                "{}\n" if path.suffix == ".json" else "# authored availability fixture\n")
        path.write_text(body)
    return names


def _sawm_binding_fixture(tmp_path, monkeypatch, *, existing=True):
    from ipfs_accelerate_py.agent_supervisor.runtime import supervisor_meta_index

    source = tmp_path / "source"
    source.mkdir()
    overlay = tmp_path / "runtime"
    populate(overlay)
    unit = "test-sawm-owner.service"
    user_dir = tmp_path / "units"
    if existing:
        dropins = user_dir / (unit + ".d")
        dropins.mkdir(parents=True)
        (dropins / "81-schema-serve-in-place.conf").write_text("# previous binding\n")
        (dropins / "90-sawm-runtime-path.conf").write_text("# retained sealed runtime\n")
    reloads, mirrors = [], []
    monkeypatch.setattr(heals, "_exclusive_owner_unit", lambda *args: unit)
    monkeypatch.setattr(heals, "_inventory_board", lambda *args: {"cwd": str(source)})
    monkeypatch.setattr(heals, "supervisor_overlay_root", lambda: str(overlay))
    # The existing applied/idempotent branch imports this exact mirror. Keep
    # its entire call in memory; the fixture never writes a live metadata DB.
    monkeypatch.setattr(supervisor_meta_index, "mirror_work_record",
                        lambda **fields: mirrors.append(fields))
    return source, overlay, unit, user_dir, reloads, mirrors


def _unit_snapshot(root):
    return {
        str(path.relative_to(root)): (
            path.read_bytes(), path.stat().st_ino, path.stat().st_mode,
            path.stat().st_mtime_ns, path.stat().st_ctime_ns,
        )
        for path in root.rglob("*") if path.is_file()
    } if root.exists() else {}


def test_sawm_data_only_source_refuses_without_publication_or_reload(tmp_path, monkeypatch):
    source, overlay, unit, user_dir, reloads, mirrors = _sawm_binding_fixture(
        tmp_path, monkeypatch)
    (source / "data" / "retained-generation").mkdir(parents=True)
    before = _unit_snapshot(user_dir)
    # These are precisely the old binder's two availability gates. Both pass
    # on a real residual-data-only tree with all five overlay assets present.
    assert source.is_dir()
    assert schema_runtime_assets(overlay, board_id="sawm")["valid"] is True
    result = heals.bind_schema_serve_in_place(
        {"id": "sawm", "cwd": str(source)}, {}, systemd_user_dir=user_dir,
        daemon_reload=lambda: reloads.append(True))
    assert result["status"] == "skip"
    assert result["reason"] == "supervisor_source_assets_unavailable"
    assert result["missing_assets"] == []
    assert "scripts/ops/agent_supervisor/semantic_addressed_world_model.py" in result["missing_source_assets"]
    assert "config/agent_supervisor_semantic_addressed_world_model_scheduler.json" in result["missing_source_assets"]
    assert _unit_snapshot(user_dir) == before
    assert reloads == mirrors == []


def test_sawm_source_refusal_does_not_create_unit_directory(tmp_path, monkeypatch):
    source, overlay, unit, user_dir, reloads, mirrors = _sawm_binding_fixture(
        tmp_path, monkeypatch, existing=False)
    (source / "data").mkdir()
    result = heals.bind_schema_serve_in_place(
        {"id": "sawm", "cwd": str(source)}, {}, systemd_user_dir=user_dir,
        daemon_reload=lambda: reloads.append(True))
    assert result["reason"] == "supervisor_source_assets_unavailable"
    assert not user_dir.exists()
    assert reloads == mirrors == []


def test_sawm_complete_source_publishes_only_selected_binding(tmp_path, monkeypatch):
    source, overlay, unit, user_dir, reloads, mirrors = _sawm_binding_fixture(
        tmp_path, monkeypatch)
    _populate_sawm_source(source)
    retained = user_dir / (unit + ".d") / "90-sawm-runtime-path.conf"
    retained_name = str(retained.relative_to(user_dir))
    retained_before = _unit_snapshot(user_dir)[retained_name]
    assets = schema_runtime_assets(overlay, board_id="sawm", source_root=source)
    assert assets["valid"] is True and assets["missing_source_assets"] == []
    assert assets["completion_authority"] is assets["owner_restart_authority"] is False
    result = heals.bind_schema_serve_in_place(
        {"id": "sawm", "cwd": str(source)}, {}, systemd_user_dir=user_dir,
        daemon_reload=lambda: reloads.append(True))
    assert result["status"] == "applied" and result["completion_authority"] is False
    binding = (user_dir / (unit + ".d") / "81-schema-serve-in-place.conf").read_text()
    assert str(overlay / "ipfs_accelerate_py/agent_supervisor/rescue/sawm_supervise_with_heal_overlay.sh") in binding
    assert str(source) in binding and "PYTHONPATH" not in binding
    assert _unit_snapshot(user_dir)[retained_name] == retained_before
    assert reloads == [True]
    assert len(mirrors) == 1
    assert mirrors[0]["record_kind"] == "schema_serve_in_place_binding"
    assert mirrors[0]["record_ref"] == unit


@pytest.mark.parametrize("name", (
    "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
    "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
    "ipfs_accelerate_py/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/task_sources/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/__init__.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/native_dispatch_drain.py",
    "ipfs_accelerate_py/agent_supervisor/runtime/owner_status_observation.py",
))
def test_sawm_source_requires_each_direct_native_entrypoint(tmp_path, name):
    source, overlay = tmp_path / "source", tmp_path / "runtime"
    _populate_sawm_source(source)
    populate(overlay)
    (source / name).unlink()
    result = schema_runtime_assets(overlay, board_id="sawm", source_root=source)
    assert result["valid"] is False
    assert result["reason"] == "supervisor_source_assets_unavailable"
    assert result["missing_source_assets"] == [name]


@pytest.mark.parametrize("kind", ("missing", "directory", "empty"))
def test_sawm_source_rejects_unusable_native_entrypoint(tmp_path, kind):
    source, overlay = tmp_path / "source", tmp_path / "runtime"
    _populate_sawm_source(source)
    populate(overlay)
    name = "ipfs_accelerate_py/agent_supervisor/runtime/native_dispatch_drain.py"
    path = source / name
    path.unlink()
    if kind == "directory":
        path.mkdir()
    elif kind == "empty":
        path.touch()
    result = schema_runtime_assets(overlay, board_id="sawm", source_root=source)
    assert result["valid"] is False
    assert result["missing_source_assets"] == [name]


def test_sawm_nested_package_cannot_replace_direct_native_entrypoints(tmp_path):
    source, overlay = tmp_path / "source", tmp_path / "runtime"
    _populate_sawm_source(source)
    populate(overlay)
    nested = source / "external" / "ipfs_accelerate"
    nested.mkdir(parents=True)
    (source / "ipfs_accelerate_py").rename(nested / "ipfs_accelerate_py")
    result = schema_runtime_assets(overlay, board_id="sawm", source_root=source)
    assert result["valid"] is False
    assert "ipfs_accelerate_py/agent_supervisor/runtime/native_dispatch_drain.py" in result["missing_source_assets"]
    assert "ipfs_accelerate_py/agent_supervisor/runtime/owner_status_observation.py" in result["missing_source_assets"]


@pytest.mark.parametrize("name", (
    "scripts/ops/agent_supervisor/semantic_addressed_world_model.py",
    "config/agent_supervisor_semantic_addressed_world_model_scheduler.json",
    "ipfs_accelerate_py/agent_supervisor/runtime/owner_status_observation.py",
))
def test_sawm_disappeared_source_preserves_already_bound_unit(tmp_path, monkeypatch, name):
    source, overlay, unit, user_dir, reloads, mirrors = _sawm_binding_fixture(
        tmp_path, monkeypatch)
    _populate_sawm_source(source)
    first = heals.bind_schema_serve_in_place(
        {"id": "sawm", "cwd": str(source)}, {}, systemd_user_dir=user_dir,
        daemon_reload=lambda: reloads.append(True))
    assert first["status"] == "applied"
    before = _unit_snapshot(user_dir)
    (source / name).unlink()
    result = heals.bind_schema_serve_in_place(
        {"id": "sawm", "cwd": str(source)}, {}, systemd_user_dir=user_dir,
        daemon_reload=lambda: reloads.append(True))
    assert result["reason"] == "supervisor_source_assets_unavailable"
    assert result["missing_source_assets"] == [name]
    assert _unit_snapshot(user_dir) == before
    assert reloads == [True] and len(mirrors) == 1


@pytest.mark.parametrize("kind", ("relative", "missing", "file"))
def test_sawm_source_root_requires_absolute_directory(tmp_path, kind):
    overlay = tmp_path / "runtime"
    populate(overlay)
    source = tmp_path / "source"
    if kind == "relative":
        source = Path("relative-source")
    elif kind == "file":
        source.write_text("not a source directory")
    result = schema_runtime_assets(overlay, board_id="sawm", source_root=source)
    assert result["valid"] is False
    assert result["missing_source_assets"] == ["source_root"]


def test_other_board_keeps_overlay_only_preflight(tmp_path):
    overlay, source = tmp_path / "runtime", tmp_path / "data-only"
    populate(overlay)
    source.mkdir()
    assert schema_runtime_assets(overlay, board_id="doep", source_root=source) == schema_runtime_assets(
        overlay, board_id="doep")
