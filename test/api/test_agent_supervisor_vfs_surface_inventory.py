from __future__ import annotations

import json
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.vfs_surface_inventory import (
    INVENTORY_AUTHORIZES_REPAIR,
    INVENTORY_IS_COMPLETION_EVIDENCE,
    INVENTORY_IS_CORRECTNESS_EVIDENCE,
    VARIANT_PRESENCE_IS_DEFECT,
    VARIANT_SUFFIXES,
    SurfaceClassification,
    SurfaceKind,
    VfsSurfaceInventoryError,
    assert_inventory_complete,
    discover_vfs_surface_paths,
    inventory_vfs_surfaces,
    publish_vfs_surface_inventory,
)


def _write(root: Path, relative_path: str, text: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _repository_fixture(root: Path) -> None:
    manager = """\
__all__ = ["VFSManager"]

class VFSManager:
    def mount(self, path):
        return path

def register_vfs(router):
    router.mount("/vfs", VFSManager())
"""
    _write(root, "pkg/vfs_manager.py", manager)
    _write(root, "pkg/vfs_manager.py.clean", manager)
    _write(
        root,
        "pkg/vfs_manager.py.full",
        """\
class VFSManager:
    def mount(self, path, options=None):
        return path, options
""",
    )
    _write(
        root,
        "pkg/ipfs_fsspec.py",
        "class IPFSFileSystem:\n    pass\n",
    )
    _write(
        root,
        "pkg/enhanced_fsspec.py",
        "class EnhancedFsspec:\n    pass\n",
    )
    _write(
        root,
        "pkg/bucket_vfs_manager.py",
        "class BucketVFSManager:\n    pass\n",
    )
    _write(
        root,
        "pkg/filesystem_journal.py",
        "class FilesystemJournal:\n    pass\n",
    )
    _write(
        root,
        "pkg/vfs_snapshot_tracker.py",
        "class VFSSnapshotTracker:\n    pass\n",
    )
    _write(
        root,
        "backends/vfs_adapter.py",
        "class VFSBackendAdapter:\n    pass\n",
    )
    _write(
        root,
        "handlers/vfs_handler.py",
        """\
@router.get("/vfs")
def vfs_handler():
    return None
""",
    )
    _write(root, "endpoints/vfs_endpoint.py", "def vfs_endpoint():\n    pass\n")
    _write(root, "controllers/vfs_controller.py", "class VFSController:\n    pass\n")
    _write(root, "tools/vfs_tool.py", "def vfs_tool():\n    pass\n")
    _write(root, "servers/vfs_server.py", "class VFSServer:\n    pass\n")
    _write(
        root,
        "sdk/mcp-sdk.full.js",
        "// compatibility proxy to avoid duplication\nexport const VFS = {};\n",
    )
    _write(
        root,
        "compat/vfs_compat.py",
        """\
\"\"\"Backwards compatibility wrapper for the VFS manager.\"\"\"
from pkg.vfs_manager import VFSManager
""",
    )
    _write(
        root,
        "generated/vfs_schema.py",
        "# Auto-generated; do not edit\nVFS_SCHEMA = {}\n",
    )
    _write(root, "archive/vfs_legacy.py", "class LegacyVFS:\n    pass\n")
    _write(root, "pkg/vfs_placeholder.py", "# placeholder\n")
    _write(root, "sdk/manifest.json", '{"description": "VFS integration manifest"}\n')
    _write(
        root,
        "tests/test_vfs_manager.py",
        """\
from pkg.vfs_manager import VFSManager

def test_mount():
    assert VFSManager().mount("/tmp") == "/tmp"
""",
    )
    _write(
        root,
        "docs/vfs.md",
        "The `pkg/vfs_manager.py` module exports `VFSManager`.\n",
    )
    _write(
        root,
        "app.py",
        """\
from pkg.vfs_manager import VFSManager

manager = VFSManager()
manager.mount("/data")
""",
    )
    for index, suffix in enumerate(VARIANT_SUFFIXES):
        _write(
            root,
            f"historical/surface_{index}.py{suffix}",
            f'"""VFS historical surface {suffix}."""\nVALUE = {index}\n',
        )


def test_inventory_classifies_every_surface_and_maps_relationships(
    tmp_path: Path,
) -> None:
    _repository_fixture(tmp_path)

    inventory = inventory_vfs_surfaces(tmp_path)
    by_path = inventory.by_path()
    observed_classifications = {
        classification
        for surface in inventory.surfaces
        for classification in surface.classifications
    }
    assert set(SurfaceClassification) <= observed_classifications

    observed_kinds = {
        kind for surface in inventory.surfaces for kind in surface.kinds
    }
    assert {
        SurfaceKind.FSSPEC,
        SurfaceKind.VFS_MANAGER,
        SurfaceKind.BUCKET_MANAGER,
        SurfaceKind.JOURNAL_WAL,
        SurfaceKind.VERSION_SNAPSHOT,
        SurfaceKind.BACKEND_ADAPTER,
        SurfaceKind.HANDLER,
        SurfaceKind.ENDPOINT,
        SurfaceKind.CONTROLLER,
        SurfaceKind.TOOL,
        SurfaceKind.SERVER,
        SurfaceKind.SDK_MANIFEST,
        SurfaceKind.EXPORT,
    } <= observed_kinds

    manager = by_path["pkg/vfs_manager.py"]
    assert manager.classification is SurfaceClassification.CANONICAL
    assert {"VFSManager", "mount", "register_vfs"} <= {
        definition.name for definition in manager.definitions
    }
    assert manager.registrations
    assert manager.exports == ("VFSManager",)
    assert {"app.py", "tests/test_vfs_manager.py"} <= set(manager.imported_by)
    assert {"app.py", "tests/test_vfs_manager.py"} <= set(manager.called_by)
    assert manager.tested_by == ("tests/test_vfs_manager.py",)
    assert manager.documented_by == ("docs/vfs.md",)

    duplicate = by_path["pkg/vfs_manager.py.clean"]
    assert duplicate.classification is SurfaceClassification.DUPLICATE
    assert duplicate.duplicate_of == "pkg/vfs_manager.py"

    shadow = by_path["pkg/vfs_manager.py.full"]
    assert shadow.classification is SurfaceClassification.SHADOW
    assert shadow.shadows == "pkg/vfs_manager.py"
    assert not shadow.imported_by
    assert not shadow.called_by

    contradictions = [
        item
        for item in inventory.contradictions
        if item.symbol == "mount" and "pkg/vfs_manager.py.full" in item.paths
    ]
    assert contradictions
    assert contradictions[0].disposition == "inconclusive"
    assert contradictions[0].is_defect is False


def test_variants_are_discovered_but_never_defects_by_presence(
    tmp_path: Path,
) -> None:
    _repository_fixture(tmp_path)

    discovered = set(discover_vfs_surface_paths(tmp_path))
    inventory = inventory_vfs_surfaces(tmp_path)
    variants = {
        surface.path: surface.variant_suffix
        for surface in inventory.surfaces
        if surface.path.startswith("historical/")
    }
    expected = {
        f"historical/surface_{index}.py{suffix}": suffix
        for index, suffix in enumerate(VARIANT_SUFFIXES)
    }
    assert variants == expected
    assert set(expected) <= discovered
    assert VARIANT_PRESENCE_IS_DEFECT is False
    assert any(item.code == "variant_observed" for item in inventory.diagnostics)
    assert not any(
        item.code == "variant_observed" and item.is_defect
        for item in inventory.diagnostics
    )
    assert inventory.by_path()["sdk/mcp-sdk.full.js"].variant_suffix == ".full"


def test_inventory_publishes_completeness_and_unexplained_diagnostics(
    tmp_path: Path,
) -> None:
    _repository_fixture(tmp_path)

    inventory = inventory_vfs_surfaces(tmp_path, scan_roots=["pkg", "sdk"])
    assert inventory.coverage_complete is False
    assert "sdk/manifest.json" in inventory.completeness.unexplained_paths
    assert any(
        item.code == "unexplained_surface_classification"
        and item.path == "sdk/manifest.json"
        and not item.explained
        for item in inventory.unexplained_surface_diagnostics
    )
    with pytest.raises(VfsSurfaceInventoryError) as exc_info:
        assert_inventory_complete(inventory)
    assert "inventory_incomplete" in exc_info.value.reason_codes

    output = tmp_path / "out" / "vfs-surface-inventory.json"
    assert publish_vfs_surface_inventory(inventory, output) == output
    published = json.loads(output.read_text(encoding="utf-8"))
    assert published["content_id"].startswith("sha256:")
    assert published["authority"] == {
        "is_completion_evidence": False,
        "is_correctness_evidence": False,
        "authorizes_repair": False,
        "variant_presence_is_defect": False,
    }
    assert published == inventory.to_record()
    assert INVENTORY_IS_COMPLETION_EVIDENCE is False
    assert INVENTORY_IS_CORRECTNESS_EVIDENCE is False
    assert INVENTORY_AUTHORIZES_REPAIR is False


def test_complete_inventory_can_be_asserted(tmp_path: Path) -> None:
    _write(
        tmp_path,
        "vfs_manager.py",
        "class VFSManager:\n    def mount(self, path):\n        return path\n",
    )

    inventory = inventory_vfs_surfaces(tmp_path)
    assert inventory.coverage_complete is True
    assert inventory.completeness.unexplained_paths == ()
    assert_inventory_complete(inventory)
