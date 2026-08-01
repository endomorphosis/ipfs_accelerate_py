"""Tests for profile-driven repository surface inventory (LPR-021)."""

from __future__ import annotations

import json
import random
import re
import subprocess
from pathlib import Path

import pytest

from ipfs_accelerate_py.agent_supervisor.analysis.repository_surface_inventory import (
    INVENTORY_AUTHORIZES_REPAIR,
    INVENTORY_IS_COMPLETION_EVIDENCE,
    INVENTORY_IS_CORRECTNESS_EVIDENCE,
    REPOSITORY_SURFACE_INVENTORY_SCHEMA,
    VARIANT_PRESENCE_IS_DEFECT,
    SurfaceClassification,
    SurfaceInventoryError,
    SurfaceInventoryPolicy,
    SurfaceKindSpec,
    SurfaceSignal,
    SignalTarget,
    assert_inventory_complete,
    discover_inventory_schemas,
    discover_surface_paths,
    inventory_repository_surfaces,
    publish_surface_inventory,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
LOCK_PATH = REPO_ROOT / "config" / "agent_supervisor_vfs_generalization_sources.lock.json"
INVENTORY_MODULE = (
    REPO_ROOT
    / "ipfs_accelerate_py"
    / "agent_supervisor"
    / "analysis"
    / "repository_surface_inventory.py"
)
MAP_PATH = (
    REPO_ROOT
    / "docs"
    / "architecture"
    / "agent_supervisor"
    / "VFS_ASSURANCE_GENERALIZATION_MAP.md"
)

LOCKED_MODULE_BLOBS = {
    "ipfs_accelerate_py/agent_supervisor/vfs_surface_inventory.py": "76f34e1b9320e4bbc15706e4895c02af805af5e0",
    "ipfs_accelerate_py/agent_supervisor/vfs_contract_pack.py": "9acc4ceba42b8767f5b4e4b6ce7d4bc55893bcf2",
    "ipfs_accelerate_py/agent_supervisor/vfs_differential_harness.py": "8a6c8af69b6cbcb76a2b79a51f406d13e10947ce",
    "ipfs_accelerate_py/agent_supervisor/vfs_mcp_contract_checker.py": "26144a7b78c1bbbb94edc67ab13e2eab03850924",
    "ipfs_accelerate_py/agent_supervisor/vfs_symbolic_benchmark.py": "90023a09e9eb01ee454718f60fe758e33434c56b",
    "ipfs_accelerate_py/agent_supervisor/vfs_symbolic_pilot.py": "483ecaf622caa3c91d80d9710b63b1fd36fb8f90",
    "ipfs_accelerate_py/agent_supervisor/vfs_symbolic_rollout.py": "6a1ef7b87172aa413f81b37f0ba36954af774d40",
}
LOCKED_TEST_BLOBS = {
    "test/api/test_agent_supervisor_vfs_surface_inventory.py": "4c49d3d7ed7d0b495b006b89dcc12efde44639f7",
    "test/api/test_agent_supervisor_vfs_contract_pack.py": "f0115f4a81fc6e7ce3b9c4ddd1579a443a8448c7",
    "test/api/test_agent_supervisor_vfs_differential_harness.py": "c428d2857ce2caca76a5f9b43de0db6a1a5377eb",
    "test/api/test_agent_supervisor_vfs_mcp_contract_checker.py": "50310fadb3b8e6154a69ac875d68bd8ef40d925f",
    "test/api/test_agent_supervisor_vfs_symbolic_benchmark.py": "28e65a8e5c7d8653d5b767648cf60519516da70f",
    "test/api/test_agent_supervisor_vfs_symbolic_pilot.py": "1cb18acbedbd95c85eddcd0dff98fb69a5ba2b94",
    "test/api/test_vfs_symbolic_assurance_e2e.py": "854f5198482b9409c04b104977ead42ca976ffb7",
}

VARIANT_SUFFIXES = (
    ".fixed",
    ".full",
    ".new",
    ".clean",
    ".optimized",
    ".broken",
)

_FORBIDDEN_GENERIC = re.compile(
    r"(?i)\b(?:vfs|ipfs(?!_accelerate_py)|fsspec|swissknife|swiss[_-]?knife|"
    r"ipfs_kit|board[_-]?id|board[_-]?namespace)\b"
)


def _write(root: Path, relative_path: str, text: str) -> Path:
    path = root / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def build_vfs_equivalent_policy() -> SurfaceInventoryPolicy:
    """Reconstruct the locked VFS surface contract as an injectable profile.

    Domain literals belong in the profile, never in the generic engine module.
    """

    content_signal = SurfaceSignal(
        name="domain_content",
        pattern=r"""
            (?<![a-z0-9])(?:vfs|virtual[\s_-]*file[\s_-]*system|ipfs[\s_.-]*fsspec|
            enhanced[\s_.-]*fsspec|filesystem[\s_.-]*journal|fs[\s_.-]*journal|
            vfs[\s_.-]*(?:version|snapshot)|bucket[\s_.-]*vfs|
            (?:car|pin|storage|filesystem)?[\s_.-]*wal)(?![a-z0-9])
        """,
        target=SignalTarget.CONTENT,
    )
    path_signal = SurfaceSignal(
        name="domain_path",
        pattern=r"""
            (?:^|[./_-])(?:
              ipfs_fsspec|enhanced_fsspec|iroh_fsspec|iroh_vfs|
              vfs(?:_[a-z0-9]+)*|[a-z0-9]+_vfs(?:_[a-z0-9]+)*|
              filesystem_journal|fs_journal(?:_[a-z0-9]+)*|
              (?:car_|pin_|storage_|enhanced_)?wal(?:_[a-z0-9]+)*|
              vfs_version(?:_[a-z0-9]+)*|vfs_snapshot(?:_[a-z0-9]+)*
            )(?=[^a-z0-9]|$)
        """,
        target=SignalTarget.PATH,
    )
    kind_specs = (
        SurfaceKindSpec(kind="fsspec", combined_patterns=(r"fsspec",)),
        SurfaceKindSpec(
            kind="vfs_manager",
            path_patterns=(r"(?:^|[/_.-])vfs[_-]?manager(?:[/_.-]|$)",),
            content_patterns=(r"class\s+VFSManager\b",),
        ),
        SurfaceKindSpec(
            kind="bucket_manager",
            combined_patterns=(
                r"(?=.*bucket)(?=.*(?:vfs|virtual filesystem))(?=.*(?:manager|bucket_vfs))",
            ),
        ),
        SurfaceKindSpec(
            kind="journal_wal",
            combined_patterns=(
                r"(?:^|[/_.-])(?:[a-z]+[_-])?wal(?:[/_.-]|$)|(?:filesystem|fs)[_-]?journal",
            ),
        ),
        SurfaceKindSpec(
            kind="version_snapshot",
            combined_patterns=(
                r"(?=.*vfs)(?=.*(?:version|snapshot))|vfsversiontracker",
            ),
        ),
        SurfaceKindSpec(
            kind="backend_adapter",
            path_parts=("backend", "backends"),
            stem_tokens=("backend", "adapter", "integration"),
            require_domain_signal=True,
            combined_patterns=(
                r"(?:backend|adapter|integration).{0,80}(?:vfs|fsspec|filesystem)",
            ),
        ),
        SurfaceKindSpec(
            kind="handler",
            path_parts=("handler", "handlers"),
            stem_tokens=("handler",),
        ),
        SurfaceKindSpec(
            kind="endpoint",
            path_parts=("endpoint", "endpoints", "api", "apis"),
            stem_tokens=("endpoint", "_api"),
        ),
        SurfaceKindSpec(
            kind="controller",
            path_parts=("controller", "controllers"),
            stem_tokens=("controller",),
        ),
        SurfaceKindSpec(
            kind="tool",
            path_parts=("tool", "tools", "scripts", "cli"),
            stem_tokens=("_cli", "_tool"),
        ),
        SurfaceKindSpec(
            kind="server",
            path_parts=("server", "servers"),
            stem_tokens=("server",),
        ),
        SurfaceKindSpec(
            kind="sdk_manifest",
            path_parts=(
                "sdk",
                "sdks",
                "manifest",
                "manifests",
                "package.json",
                "pyproject.toml",
            ),
            path_patterns=(r"(?:^|[/_.-])sdk(?:[/_.-]|$)",),
            combined_patterns=(r"manifest|package\.json|pyproject\.toml",),
        ),
        SurfaceKindSpec(
            kind="export",
            path_patterns=(r"__init__",),
            content_patterns=(r"(?i)(?:__all__|export|entry[_-]?point|console_scripts)",),
        ),
        SurfaceKindSpec(
            kind="documentation",
            path_parts=("doc", "docs", "documentation"),
        ),
        SurfaceKindSpec(
            kind="example",
            path_parts=("example", "examples"),
            stem_tokens=("demo",),
        ),
    )
    return SurfaceInventoryPolicy(
        profile_id="legacy-vfs-surface-inventory-equivalent@1",
        schema="ipfs_accelerate_py/agent-supervisor/vfs-surface-inventory@1",
        contract_version="vfs-surface-inventory/v1",
        content_signals=(content_signal,),
        path_signals=(path_signal,),
        kind_specs=kind_specs,
        variant_suffixes=VARIANT_SUFFIXES,
        role_name_tokens=(
            "backend",
            "adapter",
            "manager",
            "journal",
            "wal",
            "fsspec",
            "snapshot",
            "version",
            "export",
            "__init__",
        ),
        default_scan_root_names=("ipfs_kit_py",),
    )


def build_widget_policy() -> SurfaceInventoryPolicy:
    """Hermetic non-VFS profile proving the engine is parameterized."""

    return SurfaceInventoryPolicy(
        profile_id="widget-surface-inventory@1",
        content_signals=(
            SurfaceSignal(
                name="widget_content",
                pattern=r"(?i)(?<![a-z0-9])(?:widget|gadget[_-]?bus)(?![a-z0-9])",
                target=SignalTarget.CONTENT,
            ),
        ),
        path_signals=(
            SurfaceSignal(
                name="widget_path",
                pattern=r"(?i)(?:^|[./_-])(?:widget|gadget)(?:[_-][a-z0-9]+)*(?=[^a-z0-9]|$)",
                target=SignalTarget.PATH,
            ),
        ),
        kind_specs=(
            SurfaceKindSpec(
                kind="widget_manager",
                path_patterns=(r"(?:^|[/_.-])widget[_-]?manager(?:[/_.-]|$)",),
                content_patterns=(r"class\s+WidgetManager\b",),
            ),
            SurfaceKindSpec(
                kind="gadget_bus",
                path_patterns=(r"gadget",),
                content_patterns=(r"GadgetBus",),
            ),
            SurfaceKindSpec(
                kind="handler",
                path_parts=("handler", "handlers"),
                stem_tokens=("handler",),
            ),
            SurfaceKindSpec(
                kind="export",
                content_patterns=(r"__all__",),
            ),
        ),
        variant_suffixes=VARIANT_SUFFIXES,
        role_name_tokens=("manager", "bus", "handler", "export", "__init__"),
    )


def _vfs_repository_fixture(root: Path) -> None:
    manager = (
        '__all__ = ["VFSManager"]\n'
        "\n"
        "class VFSManager:\n"
        "    def mount(self, path):\n"
        "        return path\n"
        "\n"
        "def register_vfs(router):\n"
        '    router.mount("/vfs", VFSManager())\n'
    )
    _write(root, "pkg/vfs_manager.py", manager)
    _write(root, "pkg/vfs_manager.py.clean", manager)
    _write(
        root,
        "pkg/vfs_manager.py.full",
        "class VFSManager:\n"
        "    def mount(self, path, options=None):\n"
        "        return path, options\n",
    )
    _write(root, "pkg/ipfs_fsspec.py", "class IPFSFileSystem:\n    pass\n")
    _write(root, "pkg/enhanced_fsspec.py", "class EnhancedFsspec:\n    pass\n")
    _write(root, "pkg/bucket_vfs_manager.py", "class BucketVFSManager:\n    pass\n")
    _write(root, "pkg/filesystem_journal.py", "class FilesystemJournal:\n    pass\n")
    _write(root, "pkg/vfs_snapshot_tracker.py", "class VFSSnapshotTracker:\n    pass\n")
    _write(root, "backends/vfs_adapter.py", "class VFSBackendAdapter:\n    pass\n")
    _write(
        root,
        "handlers/vfs_handler.py",
        '@router.get("/vfs")\n'
        "def vfs_handler():\n"
        "    return None\n",
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
        '"""Backwards compatibility wrapper for the VFS manager."""\n'
        "from pkg.vfs_manager import VFSManager\n",
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
        "from pkg.vfs_manager import VFSManager\n"
        "\n"
        "def test_mount():\n"
        '    assert VFSManager().mount("/tmp") == "/tmp"\n',
    )
    _write(
        root,
        "docs/vfs.md",
        "The `pkg/vfs_manager.py` module exports `VFSManager`.\n",
    )
    _write(
        root,
        "app.py",
        "from pkg.vfs_manager import VFSManager\n"
        "\n"
        "manager = VFSManager()\n"
        'manager.mount("/data")\n',
    )
    for index, suffix in enumerate(VARIANT_SUFFIXES):
        _write(
            root,
            f"historical/surface_{index}.py{suffix}",
            f'"""VFS historical surface {suffix}."""\nVALUE = {index}\n',
        )


def _widget_repository_fixture(root: Path) -> None:
    _write(
        root,
        "pkg/widget_manager.py",
        '__all__ = ["WidgetManager"]\n'
        "\n"
        "class WidgetManager:\n"
        "    def spin(self, rate):\n"
        "        return rate\n"
        "\n"
        "def register_widget(router):\n"
        '    router.mount("/widget", WidgetManager())\n',
    )
    _write(root, "pkg/gadget_bus.py", "class GadgetBus:\n    pass\n")
    _write(
        root,
        "handlers/widget_handler.py",
        '@router.get("/widget")\n'
        "def widget_handler():\n"
        "    return None\n",
    )
    _write(
        root,
        "app.py",
        "from pkg.widget_manager import WidgetManager\n"
        "\n"
        "manager = WidgetManager()\n"
        "manager.spin(1)\n",
    )
    # Deliberate VFS-looking names that must be ignored under the widget profile.
    _write(root, "pkg/vfs_manager.py", "class VFSManager:\n    pass\n")
    _write(
        root,
        "docs/widget.md",
        "Uses `pkg/widget_manager.py` and `WidgetManager`.\n",
    )


def test_source_lock_pins_declared_blobs_and_presence_states() -> None:
    assert LOCK_PATH.is_file()
    lock = json.loads(LOCK_PATH.read_text(encoding="utf-8"))
    assert lock["source"]["revision"] == "0cc04ebb640c4c981cf4650016e096a73ab0e8c0"
    assert lock["source"]["merge_or_cherry_pick_source_revision"] is False

    by_source = {item["source_path"]: item for item in lock["modules"]}
    assert set(by_source) == set(LOCKED_MODULE_BLOBS)

    for path, blob in LOCKED_MODULE_BLOBS.items():
        entry = by_source[path]
        assert entry["source_blob"] == blob
        assert entry["source_path_state"] == "source_only"
        assert entry["planned_path_state"] in {"planned_only", "target_present"}
        assert entry["source_path_state"] != "planned_only"
        assert entry["planned_path_state"] != "source_only"
        assert entry["source_test"]["blob"] == LOCKED_TEST_BLOBS[entry["source_test"]["path"]]
        assert entry["source_test"]["path_state"] == "source_only"
        assert entry["merge_or_cherry_pick_source"] is False
        assert entry["public_exports"]

    inventory_entry = by_source[
        "ipfs_accelerate_py/agent_supervisor/vfs_surface_inventory.py"
    ]
    assert inventory_entry["planned_path_state"] == "target_present"
    assert inventory_entry["planned_path"].endswith(
        "analysis/repository_surface_inventory.py"
    )

    for path, entry in by_source.items():
        if path.endswith("vfs_surface_inventory.py"):
            continue
        assert entry["planned_path_state"] == "planned_only"

    assert MAP_PATH.is_file()
    map_text = MAP_PATH.read_text(encoding="utf-8")
    assert "source_only" in map_text
    assert "planned_only" in map_text
    assert "target_present" in map_text
    assert "non-conflating" in map_text.lower() or "must not be conflated" in map_text
    assert "repository_surface_inventory.py" in map_text


def test_locked_blobs_are_available_in_git_object_store() -> None:
    for blob in list(LOCKED_MODULE_BLOBS.values()) + list(LOCKED_TEST_BLOBS.values()):
        subprocess.check_call(
            ["git", "cat-file", "-e", blob],
            cwd=REPO_ROOT,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        recomputed = subprocess.check_output(
            ["git", "hash-object", "--stdin"],
            input=subprocess.check_output(
                ["git", "cat-file", "-p", blob], cwd=REPO_ROOT
            ),
            cwd=REPO_ROOT,
        ).decode().strip()
        assert recomputed == blob


def test_generic_inventory_module_has_no_domain_literals() -> None:
    text = INVENTORY_MODULE.read_text(encoding="utf-8")
    sanitized = text.replace(
        "ipfs_accelerate_py/agent-supervisor/repository-surface-inventory@1",
        "<SCHEMA>",
    )
    sanitized = sanitized.replace("ipfs_accelerate_py", "<PKG>")
    hits = _FORBIDDEN_GENERIC.findall(sanitized)
    assert hits == [], f"generic module contains domain literals: {hits[:10]}"


def test_generic_module_has_no_implicit_provider_imports() -> None:
    text = INVENTORY_MODULE.read_text(encoding="utf-8")
    assert "llm_router" not in text
    assert "importlib" not in text
    assert "integrations" not in text
    assert "todo_daemon" not in text


def test_vfs_equivalent_profile_classifies_surfaces_and_relationships(
    tmp_path: Path,
) -> None:
    _vfs_repository_fixture(tmp_path)
    policy = build_vfs_equivalent_policy()

    inventory = inventory_repository_surfaces(tmp_path, policy)
    by_path = inventory.by_path()
    observed_classifications = {
        classification
        for surface in inventory.surfaces
        for classification in surface.classifications
    }
    assert set(SurfaceClassification) <= observed_classifications

    observed_kinds = {kind for surface in inventory.surfaces for kind in surface.kinds}
    assert {
        "fsspec",
        "vfs_manager",
        "bucket_manager",
        "journal_wal",
        "version_snapshot",
        "backend_adapter",
        "handler",
        "endpoint",
        "controller",
        "tool",
        "server",
        "sdk_manifest",
        "export",
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
    _vfs_repository_fixture(tmp_path)
    policy = build_vfs_equivalent_policy()

    discovered = set(discover_surface_paths(tmp_path, policy))
    inventory = inventory_repository_surfaces(tmp_path, policy)
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
    _vfs_repository_fixture(tmp_path)
    policy = build_vfs_equivalent_policy()

    inventory = inventory_repository_surfaces(
        tmp_path, policy, scan_roots=["pkg", "sdk"]
    )
    assert inventory.coverage_complete is False
    assert "sdk/manifest.json" in inventory.completeness.unexplained_paths
    assert any(
        item.code == "unexplained_surface_classification"
        and item.path == "sdk/manifest.json"
        and not item.explained
        for item in inventory.unexplained_surface_diagnostics
    )
    with pytest.raises(SurfaceInventoryError) as exc_info:
        assert_inventory_complete(inventory)
    assert "inventory_incomplete" in exc_info.value.reason_codes

    output = tmp_path / "out" / "surface-inventory.json"
    assert publish_surface_inventory(inventory, output) == output
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
    policy = build_vfs_equivalent_policy()
    inventory = inventory_repository_surfaces(tmp_path, policy)
    assert inventory.coverage_complete is True
    assert inventory.completeness.unexplained_paths == ()
    assert_inventory_complete(inventory)


def test_non_vfs_profile_parameterization(tmp_path: Path) -> None:
    _widget_repository_fixture(tmp_path)
    widget_policy = build_widget_policy()
    vfs_policy = build_vfs_equivalent_policy()

    widget_inventory = inventory_repository_surfaces(tmp_path, widget_policy)
    vfs_inventory = inventory_repository_surfaces(tmp_path, vfs_policy)

    widget_paths = {item.path for item in widget_inventory.surfaces}
    vfs_paths = {item.path for item in vfs_inventory.surfaces}

    assert "pkg/widget_manager.py" in widget_paths
    assert "pkg/gadget_bus.py" in widget_paths
    assert "pkg/vfs_manager.py" not in widget_paths

    assert "pkg/vfs_manager.py" in vfs_paths
    assert "pkg/widget_manager.py" not in vfs_paths

    manager = widget_inventory.by_path()["pkg/widget_manager.py"]
    assert manager.classification is SurfaceClassification.CANONICAL
    assert "widget_manager" in manager.kinds
    assert "WidgetManager" in {item.name for item in manager.definitions}
    assert "app.py" in manager.imported_by


def test_inventory_is_deterministic_under_reordered_inputs(tmp_path: Path) -> None:
    _vfs_repository_fixture(tmp_path)
    policy = build_vfs_equivalent_policy()

    first = inventory_repository_surfaces(tmp_path, policy).to_record()

    paths = sorted(tmp_path.rglob("*.py"))
    rng = random.Random(0)
    rng.shuffle(paths)
    for path in paths[:5]:
        data = path.read_bytes()
        path.write_bytes(data)

    second = inventory_repository_surfaces(tmp_path, policy).to_record()
    assert first["content_id"] == second["content_id"]
    assert first["surfaces"] == second["surfaces"]
    assert first["diagnostics"] == second["diagnostics"]
    assert first["completeness"] == second["completeness"]


def test_policy_identity_is_stable_and_profile_scoped() -> None:
    vfs = build_vfs_equivalent_policy()
    widget = build_widget_policy()
    assert vfs.identity() == build_vfs_equivalent_policy().identity()
    assert vfs.identity() != widget.identity()
    assert vfs.profile_id != widget.profile_id
    assert REPOSITORY_SURFACE_INVENTORY_SCHEMA in discover_inventory_schemas()


def test_scan_roots_bound_discovery(tmp_path: Path) -> None:
    _vfs_repository_fixture(tmp_path)
    policy = build_vfs_equivalent_policy()
    discovered = discover_surface_paths(tmp_path, policy, scan_roots=["pkg"])
    assert all(path.startswith("pkg/") for path in discovered)
    assert "pkg/vfs_manager.py" in discovered
    assert not any(path.startswith("handlers/") for path in discovered)
