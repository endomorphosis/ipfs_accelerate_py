"""Root-level implied validation mirror for KITA-001 inventory artifacts.

The authoritative suite lives under the nested ``ipfs_kit_py`` package tree.
This module asserts the declared outputs exist from a superproject checkout.
"""

from __future__ import annotations

import json
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parents[3]
NESTED_ROOT = WORKSPACE / "ipfs_kit_py"
MANIFEST = NESTED_ROOT / "docs" / "runtime_readiness" / "capability_manifest.json"
INVENTORY_MD = NESTED_ROOT / "docs" / "runtime_readiness" / "surface_inventory.md"
NESTED_TEST = (
    NESTED_ROOT
    / "tests"
    / "runtime_readiness"
    / "foundations"
    / "test_capability_manifest.py"
)

REQUIRED_DEFECT_IDS = {
    "defect:vfs-noop-rename-journal-mismatch",
    "defect:overlapping-bucket-planes",
    "defect:wal-transaction-protocol",
    "defect:arc-accounting-concurrency",
    "defect:shadowed-replica-methods",
    "defect:backend-registry-factory-fracture",
    "defect:mcplusplus-construction-failure",
    "defect:graphrag-persistence-safety-drift",
    "defect:lazy-import-dependency-version-drift",
    "defect:default-test-exclusions",
}

CLOSED_TIERS = {
    "production",
    "conditional",
    "configuration-only",
    "experimental",
    "unsupported",
    "unknown-pending-proof",
}


def test_declared_outputs_present_from_superproject():
    assert MANIFEST.is_file(), f"missing {MANIFEST}"
    assert INVENTORY_MD.is_file(), f"missing {INVENTORY_MD}"
    assert NESTED_TEST.is_file(), f"missing {NESTED_TEST}"


def test_manifest_policy_and_required_defects_from_superproject():
    data = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert data["schema"] == "CapabilityManifest@1"
    assert data["task_id"] == "KITA-001"
    assert data["policy"]["correctness_from_presence"] is False
    assert data["policy"]["presence_is_not_support"] is True
    assert set(data["policy"]["support_tiers"]) == CLOSED_TIERS
    present = {d["id"] for d in data["confirmed_defects"]}
    assert REQUIRED_DEFECT_IDS <= present
    assert data["backends"]["registered_type_count"] == 22
    assert data["backends"]["production_count"] == 0
    assert "tests/integration" in data["test_gates"]["default_pytest_ini"]["norecursedirs"]
    for entry in data["backends"]["types"]:
        assert entry["support_tier"] in CLOSED_TIERS
        assert entry["support_tier"] != "production"
