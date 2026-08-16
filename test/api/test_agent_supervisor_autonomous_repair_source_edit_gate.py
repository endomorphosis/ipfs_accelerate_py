"""DCR-071/072: only an admitted exact source edit can mutate or count applied."""

from __future__ import annotations

import base64
import hashlib
from pathlib import Path

from ipfs_accelerate_py.agent_supervisor.autonomous_repair.materialize import (
    AutonomousRepairMaterializer,
    MaterializePolicy,
)


def _digest(value: bytes) -> str:
    return "sha256:" + hashlib.sha256(value).hexdigest()


def _plan(root: Path, old: bytes, new: bytes, *, operator: bool) -> dict[str, object]:
    source_edit: dict[str, object] | None = None
    if operator:
        old_digest, new_digest = _digest(old), _digest(new)
        source_edit = {
            "operator_id": "source-edit:catalog-read",
            "owner_root": str(root.resolve()),
            "relative_path": "surface.py",
            "old_digest": old_digest,
            "new_digest": new_digest,
            "old_bytes_b64": base64.b64encode(old).decode("ascii"),
            "new_bytes_b64": base64.b64encode(new).decode("ascii"),
            "forward_diff": f"--- {old_digest}\n+++ {new_digest}\n+ patched",
            "inverse_diff": f"--- {new_digest}\n+++ {old_digest}\n- patched",
            "disposition": "validation_pending",
            "admitted": True,
            "kind": "replace_exact_bytes",
        }
    return {
        "plan_id": "plan:catalog-read",
        "work_id": "work:catalog-read",
        "operation": "catalog.read",
        "materialize_ready": True,
        "preferred_path": "surface.py",
        "handler": "read_catalog",
        "source_edit_operator": source_edit,
    }


def test_catalog_or_analysis_only_rows_cannot_count_applied_but_exact_source_edit_is_pending(
    tmp_path: Path,
) -> None:
    path = tmp_path / "surface.py"
    old = (
        b"def read_catalog():\n"
        b"    return {'state': 'old'}\n\n"
        b'registry.register_tool("catalog.read", read_catalog)\n'
    )
    new = old.replace(b"'old'", b"'new'")
    path.write_bytes(old)
    materializer = AutonomousRepairMaterializer(
        repo_root=tmp_path,
        surface_files=(("accelerate", path),),
        policy=MaterializePolicy(write_data_catalog=False),
    )

    catalog_only = materializer.materialize_plans([_plan(tmp_path, old, new, operator=False)])
    assert catalog_only["passed"] is False
    assert catalog_only["summary"]["applied"] == 0
    assert catalog_only["receipts"][0]["status"] == "rejected"
    assert catalog_only["receipts"][0]["mutation_applied"] is False
    assert path.read_bytes() == old

    result = materializer.materialize_plans([_plan(tmp_path, old, new, operator=True)])
    receipt = result["receipts"][0]
    assert result["passed"] is False
    assert result["summary"]["applied"] == 1
    assert result["summary"]["validation_pending"] == 1
    assert receipt["status"] == "source_edit_validation_pending"
    assert receipt["source_edit_disposition"] == "validation_pending"
    assert receipt["mutation_applied"] is True
    assert receipt["old_digest"] == _digest(old)
    assert receipt["new_digest"] == _digest(new)
    assert path.read_bytes() == new
