"""Cross-repository coverage for supervisor UI/UX IR projections."""

from __future__ import annotations

from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.proof.ir_logic_application import (
    apply_ui_logic,
)


def test_ui_logic_projects_web_mobile_and_glasses(monkeypatch: pytest.MonkeyPatch) -> None:
    datasets_root = Path(__file__).resolve().parents[3] / "ipfs_datasets"
    if not (datasets_root / "ipfs_datasets_py" / "logic" / "ui_ux_ir").is_dir():
        pytest.skip("ipfs_datasets UI/UX IR sibling checkout is unavailable")
    monkeypatch.syspath_prepend(str(datasets_root))

    result = apply_ui_logic(
        {
            "operation": "mobile_delete_account",
            "contract_id": "sca:test:mobile-delete",
            "finding_kind": "missing_surface",
            "risk_class": "critical",
            "confirmation_class": "explicit",
        }
    )

    projection = result["multi_target_projection"]
    assert result["full_ui_ux_ir"] is True
    assert result["formalization"]["roundtrip_passed"] is True
    assert projection["passed"] is True
    assert projection["targets"] == ["glasses", "mobile", "web"]
    assert set(projection["projections"]) == {"glasses", "mobile", "web"}
    assert all(
        target.get("grants_execution_authority") is False
        for target in projection["projections"].values()
    )
    assert any(
        node.get("surface") == "confirmation_sheet"
        for node in projection["projections"]["mobile"]["nodes"]
    )
