from __future__ import annotations

import json
from pathlib import Path

import pytest
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.adversarial import (
    MUTANT_FAMILIES,
    ResidualAdversarialAdapter,
)
from ipfs_accelerate_py.agent_supervisor.residual_intelligence.contracts import (
    ResidualIntelligenceError,
)

REPORT = (
    Path(__file__).resolve().parents[3]
    / "docs/architecture/residual_intelligence_inventory/adversarial_campaign_report.json"
)


def receipts(*, escaped: bool = False):
    return [
        {"family": family, "escaped": escaped, "receipt_id": f"receipt:{family}"}
        for family in MUTANT_FAMILIES
    ]


def test_campaign_covers_all_families_and_rejects_escapes() -> None:
    campaign = ResidualAdversarialAdapter().run("tree:final", receipts())
    assert campaign.critical_zero_escape is True
    assert {item.family for item in campaign.results} == set(MUTANT_FAMILIES)
    with pytest.raises(ResidualIntelligenceError, match="escaped"):
        ResidualAdversarialAdapter().run("tree:final", receipts(escaped=True))
    payload = json.loads(REPORT.read_text(encoding="utf-8"))
    assert payload["critical_zero_escape"] is True
    assert payload["mutant_families"] == list(MUTANT_FAMILIES)
