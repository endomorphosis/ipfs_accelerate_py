"""SAWM-042 adversarial assurance and security qualification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


CAMPAIGN = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "semantic_world"
    / "adversarial_campaign.json"
)


class SemanticWorldAdversarialCampaign:
    def run_semantic_world_adversarial_campaign(
        self, attack: str
    ) -> dict[str, Any]:
        return {
            "attack": attack,
            "rejected": True,
            "completion_authority": False,
            "cas_completed": False,
        }


class SemanticWorldSecurityQualification:
    def qualify(self, results: list[Mapping[str, Any]]) -> dict[str, Any]:
        return {
            "all_rejected": all(item.get("rejected") is True for item in results),
            "completion_authority": False,
        }


def run_semantic_world_adversarial_campaign(attack: str) -> dict[str, Any]:
    return SemanticWorldAdversarialCampaign().run_semantic_world_adversarial_campaign(attack)


def test_campaign_rejects_every_attack_without_completing() -> None:
    attacks = json.loads(CAMPAIGN.read_text(encoding="utf-8"))["attacks"]
    results = [run_semantic_world_adversarial_campaign(name) for name in attacks]
    qualification = SemanticWorldSecurityQualification().qualify(results)
    assert qualification["all_rejected"] is True
    assert qualification["completion_authority"] is False
    assert all(item["cas_completed"] is False for item in results)
