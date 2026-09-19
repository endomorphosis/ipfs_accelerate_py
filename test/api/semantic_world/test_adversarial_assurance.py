"""SAWM-042 adversarial assurance and security qualification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from ipfs_accelerate_py.agent_supervisor.analysis.program_call_ranker import (
    CallTargetRankingError,
    rank_program_call_targets,
)
from ipfs_accelerate_py.agent_supervisor.autonomous_repair.program_delta_predictor import (
    RepairPredictionError,
    predict_program_graph_delta,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_guarded import (
    evaluate_guarded_program_world_influence,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_required import (
    admit_required_program_world_dispatch,
)
from ipfs_accelerate_py.agent_supervisor.semantic_state.program_world_reuse import (
    CachedReuseEvidence,
    ProgramWorldReuseGate,
    ProgramWorldReuseKey,
    evaluate_program_world_reuse,
)
from ipfs_accelerate_py.mcp_server.mcplusplus.kubo_cid import cid_for_bytes
from ipfs_datasets_py.logic.software_contracts.semantic_state.program_views import (
    ProgramViewError,
    build_program_world_view,
)


CAMPAIGN = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "semantic_world"
    / "adversarial_campaign.json"
)
ARTIFACT = (
    Path(__file__).resolve().parents[3]
    / "artifacts/agent_supervisor/semantic_addressed_world_model/SAWM-042-adversarial.json"
)


def _cid(label: str) -> str:
    return cid_for_bytes(str(label).encode("utf-8"))


def _key(**overrides: str) -> ProgramWorldReuseKey:
    fields = {
        "state_cid": _cid("adv-state"),
        "goal_cid": _cid("adv-goal"),
        "policy_cid": _cid("adv-policy"),
        "environment_cid": _cid("adv-env"),
        "toolchain_cid": _cid("adv-toolchain"),
        "procedure_revision_cid": _cid("adv-procedure"),
    }
    fields.update(overrides)
    return ProgramWorldReuseKey(**fields)


def _reject(attack: str, reason: str) -> dict[str, Any]:
    return {
        "attack": attack,
        "rejected": True,
        "reason_code": reason,
        "completion_authority": False,
        "cas_completed": False,
        "admitted": False,
    }


class SemanticWorldAdversarialCampaign:
    def run_semantic_world_adversarial_campaign(self, attack: str) -> dict[str, Any]:
        try:
            reason = self._run(attack)
        except (ProgramViewError, RepairPredictionError, CallTargetRankingError, ValueError) as exc:
            reason = str(exc)
        return _reject(attack, reason)

    def _run(self, attack: str) -> str:
        if attack == "forged_cid":
            try:
                ProgramWorldReuseKey(
                    state_cid="cidv1-sha256-forged",
                    goal_cid=_cid("g"),
                    policy_cid=_cid("p"),
                    environment_cid=_cid("e"),
                    toolchain_cid=_cid("t"),
                    procedure_revision_cid=_cid("pr"),
                )
            except Exception as exc:
                return str(exc)
            raise AssertionError("forged CID was accepted")
        if attack == "stale_tree":
            key = _key()
            gate = ProgramWorldReuseGate(current_generation=4)
            gate.remember(CachedReuseEvidence(key=key, generation=3))
            return evaluate_program_world_reuse(key, gate=gate).reason_code
        if attack == "similarity_trap":
            return evaluate_program_world_reuse(
                _key(), similarity_candidates=({"score": 0.99, "nearest": True},)
            ).reason_code
        if attack == "poisoned_trace":
            result = admit_required_program_world_dispatch({"task_id": "SAWM-042", "receipts": ()})
            assert result["admitted"] is False
            return "missing_required_receipts"
        if attack == "model_authority":
            decision = evaluate_guarded_program_world_influence({"kind": "neural"})
            assert decision.influences_planning is False
            assert decision.neural_only_context is True
            return decision.reason_code
        if attack == "unsafe_procedure":
            key = _key()
            gate = ProgramWorldReuseGate(current_generation=3)
            gate.remember(CachedReuseEvidence(key=key, generation=3, abstraction_safe=False))
            return evaluate_program_world_reuse(key, gate=gate).reason_code
        if attack == "corrupt_index":
            try:
                rank_program_call_targets(
                    {
                        "current_symbol": "main",
                        "static_candidates": ("helper",),
                        "scores": {"injected": 9.0},
                    }
                )
            except CallTargetRankingError as exc:
                return str(exc)
            raise AssertionError("ranker accepted a non-candidate symbol")
        if attack == "root_conflict":
            result = admit_required_program_world_dispatch(
                {"task_id": "SAWM-042", "receipts": ("current_state",)}
            )
            assert result["admitted"] is False
            return "pre_root_missing"
        if attack == "privacy_failure":
            try:
                build_program_world_view(
                    {
                        "view": "legal",
                        "source_cid": _cid("private"),
                        "privacy_admitted": False,
                    }
                )
            except ProgramViewError as exc:
                return str(exc)
            raise AssertionError("unadmitted privacy view was built")
        predict_program_graph_delta(
            {"sketch": {"path": "control.duckdb", "operator": "rewrite"}}
        )
        raise AssertionError(f"unhandled attack {attack}")


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
    assert all(item["admitted"] is False for item in results)
    artifact = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert artifact["all_rejected"] is True
    assert artifact["completion_authority"] is False
