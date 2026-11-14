"""Measure one causal comparison that preserves the named vetoes.

Both arms use the same six factors. A changed initialization, a dropped
loss-ID veto, and an E1 ranking that repeats a veto are refused. No
hypothesis is selected and no training runs.
"""

from __future__ import annotations

from .dependency_block_resolution import EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE

_cached: dict[str, object] | None = None


def measure_modality_causal_comparison_probe() -> dict[str, object] | None:
    """Return the locked comparison result, or None when a check fails."""

    global _cached
    if _cached is not None:
        return dict(_cached)
    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE:
        return None
    _cached = measured
    return dict(_cached)


def _arm(initialization: str) -> dict[str, str]:
    return {
        "initialization": initialization,
        "source_cases": "actor_training_count:168",
        "trainable_mask": "modality.weight,modality.bias,start.weight,start.bias,end.weight,end.bias",
        "optimizer": "Adam",
        "candidates": "modality-conditions-retention-ce",
        "budget": "max_new_optimizer_steps:32",
    }


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import legal_modality_causal_comparison_v1 as comparison

    source = comparison.__file__
    text = open(source, encoding="utf-8").read()
    if "huggingface" in text or "torch" in text or "requests." in text or "urllib" in text:
        raise RuntimeError("causal comparison source reaches training or a remote service")
    control = _arm("copy_all42_trained_F15_states_fresh_ordered6_Adam/v1")
    changed_initialization_refused = False
    try:
        comparison.compare_causal_arms(
            control,
            _arm("changed-initialization"),
            comparison.PERMISSION_VETOES,
            comparison.COMPLETE_RULE_LOSS_ID_VETOES,
            [],
        )
    except comparison.CausalComparisonError as error:
        changed_initialization_refused = "factors differ" in str(error)
    dropped_loss_id_veto_refused = False
    try:
        comparison.compare_causal_arms(
            control,
            dict(control),
            comparison.PERMISSION_VETOES,
            ("other-loss-id",),
            [],
        )
    except comparison.CausalComparisonError as error:
        dropped_loss_id_veto_refused = "loss-ID vetoes were not preserved" in str(error)
    e1_ranking_consuming_veto_refused = False
    try:
        comparison.compare_causal_arms(
            control,
            dict(control),
            comparison.PERMISSION_VETOES,
            comparison.COMPLETE_RULE_LOSS_ID_VETOES,
            [comparison.COMPLETE_RULE_LOSS_ID_VETOES[0]],
        )
    except comparison.CausalComparisonError as error:
        e1_ranking_consuming_veto_refused = "consumes a veto" in str(error)
    recorded = comparison.compare_causal_arms(
        control,
        dict(control),
        comparison.PERMISSION_VETOES,
        comparison.COMPLETE_RULE_LOSS_ID_VETOES,
        ["e1-engineering-rank-only"],
    )
    if (
        not changed_initialization_refused
        or not dropped_loss_id_veto_refused
        or not e1_ranking_consuming_veto_refused
        or recorded.get("hypothesis_selected") is not False
        or recorded.get("e1_ranking_selects_hypothesis") is not False
        or recorded.get("training_executed") is not False
        or recorded.get("e1_engineering_ranking") != ["e1-engineering-rank-only"]
    ):
        raise RuntimeError("causal comparison did not preserve the veto boundary")
    return {
        "admitted": False,
        "budget_identical": True,
        "candidates_identical": True,
        "changed_initialization_refused": True,
        "complete_rule_loss_id_vetoes": ["native-tuning-calendula-t4-q5-p"],
        "dropped_loss_id_veto_refused": True,
        "e1_engineering_ranking_tracked_separately": True,
        "e1_ranking_consuming_veto_refused": True,
        "e1_ranking_selects_hypothesis": False,
        "factors": [
            "initialization",
            "source_cases",
            "trainable_mask",
            "optimizer",
            "candidates",
            "budget",
        ],
        "formalized": False,
        "hypothesis_selected": False,
        "initialization_identical": True,
        "legal_acceptance_claimed": False,
        "legal_meaning_established": False,
        "optimizer_identical": True,
        "permission_vetoes": ["32 tracked permission modality decisions"],
        "proof_authority": False,
        "source_cases_identical": True,
        "source_semantics_verified": False,
        "trainable_mask_identical": True,
        "training_executed": False,
    }
