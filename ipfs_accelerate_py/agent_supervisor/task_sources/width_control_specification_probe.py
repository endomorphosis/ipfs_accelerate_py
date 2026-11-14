"""Measure the width and control specification without executing a control.

The three width classes stay distinct. The six control names stay explicit
and unexecuted. A numeric mean-coordinate MSE and a semantic-superiority
claim are refused. No producer, encoder, or training call runs.
"""

from __future__ import annotations

from pathlib import Path

from .dependency_block_resolution import EXPECTED_WIDTH_CONTROL_PROBE

_cached: dict[str, object] | None = None


def measure_width_control_probe() -> dict[str, object] | None:
    """Return the locked specification result, or None when a check fails."""

    global _cached
    if _cached is not None:
        return dict(_cached)
    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_WIDTH_CONTROL_PROBE:
        return None
    _cached = measured
    return dict(_cached)


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import (
        legal_width_control_specification_v1 as specification,
    )

    text = Path(specification.__file__).read_text(encoding="utf-8")
    if "huggingface" in text or "torch" in text or "requests." in text or "urllib" in text:
        raise RuntimeError("width specification source reaches training or a remote service")
    for sentence in (
        "Historical 8D variants, active native vectors and detached disabled-latent components remain distinct.",
        "Correct, zero, branch-disabled and permuted inputs plus source-only and retrieval controls stay named and unexecuted.",
        "Mean-coordinate MSE is not semantic superiority.",
        "No control was executed and no mean-coordinate MSE was measured.",
    ):
        if sentence not in text:
            raise RuntimeError("width specification boundary is missing")
    merged_width_refused = False
    try:
        specification.specify_width_controls(
            ("historical_8d", "active_native"),
            specification.CONTROL_NAMES,
            None,
        )
    except specification.WidthControlSpecificationError as error:
        merged_width_refused = "width classes remain distinct" in str(error)
    omitted_control_refused = False
    try:
        specification.specify_width_controls(
            specification.WIDTH_CLASSES,
            specification.CONTROL_NAMES[:-1],
            None,
        )
    except specification.WidthControlSpecificationError as error:
        omitted_control_refused = "controls are not explicit" in str(error)
    mse_value_refused = False
    try:
        specification.specify_width_controls(
            specification.WIDTH_CLASSES,
            specification.CONTROL_NAMES,
            0.0,
        )
    except specification.WidthControlSpecificationError as error:
        mse_value_refused = "mean-coordinate MSE is not semantic superiority" in str(error)
    semantic_superiority_refused = False
    try:
        specification.specify_width_controls(
            specification.WIDTH_CLASSES,
            specification.CONTROL_NAMES,
            True,
        )
    except specification.WidthControlSpecificationError as error:
        semantic_superiority_refused = (
            "mean-coordinate MSE is not semantic superiority" in str(error)
        )
    controls_executed_refused = False
    try:
        specification.specify_width_controls(
            specification.WIDTH_CLASSES,
            specification.CONTROL_NAMES,
            None,
            controls_executed=True,
        )
    except specification.WidthControlSpecificationError as error:
        controls_executed_refused = "controls stay unexecuted" in str(error)
    recorded = specification.specify_width_controls(
        specification.WIDTH_CLASSES,
        specification.CONTROL_NAMES,
        None,
    )
    if (
        not merged_width_refused
        or not omitted_control_refused
        or not mse_value_refused
        or not semantic_superiority_refused
        or not controls_executed_refused
        or recorded.get("width_classes") != list(specification.WIDTH_CLASSES)
        or recorded.get("control_names") != list(specification.CONTROL_NAMES)
        or recorded.get("width_classes_distinct") is not True
        or recorded.get("controls_explicit") is not True
        or recorded.get("controls_executed") is not False
        or recorded.get("mean_coordinate_mse_measured") is not False
        or recorded.get("mean_coordinate_mse_is_semantic_superiority") is not False
        or recorded.get("producer_called") is not False
        or recorded.get("encoder_called") is not False
        or recorded.get("preprocessing_specified") is not False
        or recorded.get("budgets_recorded") is not False
        or recorded.get("cost_measured") is not False
        or recorded.get("training_executed") is not False
        or recorded.get("model_executed") is not False
        or recorded.get("worker_launched") is not False
        or recorded.get("encoder_unfrozen") is not False
        or recorded.get("federal_corpus_scored") is not False
        or recorded.get("legal_meaning_established") is not False
        or recorded.get("legal_acceptance_claimed") is not False
        or recorded.get("admitted") is not False
        or recorded.get("formalized") is not False
        or "mean_coordinate_mse" in recorded
    ):
        raise RuntimeError("width specification did not keep the named boundary")
    return {
        "active_native_distinct": True,
        "admitted": False,
        "budgets_recorded": False,
        "control_names": [
            "correct",
            "zero",
            "branch_disabled",
            "permuted",
            "source_only",
            "retrieval",
        ],
        "controls_executed": False,
        "controls_executed_refused": True,
        "controls_explicit": True,
        "cost_measured": False,
        "detached_disabled_latent_distinct": True,
        "encoder_called": False,
        "encoder_unfrozen": False,
        "federal_corpus_scored": False,
        "formalized": False,
        "historical_8d_distinct": True,
        "legal_acceptance_claimed": False,
        "legal_meaning_established": False,
        "mean_coordinate_mse_is_semantic_superiority": False,
        "mean_coordinate_mse_measured": False,
        "merged_width_refused": True,
        "model_executed": False,
        "mse_value_refused": True,
        "omitted_control_refused": True,
        "preprocessing_specified": False,
        "producer_called": False,
        "semantic_superiority_refused": True,
        "training_executed": False,
        "width_classes": [
            "historical_8d",
            "active_native",
            "detached_disabled_latent",
        ],
        "width_classes_distinct": True,
        "worker_launched": False,
    }
