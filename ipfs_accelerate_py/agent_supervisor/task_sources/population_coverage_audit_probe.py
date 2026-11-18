"""Measure separate population counts without scaling the fixture.

The authored fixture keeps inventory, eligible, attempted, missing, invalid,
unsupported, and accepted counts apart. Compared stays unaccepted. Every
attempted row has one review record. A mismatch returns None.
"""

from __future__ import annotations

from pathlib import Path

from .dependency_block_resolution import EXPECTED_POPULATION_COVERAGE_PROBE

_BOUNDARY = (
    "Inventory, eligible, attempted, missing, invalid, unsupported and accepted counts stay separate.",
    "Compared is not an accepted population.",
    "Attempted rows include a review record.",
    "No throughput extrapolation from the small replay or the row-only total.",
)
_EVALUATION_MARKERS = (
    "The independently pinned selected population is the denominator; selection, partition and source/context hashes remain unauthenticated caller declarations.",
    "Unsupported is an explicit caller abstention category; rejected typed outputs do not establish a syntax-versus-family-capability taxonomy.",
    "Full selected and comparable-pair rates are separate descriptive structural diagnostics; neither is reviewed accuracy or training qualification.",
)
_cached: dict[str, object] | None = None


def measure_population_coverage_probe() -> dict[str, object] | None:
    """Return the locked authored-fixture audit, or None when a check fails."""

    global _cached
    if _cached is not None:
        return dict(_cached)
    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_POPULATION_COVERAGE_PROBE:
        return None
    _cached = measured
    return dict(_cached)


def _forbidden(text: str) -> bool:
    return "huggingface" in text or "torch" in text or "requests." in text or "urllib" in text


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import legal_population_coverage_audit_v1 as audit

    root = Path("/home/barberb/lift_coding/external/ipfs_datasets")
    audit_text = (
        root / "ipfs_datasets_py/logic/autoformal/legal_population_coverage_audit_v1.py"
    ).read_text(encoding="utf-8")
    plan_text = (
        root / "docs/autoencoders/legal_ir_span_graph_lean_execution_plan_20261016.md"
    ).read_text(encoding="utf-8")
    evaluation_text = (
        root / "ipfs_datasets_py/logic/autoformal/legal_relational_batch_evaluation_v1.py"
    ).read_text(encoding="utf-8")
    if _forbidden(audit_text):
        raise RuntimeError("population coverage source reaches training or a remote service")
    for sentence in _BOUNDARY:
        if sentence not in audit_text or sentence not in plan_text:
            raise RuntimeError("population coverage boundary is missing")
    for sentence in _EVALUATION_MARKERS:
        if sentence not in evaluation_text:
            raise RuntimeError("evaluation population denominator is missing")
    terminal = evaluation_text.split("TERMINAL_STATUSES = (", 1)[1].split(")", 1)[0]
    for name in ("missing", "unsupported", "invalid_typed_output", "compared"):
        if name not in terminal:
            raise RuntimeError("evaluation terminal statuses changed")
    if "accepted" in terminal:
        raise RuntimeError("evaluation terminal statuses include accepted")
    if "def prepare_relational_batch_evaluation_v1" not in evaluation_text:
        raise RuntimeError("population coverage boundary replaced the evaluator")
    accepted_relabel_refused = False
    try:
        audit.audit_population_coverage(refusal="accepted-relabel")
    except audit.PopulationCoverageAuditError as error:
        accepted_relabel_refused = "compared was counted as accepted" in str(error)
    extrapolation_refused = False
    try:
        audit.audit_population_coverage(refusal="extrapolation")
    except audit.PopulationCoverageAuditError as error:
        extrapolation_refused = "throughput was extrapolated" in str(error)
    omitted_review_refused = False
    try:
        audit.audit_population_coverage(refusal="omitted-review")
    except audit.PopulationCoverageAuditError as error:
        omitted_review_refused = "attempted row omitted its review record" in str(error)
    recorded = audit.audit_population_coverage()
    if (
        not accepted_relabel_refused
        or not extrapolation_refused
        or not omitted_review_refused
        or recorded.get("schema") != audit.SCHEMA
        or recorded.get("authored_fixture_only") is not True
        or recorded.get("inventory_count") != 9
        or recorded.get("eligible_count") != 8
        or recorded.get("attempted_count") != 5
        or recorded.get("missing_count") != 3
        or recorded.get("invalid_count") != 1
        or recorded.get("unsupported_count") != 2
        or recorded.get("accepted_count") != 0
        or recorded.get("compared_count") != 1
        or recorded.get("compared_counted_as_accepted") is not False
        or recorded.get("counts_are_distinct") is not True
        or recorded.get("review_cost_included") is not True
        or recorded.get("review_cost_record_count") != 5
        or recorded.get("review_cost_kind") != "authored-fixture-review-record"
        or recorded.get("pilot_review_cost_measured") is not False
        or recorded.get("throughput_extrapolated") is not False
        or recorded.get("row_only_total_used_as_throughput") is not False
        or recorded.get("decoder_executed") is not False
        or recorded.get("model_executed") is not False
        or recorded.get("training_executed") is not False
        or recorded.get("worker_launched") is not False
        or recorded.get("federal_corpus_scored") is not False
        or recorded.get("legal_meaning_established") is not False
        or recorded.get("legal_acceptance_claimed") is not False
        or recorded.get("admitted") is not False
        or recorded.get("formalized") is not False
    ):
        raise RuntimeError("population coverage audit did not keep the named boundary")
    return {
        "accepted_count": 0,
        "accepted_relabel_refused": True,
        "admitted": False,
        "attempted_count": 5,
        "authored_fixture_only": True,
        "compared_count": 1,
        "compared_counted_as_accepted": False,
        "counts_are_distinct": True,
        "decoder_executed": False,
        "eligible_count": 8,
        "extrapolation_refused": True,
        "federal_corpus_scored": False,
        "formalized": False,
        "inventory_count": 9,
        "invalid_count": 1,
        "legal_acceptance_claimed": False,
        "legal_meaning_established": False,
        "missing_count": 3,
        "model_executed": False,
        "omitted_review_refused": True,
        "pilot_review_cost_measured": False,
        "review_cost_included": True,
        "review_cost_kind": "authored-fixture-review-record",
        "review_cost_record_count": 5,
        "row_only_total_used_as_throughput": False,
        "schema": audit.SCHEMA,
        "throughput_extrapolated": False,
        "training_executed": False,
        "unsupported_count": 2,
        "worker_launched": False,
    }
