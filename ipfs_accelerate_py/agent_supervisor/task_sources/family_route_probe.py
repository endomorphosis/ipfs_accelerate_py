"""Measure the current deontic route without a Lake build.

Obligation, permission, and prohibition stay distinct modal parameters.
A non-deontic relabeling is refused. A mismatch returns None.
"""

from __future__ import annotations

import hashlib
import json

from .dependency_block_resolution import EXPECTED_FAMILY_ROUTE_PROBE

_DISTINCT_OPERATORS = (
    "Deontic/cognitive operators are distinct interpretation parameters; "
    "no modal axioms or source truth are asserted."
)
_FORMULAS = (
    ("obligation", "O", "O(retain(entity:agency, entity:record))"),
    ("permission", "P", "P(retain(entity:agency, entity:record))"),
    ("prohibition", "F", "F(retain(entity:agency, entity:record))"),
)


def measure_family_route_probe() -> dict[str, object] | None:
    """Return the fixed route contract, or None when the live route disagrees."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_FAMILY_ROUTE_PROBE:
        return None
    return measured


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import family_qualification as qualification
    from ipfs_datasets_py.logic.autoformal import legal_family_routes as routes
    from ipfs_datasets_py.logic.intent_ir.formalize.modal_projections import _ast

    labels: dict[str, str] = {}
    flags: dict[str, object] | None = None
    for _name, operator, formula in _FORMULAS:
        parsed = qualification._strict_tdfol(formula)[0]
        native_ast = _ast(parsed)
        if native_ast.get("operator", {}).get("value") != operator:
            raise RuntimeError("native operator value changed")
        result = routes.prepare_native_fragment(
            family="deontic_fol",
            formula=formula,
            expected_ast=native_ast,
        )
        label = "deontic:" + operator
        if f'"{label}"' not in result["lean_body"]:
            raise RuntimeError("modal label missing")
        assumptions = result["lowering_details"].get("assumptions")
        if not isinstance(assumptions, list) or _DISTINCT_OPERATORS not in assumptions:
            raise RuntimeError("distinct operator assumption missing")
        observed = {
            "exact_native_ast_roundtrip": result["exact_native_ast_roundtrip"],
            "source_bound": result["source_bound"],
            "source_semantics_verified": result["source_semantics_verified"],
            "cross_family_equivalence_verified": result["cross_family_equivalence_verified"],
            "backend_executed": result["backend_executed"],
            "all_logic_families_supported": result["all_logic_families_supported"],
            "admitted": result["admitted"],
        }
        if flags is None:
            flags = observed
        elif flags != observed:
            raise RuntimeError("operator flags diverged")
        labels[operator] = label
    if flags is None:
        raise RuntimeError("no deontic fragment was measured")
    if labels["O"] == labels["P"] or labels["P"] == labels["F"] or labels["O"] == labels["F"]:
        raise RuntimeError("deontic labels collapsed")
    matrix = routes.capability_matrix()
    try:
        routes.route_canonical({}, {}, family="fol")
    except ValueError as exc:
        refusal = str(exc)
    else:
        raise RuntimeError("non-deontic canonical route was accepted")
    try:
        routes.prepare_native_fragment(
            family="deontic_fol",
            formula="G(O(retain(entity:record)))",
            expected_ast={},
        )
    except ValueError as exc:
        temporal = str(exc)
    else:
        raise RuntimeError("temporal operator was accepted inside deontic_fol")
    if "temporal_operator_outside_deontic_fol" not in temporal:
        raise RuntimeError("temporal refusal changed")
    pins = routes.producer_pins()
    if not isinstance(pins, dict) or not pins:
        raise RuntimeError("import closure was empty")
    encoded = json.dumps(
        sorted(pins.items()),
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    supported = matrix["canonical_legal"]["supported_routes"]
    return {
        "obligation_label": labels["O"],
        "permission_label": labels["P"],
        "prohibition_label": labels["F"],
        "obligation_operator": "O",
        "permission_operator": "P",
        "prohibition_operator": "F",
        "operators_distinct": True,
        "permission_rewritten_as_obligation": labels["P"] == labels["O"],
        "exact_native_ast_roundtrip": flags["exact_native_ast_roundtrip"] is True,
        "source_bound": flags["source_bound"],
        "source_semantics_verified": flags["source_semantics_verified"],
        "cross_family_equivalence_verified": flags["cross_family_equivalence_verified"],
        "backend_executed": flags["backend_executed"],
        "all_logic_families_supported": flags["all_logic_families_supported"],
        "admitted": flags["admitted"],
        "supported_routes": list(supported),
        "source_semantics_matrix": matrix["source_semantics_verified"],
        "non_deontic_refusal": refusal,
        "temporal_in_deontic_refusal": "temporal_operator_outside_deontic_fol",
        "distinct_operator_assumption": _DISTINCT_OPERATORS,
        "no_source_truth_asserted": "no modal axioms or source truth are asserted" in _DISTINCT_OPERATORS,
        "lake_build_executed": False,
        "producer_pin_count": len(pins),
        "producer_pin_sha256": hashlib.sha256(encoded).hexdigest(),
    }
