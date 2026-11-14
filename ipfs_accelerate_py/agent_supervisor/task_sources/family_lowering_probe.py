"""Measure exact-profile lowering without merging counted-frame and DCEC support.

The relational profile must keep structure, signature, and its deontic
annotation. Counted-frame equality stays in that profile. A DCEC ground
fragment stays on the DCEC route. A mismatch returns None.
"""

from __future__ import annotations

import contextlib
import hashlib
import io

from .dependency_block_resolution import EXPECTED_FAMILY_LOWERING_PROBE

_DCEC_FORMULA = "O(retain(Agency, Record))"
_QUANTIFIED_DCEC = "forall x. O(retain(x))"
_COGNITIVE_DCEC = "I(Agent, retain(Record))"
_EQUALITY_DCEC = "Agency = Record"
_CALENDAR_ROUTE = "calendar_to_FOL_or_LTL_without_clock_translation"
_COGNITIVE_ROUTE = "DCEC_quantified_or_cognitive_or_event_semantics"
_CALENDAR_SENTENCE = (
    "TimeFrame is an arbitrary index with no order, duration, "
    "simultaneity or calendar interpretation."
)
_NON_DEONTIC = (
    "canonical O/P/F rules require the deontic route; "
    "broader family equivalence is unverified"
)


def measure_family_lowering_probe() -> dict[str, object] | None:
    """Return the fixed lowering contract, or None when the live owners disagree."""

    try:
        with contextlib.redirect_stdout(io.StringIO()):
            measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_FAMILY_LOWERING_PROBE:
        return None
    return measured


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import legal_counted_frame_v1 as counted
    from ipfs_datasets_py.logic.autoformal import legal_family_routes as routes
    from ipfs_datasets_py.logic.autoformal import legal_relational_norm_v1 as relational
    from ipfs_datasets_py.logic.CEC.native.dcec_integration import parse_dcec_string
    from ipfs_datasets_py.logic.intent_ir.formalize.modal_projections import _ast
    from ipfs_datasets_py.logic.syntax_core import ast, signatures

    from .codec_intact_probe import _approval_request, _prepare, _refusal

    request = _approval_request(ast, signatures, relational.REQUEST_SCHEMA)
    report, _raw = _prepare(relational, request)
    mapping = report["structural_mapping"]
    paths = [row["typed_path"] for row in mapping]
    if len(paths) != len(set(paths)):
        raise RuntimeError("structural paths collided")
    annotations = [
        row for row in mapping if row.get("kind") == "legacy.deontic/v1"
    ]
    if len(annotations) != 1 or annotations[0].get("operator") != "P":
        raise RuntimeError("deontic annotation was not preserved")
    if report["argument_roles"] != request["argument_roles"]:
        raise RuntimeError("signature roles changed")
    native = report["native_fragment"]
    if native["exact_native_ast_roundtrip"] is not True:
        raise RuntimeError("relational roundtrip failed")
    if report["source_bound"] is not False or report["source_translation_performed"] is not False:
        raise RuntimeError("relational profile bound or translated source")
    if any(report["authority"].values()):
        raise RuntimeError("relational authority flag was set")
    extra = _refusal(
        relational,
        request,
        lambda item: item["expression"]["root"].__setitem__(
            "metadata", {"note": "extra"}
        ),
    )
    if extra != "valid exact TypedExpression required":
        raise RuntimeError("extra annotation refusal changed")
    signature = _refusal(
        relational,
        request,
        lambda item: item["argument_roles"][0].__setitem__("roles", ["actor"]),
    )
    if signature != "ordered distinct bounded argument roles required":
        raise RuntimeError("signature refusal changed")

    counted_raw = counted._wire(
        counted.authored_counted_frame_request(), counted.MAX_REQUEST_BYTES
    )
    counted_report = counted.prepare_counted_frame(
        counted_raw,
        expected_request_sha256=hashlib.sha256(counted_raw).hexdigest(),
    )
    assumptions = counted_report["assumptions"]
    if not isinstance(assumptions, list) or _CALENDAR_SENTENCE not in assumptions:
        raise RuntimeError("counted-frame calendar limit missing")
    if (
        counted_report["native_TDFOL_route_used"] is not False
        or counted_report["source_bound"] is not False
        or counted_report["native_build_executed"] is not False
        or any(counted_report["authority"].values())
    ):
        raise RuntimeError("counted-frame support boundary changed")
    if counted_report["fol_receipt"]["canonical_print_stable"] is not True:
        raise RuntimeError("counted-frame print was not stable")

    parsed = parse_dcec_string(_DCEC_FORMULA)
    dcec = routes.prepare_native_fragment(
        family="dcec", formula=_DCEC_FORMULA, expected_ast=_ast(parsed)
    )
    if (
        dcec["printed"] != _DCEC_FORMULA
        or "=" in dcec["printed"]
        or dcec["exact_native_ast_roundtrip"] is not True
        or dcec["family"] != "dcec"
        or dcec["source_bound"] is not False
        or dcec["cross_family_equivalence_verified"] is not False
        or dcec["backend_executed"] is not False
        or dcec["admitted"] is not False
    ):
        raise RuntimeError("DCEC ground fragment changed")
    quantified = _dcec_refusal(routes, parse_dcec_string, _ast, _QUANTIFIED_DCEC)
    cognitive = _dcec_refusal(routes, parse_dcec_string, _ast, _COGNITIVE_DCEC)
    equality = _dcec_refusal(routes, parse_dcec_string, _ast, _EQUALITY_DCEC)
    if quantified != "DCEC route requires nonempty ground predicate arguments":
        raise RuntimeError("quantified DCEC refusal changed")
    if cognitive != "DCEC route requires nonempty ground predicate arguments":
        raise RuntimeError("cognitive DCEC refusal changed")
    if (
        "legacy.unknown_character" not in equality
        or "unknown character '='" not in equality
    ):
        raise RuntimeError("DCEC equality refusal changed")
    matrix = routes.capability_matrix()
    unsupported = matrix["unsupported_routes"]
    if _CALENDAR_ROUTE not in unsupported or _COGNITIVE_ROUTE not in unsupported:
        raise RuntimeError("unsupported translation routes changed")
    try:
        routes.route_canonical({}, {}, family="fol")
    except ValueError as exc:
        non_deontic = str(exc)
    else:
        raise RuntimeError("non-deontic canonical route was accepted")
    if non_deontic != _NON_DEONTIC:
        raise RuntimeError("non-deontic refusal changed")
    if (
        counted_report["lowering_route"] == "dcec"
        or counted_report["profile"] == dcec["family"]
    ):
        raise RuntimeError("counted-frame and DCEC support collapsed")
    counts = report["counts"]
    counted_counts = counted_report["counts"]
    return {
        "structural_mapping_rows": len(paths),
        "structural_paths_unique": True,
        "node_count": counts["nodes"],
        "quantifier_count": counts["quantifiers"],
        "modal_node_count": counts["modal_nodes"],
        "predicate_occurrence_count": counts["predicate_occurrences"],
        "signature_preserved": True,
        "annotation_operator": "P",
        "annotation_kind": "legacy.deontic/v1",
        "exact_native_ast_roundtrip": True,
        "source_bound": False,
        "source_translation_performed": False,
        "relational_authority_all_false": True,
        "extra_annotation_refused": extra,
        "signature_mismatch_refused": signature,
        "counted_profile": counted_report["profile"],
        "counted_lowering_route": counted_report["lowering_route"],
        "counted_native_tdfol_route_used": False,
        "counted_print_stable": True,
        "counted_nodes": counted_counts["nodes"],
        "counted_equality_occurrences": counted_counts["equality_occurrences"],
        "counted_membership_occurrences": counted_counts["membership_occurrences"],
        "counted_source_bound": False,
        "counted_native_build_executed": False,
        "counted_authority_all_false": True,
        "counted_calendar_interpretation": False,
        "dcec_family": "dcec",
        "dcec_printed": _DCEC_FORMULA,
        "dcec_exact_native_ast_roundtrip": True,
        "dcec_source_bound": False,
        "dcec_cross_family_equivalence_verified": False,
        "dcec_backend_executed": False,
        "dcec_admitted": False,
        "quantified_dcec_refused": quantified,
        "cognitive_dcec_refused": cognitive,
        "equality_in_dcec_refused": "legacy.unknown_character",
        "calendar_translation_unsupported": _CALENDAR_ROUTE,
        "cognitive_translation_unsupported": _COGNITIVE_ROUTE,
        "equality_translated_into_dcec": False,
        "profiles_share_support_boundary": False,
        "non_deontic_refusal": non_deontic,
    }


def _dcec_refusal(routes, parse_dcec_string, ast_for, formula: str) -> str:
    try:
        parsed = parse_dcec_string(formula)
        routes.prepare_native_fragment(
            family="dcec", formula=formula, expected_ast=ast_for(parsed)
        )
    except (ValueError, Exception) as exc:
        if type(exc) is not ValueError and type(exc).__name__ not in {
            "ValueError",
            "DCECParsingError",
        }:
            raise
        return str(exc)
    raise RuntimeError("unsupported DCEC formula was accepted")
