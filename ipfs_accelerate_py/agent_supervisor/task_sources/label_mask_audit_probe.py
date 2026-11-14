"""Measure distinct label tiers without creating gold from a parse.

One authored relational request parses and stays valid without a
reference. Its training mask and evaluation-gold mask stay 0. Resource
statuses stay budget, error, timeout, and cancellation. The four
example tiers keep distinct masks and origins. A mismatch returns None.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dependency_block_resolution import EXPECTED_LABEL_MASK_PROBE

_AUDIT_MARKERS = (
    "Solver or resource failure is not a negative legal label.",
    "Reviewed, compiler-weak, synthetic and unresolved tiers keep distinct masks and origins.",
    "Parser or Lean success creates no gold.",
)
_EVALUATION_MARKERS = (
    "Available references are replayed authored unreviewed declarations, not reviewed statutory gold; unavailable references remain in the denominator.",
    "No models, training, native Lake/backend execution, legal review, admission or publication occurs; existing owners may regenerate syntax-only Lean text.",
    "Incomplete, invalid frames, invalid content and exhausted local token budgets remain distinct; invalid UTF8 is not representable by this fixed lexical vocabulary and is not inferred from missing output.",
    "Explicit error, timeout, cancellation and not_started are caller declarations. Actual context, owner, source or full-bound refusal and interruption abort without a partial report.",
)
_cached: dict[str, object] | None = None


def measure_label_mask_probe() -> dict[str, object] | None:
    """Return the locked authored-fixture audit, or None when a check fails."""

    global _cached
    if _cached is not None:
        return dict(_cached)
    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_LABEL_MASK_PROBE:
        return None
    _cached = measured
    return dict(_cached)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _wire(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _transport(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


def _forbidden(text: str) -> bool:
    return "huggingface" in text or "torch" in text or "requests." in text or "urllib" in text


def _approval_request() -> dict:
    from ipfs_datasets_py.logic.autoformal import legal_relational_norm_v1 as relational
    from ipfs_datasets_py.logic.syntax_core import ast, signatures

    object_sort = signatures.LogicSort("Object")
    count = 0

    def nid() -> str:
        nonlocal count
        count += 1
        return "fixture:node:" + str(count)

    def atom(name: str, *arguments: str):
        return ast.mk_predicate(
            nid(),
            name,
            [ast.mk_variable(nid(), argument, object_sort) for argument in arguments],
        )

    def modal(letter: str, body):
        kind = {"O": "obligation", "P": "permission", "F": "forbidden"}[letter]
        return ast.mk_extension(
            nid(),
            family="tdfol",
            profile="tdfol_default",
            features=("legacy." + kind, "legacy.classic_letter"),
            payload_schema="legacy.deontic/v1",
            payload={
                "kind": kind,
                "letter": letter,
                "resolution": "deontic",
                "schema_version": "legacy.deontic/v1",
            },
            children=(body,),
        )

    action = ast.mk_or(
        nid(),
        atom("Assign", "actor", "recipient"),
        atom("Transfer", "actor", "recipient"),
    )
    approval = atom("Approved", "approver", "actor", "recipient")
    body = ast.mk_implies(nid(), modal("P", action), approval)
    for name, kind in (("approver", "forall"), ("recipient", "forall"), ("actor", "forall")):
        quantifier = ast.mk_forall if kind == "forall" else ast.mk_exists
        body = quantifier(nid(), [ast.Binder(name, object_sort)], body)
    declarations = (
        ("Assign", ("actor", "recipient")),
        ("Transfer", ("actor", "recipient")),
        ("Approved", ("approver", "actor", "recipient")),
    )
    signature = signatures.LogicSignature(
        "fixture:signature",
        family="tdfol",
        profile="tdfol_default",
        sorts=(object_sort,),
        symbols=tuple(
            signatures.declare_predicate(name, (object_sort,) * len(roles))
            for name, roles in declarations
        ),
    )
    expression = ast.TypedExpression("fixture:expression", body, signature)
    return {
        "schema": relational.REQUEST_SCHEMA,
        "interpretation_status": "authored_unreviewed_declaration",
        "expression": expression.to_dict(),
        "argument_roles": [
            {"symbol": name, "roles": list(roles)} for name, roles in declarations
        ],
        "source_binding": None,
    }


def _parsed_report(evaluation) -> dict:
    identity = "occurrence:parser-success"
    request = _approval_request()
    request_raw = _wire(request)
    row = {
        "id": identity,
        "source_sha256": _sha(b"source:" + identity.encode("utf-8")),
        "context_sha256": _sha(b"context:" + identity.encode("utf-8")),
    }
    population = {
        "schema": "legal-relational-batch-population/v1",
        "partition": "authored-controls",
        "selection_sha256": _sha(b"complete hand-authored label-mask selection"),
        "rows": [row],
    }
    population_raw = _transport(population)
    population_pin = _sha(population_raw)
    references = {
        "schema": "legal-relational-batch-references/v1",
        "population_sha256": population_pin,
        "rows": [
            {
                "id": identity,
                "status": "unavailable",
                "request_utf8": None,
                "request_sha256": None,
                "report": None,
                "report_sha256": None,
                "reason": "No authored reference supplied",
            }
        ],
    }
    predictions = {
        "schema": "legal-relational-batch-predictions/v1",
        "generation": {
            "kind": "authored_control",
            "population_sha256": population_pin,
            "latent_dimension": None,
            "checkpoint_sha256": None,
            "encoder_sha256": None,
            "settings_sha256": _sha(b"authored settings; no decoder execution"),
            "teacher_forcing": False,
            "target_access": False,
        },
        "rows": [
            {
                "id": identity,
                "source_sha256": row["source_sha256"],
                "context_sha256": row["context_sha256"],
                "status": "decoded",
                "teacher_forcing": False,
                "target_access": False,
                "request_utf8": request_raw.decode("utf-8"),
                "request_sha256": _sha(request_raw),
                "reason": None,
            }
        ],
    }
    references_raw = _transport(references)
    predictions_raw = _transport(predictions)
    return evaluation.prepare_relational_batch_evaluation_v1(
        population_raw,
        references_raw,
        predictions_raw,
        expected_population_sha256=population_pin,
        expected_references_sha256=_sha(references_raw),
        expected_predictions_sha256=_sha(predictions_raw),
    )


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import legal_label_mask_audit_v1 as audit
    from ipfs_datasets_py.logic.autoformal import (
        legal_relational_batch_evaluation_v1 as evaluation,
    )
    from ipfs_datasets_py.logic.formalization.autoencoder import (
        legal_relational_decoder_example_v1 as example,
    )

    audit_text = Path(audit.__file__).read_text(encoding="utf-8")
    evaluation_text = Path(evaluation.__file__).read_text(encoding="utf-8")
    example_text = Path(example.__file__).read_text(encoding="utf-8")
    for text in (audit_text, evaluation_text, example_text):
        if _forbidden(text):
            raise RuntimeError("label mask source reaches training or a remote service")
    for sentence in _AUDIT_MARKERS:
        if sentence not in audit_text or sentence not in example_text:
            raise RuntimeError("label mask boundary is missing")
    for sentence in _EVALUATION_MARKERS:
        if sentence not in evaluation_text:
            raise RuntimeError("evaluation label boundary is missing")
    if "def prepare_relational_decoder_example_v1" not in example_text.split("LABEL_MASK_BOUNDARY", 1)[0]:
        raise RuntimeError("label mask boundary replaced the example contract")
    tiers = example.LABEL_TIER_MASKS
    report = _parsed_report(evaluation)
    occurrences = report["occurrences"]
    if len(occurrences) != 1 or occurrences[0]["status"] != "valid_without_reference":
        raise RuntimeError("parser success was relabeled")
    if occurrences[0]["reasons"] != ["reference_unavailable"]:
        raise RuntimeError("parser success was relabeled")
    authority = report["authority"]
    if (
        authority.get("accepted") is not False
        or authority.get("evaluation_gold_qualified") is not False
        or authority.get("Lean_admitted") is not False
        or authority.get("formalized") is True
        or authority.get("admitted") is not False
        or report["training_mask"] != 0
        or report["evaluation_gold_mask"] != 0
        or report["status"] == "accepted"
        or report["generation"]["kind"] != "authored_control"
    ):
        raise RuntimeError("parser success created gold or acceptance")
    owner_statuses = tuple(evaluation.CONTENT_TERMINAL_STATUSES)
    resource_map = {
        key: evaluation._CONTENT_REJECTION_STATUSES[key] for key in audit.RESOURCE_MAP_KEYS
    }
    shared_mask_refused = False
    try:
        audit.audit_label_masks(
            (
                ("reviewed", 1, "declared-reviewed-tier"),
                ("compiler_weak", 1, "declared-compiler-weak-tier"),
                ("synthetic", 4, "declared-synthetic-tier"),
                ("unresolved", 8, "declared-unresolved-tier"),
            ),
            owner_terminal_statuses=owner_statuses,
            resource_status_map=resource_map,
            parser_success_status="valid_without_reference",
            training_mask=0,
            evaluation_gold_mask=0,
            evaluation_gold_qualified=False,
            lean_admitted=False,
            accepted=False,
        )
    except audit.LabelMaskAuditError as error:
        shared_mask_refused = "label tiers share a mask" in str(error)
    resource_label_refused = False
    try:
        audit.audit_label_masks(
            tiers,
            owner_terminal_statuses=tuple(
                status for status in owner_statuses if status != "budget_exhausted"
            ),
            resource_status_map=resource_map,
            parser_success_status="valid_without_reference",
            training_mask=0,
            evaluation_gold_mask=0,
            evaluation_gold_qualified=False,
            lean_admitted=False,
            accepted=False,
        )
    except audit.LabelMaskAuditError as error:
        resource_label_refused = "solver or resource failure is a negative legal label" in str(error)
    gold_mask_refused = False
    try:
        audit.audit_label_masks(
            tiers,
            owner_terminal_statuses=owner_statuses,
            resource_status_map=resource_map,
            parser_success_status="valid_without_reference",
            training_mask=0,
            evaluation_gold_mask=1,
            evaluation_gold_qualified=False,
            lean_admitted=False,
            accepted=False,
        )
    except audit.LabelMaskAuditError as error:
        gold_mask_refused = "parser or Lean success created gold" in str(error)
    accepted_parser_refused = False
    try:
        audit.audit_label_masks(
            tiers,
            owner_terminal_statuses=owner_statuses,
            resource_status_map=resource_map,
            parser_success_status="accepted",
            training_mask=0,
            evaluation_gold_mask=0,
            evaluation_gold_qualified=False,
            lean_admitted=False,
            accepted=False,
        )
    except audit.LabelMaskAuditError as error:
        accepted_parser_refused = "parser success created a legal label" in str(error)
    recorded = audit.audit_label_masks(
        tiers,
        owner_terminal_statuses=owner_statuses,
        resource_status_map=resource_map,
        parser_success_status=occurrences[0]["status"],
        training_mask=report["training_mask"],
        evaluation_gold_mask=report["evaluation_gold_mask"],
        evaluation_gold_qualified=authority["evaluation_gold_qualified"],
        lean_admitted=authority["Lean_admitted"],
        accepted=authority["accepted"],
    )
    if (
        not shared_mask_refused
        or not resource_label_refused
        or not gold_mask_refused
        or not accepted_parser_refused
        or recorded.get("schema") != audit.SCHEMA
        or recorded.get("authored_fixture_only") is not True
        or recorded.get("masks_distinct") is not True
        or recorded.get("origins_distinct") is not True
        or recorded.get("tier_count") != 4
        or recorded.get("reviewed_tier_review_completed") is not False
        or recorded.get("resource_failure_is_negative_legal_label") is not False
        or recorded.get("negative_legal_label_count") != 0
        or recorded.get("parser_success_status") != "valid_without_reference"
        or recorded.get("gold_created_by_parser_or_lean_success") is not False
        or recorded.get("training_mask") != 0
        or recorded.get("evaluation_gold_mask") != 0
        or recorded.get("evaluation_gold_qualified") is not False
        or recorded.get("Lean_admitted") is not False
        or recorded.get("solver_executed") is not False
        or recorded.get("lean_executed") is not False
        or recorded.get("resource_failure_executed") is not False
        or recorded.get("decoder_executed") is not False
        or recorded.get("training_executed") is not False
        or recorded.get("model_executed") is not False
        or recorded.get("federal_corpus_scored") is not False
        or recorded.get("legal_meaning_established") is not False
        or recorded.get("legal_acceptance_claimed") is not False
        or recorded.get("admitted") is not False
        or recorded.get("formalized") is not False
    ):
        raise RuntimeError("label mask audit did not keep the named boundary")
    return {
        "Lean_admitted": False,
        "accepted_parser_refused": True,
        "admitted": False,
        "authored_fixture_only": True,
        "compiler_weak_mask": recorded["compiler_weak_mask"],
        "compiler_weak_origin": recorded["compiler_weak_origin"],
        "decoder_executed": False,
        "evaluation_gold_mask": 0,
        "evaluation_gold_qualified": False,
        "federal_corpus_scored": False,
        "formalized": False,
        "gold_created_by_parser_or_lean_success": False,
        "gold_mask_refused": True,
        "lean_executed": False,
        "legal_acceptance_claimed": False,
        "legal_meaning_established": False,
        "masks_distinct": True,
        "model_executed": False,
        "negative_legal_label_count": 0,
        "origins_distinct": True,
        "parser_success_status": "valid_without_reference",
        "resource_failure_executed": False,
        "resource_failure_is_negative_legal_label": False,
        "resource_failure_statuses": list(recorded["resource_failure_statuses"]),
        "resource_label_refused": True,
        "reviewed_mask": recorded["reviewed_mask"],
        "reviewed_origin": recorded["reviewed_origin"],
        "reviewed_tier_review_completed": False,
        "schema": "legal-label-mask-audit/v1",
        "shared_mask_refused": True,
        "solver_executed": False,
        "synthetic_mask": recorded["synthetic_mask"],
        "synthetic_origin": recorded["synthetic_origin"],
        "tier_count": 4,
        "training_executed": False,
        "training_mask": 0,
        "unresolved_mask": recorded["unresolved_mask"],
        "unresolved_origin": recorded["unresolved_origin"],
        "worker_launched": False,
    }
