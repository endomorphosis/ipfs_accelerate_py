"""Measure declared meaning coverage without selecting a legal meaning.

The scope owner accounts for every selected character as represented or
explicitly unsupported, refuses a statement that drops a rule, and keeps a
reversed rule order as an unapproved difference. The join owner preserves a
temporal leaf, null conditions and exceptions, unresolved approval, and the
caller polarity. A mismatch returns None.
"""

from __future__ import annotations

import hashlib
import json
from copy import deepcopy

from .dependency_block_resolution import EXPECTED_DECLARED_MEANING_PROBE

_URL = (
    "https://www.govinfo.gov/content/pkg/USCODE-2024-title10/html/"
    "USCODE-2024-title10-chap1-sec1.htm"
)


def measure_declared_meaning_probe() -> dict[str, object] | None:
    """Return the fixed declared-meaning contract, or None when owners disagree."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_DECLARED_MEANING_PROBE:
        return None
    return measured


def _wire(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _digest(value: dict) -> str:
    return hashlib.sha256(_wire(value)).hexdigest()


def _scope():
    from ipfs_datasets_py.logic.autoformal import legal_source_norm_scope_v1

    return legal_source_norm_scope_v1


def _sampling():
    from ipfs_datasets_py.logic.autoformal import legal_source_sampling

    return legal_source_sampling


def _custody():
    from ipfs_datasets_py.logic.autoformal import legal_official_source_custody

    return legal_official_source_custody


def _lexical():
    from ipfs_datasets_py.logic.legal_ir import canonical_statement_scope

    return canonical_statement_scope


def _relational():
    from ipfs_datasets_py.logic.autoformal import legal_relational_norm_v1

    return legal_relational_norm_v1


def _join():
    from ipfs_datasets_py.logic.autoformal import legal_source_typed_norm_join_v1

    return legal_source_typed_norm_join_v1


def _anchor(text: str, start: int, end: int) -> dict:
    return {
        "origin": "source",
        "start": start,
        "end": end,
        "text": text[start:end],
        "offset_unit": "unicode_character_half_open",
    }


def _seal(declaration: dict) -> dict:
    declaration["content_sha256"] = _digest(
        {key: value for key, value in declaration.items() if key != "content_sha256"}
    )
    return declaration


def _html(statute: str) -> bytes:
    return (
        '<html><head><meta charset="utf-8"><title>Fictional title</title></head><body>'
        "<span>2024 edition metadata</span><br><!-- field-start:head -->"
        '<h3 class="section-head">§1. Fictional section</h3><!-- field-end:head -->'
        "<!-- field-start:statute -->"
        + statute
        + "<!-- field-end:statute --><!-- field-start:sourcecredit -->"
        '<p class="source-credit">Pub. L. 1, Stat. 2.</p><!-- field-end:sourcecredit -->'
        "<!-- field-start:notes --><!-- field-start:miscellaneous-note -->"
        '<h4 class="note-head">Editorial Notes</h4><!-- field-end:miscellaneous-note -->'
        '<!-- field-start:amendment-note --><p class="statutory-body">Quoted shall text.</p>'
        "<!-- field-end:amendment-note --><!-- field-start:historicalandrevision-note -->"
        "<table><tr><td>Earlier</td><td>Later Café</td></tr></table>"
        "<!-- field-end:historicalandrevision-note --><!-- field-end:notes -->"
        "</body></html>"
    ).encode()


def _sampling_inputs(raw: bytes) -> dict:
    custody = _custody()
    descriptor = {
        "schema": custody.DESCRIPTOR_SCHEMA,
        "legal_id": "usc:us:10:1",
        "edition": 2024,
        "package_id": "USCODE-2024-title10",
        "granule_id": "USCODE-2024-title10-chap1-sec1",
        "url": _URL,
        "retrieved_at": "2026-10-15T01:00:00Z",
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "raw_bytes": len(raw),
        "http_provenance": {
            "requested_url": _URL,
            "final_url": _URL,
            "status_code": 200,
            "content_type": "text/html",
            "etag": None,
            "last_modified": None,
        },
    }
    return {
        "descriptor": descriptor,
        "expected_descriptor_sha256": custody.acquisition_descriptor_sha256(descriptor),
        "expected_raw_sha256": descriptor["raw_sha256"],
        "expected_raw_bytes": len(raw),
    }


def _occurrences(text: str, rows: tuple[tuple[str, str, str, str], ...]) -> list[dict]:
    found = []
    for identity, facet, literal, symbol in rows:
        start = text.index(literal)
        found.append(
            {
                "occurrence_id": identity,
                "facet": facet,
                "canonical_symbol": symbol,
                "anchor": _anchor(text, start, start + len(literal)),
            }
        )
    return found


def _declaration(text: str, occurrences: list[dict], *, unresolved: list[dict]) -> dict:
    lexical = _lexical()
    request = {
        "source_text": text,
        "context": {
            "role": "required_unavailable",
            "text": "",
            "bindings": {},
            "sha256": hashlib.sha256(b"").hexdigest(),
        },
    }
    declaration = {
        "schema": lexical.SCHEMA,
        "family": "deontic",
        "profile": lexical.PROFILE,
        "input": deepcopy(request),
        "input_sha256": _digest(request),
        "occurrences": occurrences,
        "rules": [
            {
                "rule_id": "r1",
                "body": {
                    "actor": "actor",
                    "modality": "modal",
                    "action": "action",
                    "object": "object",
                },
                "qualifiers": {"conditions": None, "exceptions": None, "temporal": None},
            }
        ],
        "statement_structure": {"op": "rule", "rule_id": "r1"},
        "binders": [],
        "coverage": {
            "declared_status": "partial",
            "segments": [
                {
                    "span": _anchor(text, 0, len(text)),
                    "disposition": "unresolved",
                    "rule_ids": ["r1"],
                    "occurrence_ids": [row["occurrence_id"] for row in occurrences],
                    "reason": "Full fictional source retained; legal meaning remains unreviewed.",
                }
            ],
        },
        "unresolved": unresolved,
    }
    return {"request": request, "declaration": _seal(declaration)}


def _sampled(text: str) -> dict:
    sampling = _sampling()
    raw = _html('<p class="statutory-body">' + text + "</p>")
    args = _sampling_inputs(raw)
    report = sampling.prepare_source_sampling(raw, **args)
    units = [
        row
        for row in report["records"]
        if row["schema"] == sampling.UNIT_SCHEMA and row["candidate_input"]
    ]
    return {"raw": raw, "sampling": report, "args": args, "unit": units[0]}


def _prepare_scope(text: str, built: dict, sampled: dict) -> dict:
    sampling = _sampling()
    request_raw = _wire(built["request"])
    declaration_raw = _wire(built["declaration"])
    unit = sampled["unit"]
    inputs = {
        "expected_sampling_report_sha256": sampling.source_sampling_report_sha256(
            sampled["sampling"]
        ),
        "selected_unit_ref": {
            "schema": unit["schema"],
            "record_cid": unit["record_cid"],
            "sha256": _digest(unit),
        },
        "expected_scope_request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "expected_scope_declaration_sha256": hashlib.sha256(declaration_raw).hexdigest(),
        **deepcopy(sampled["args"]),
    }
    return _scope().prepare_sampled_norm_scope_v1(
        sampled["sampling"],
        sampled["raw"],
        request_raw,
        declaration_raw,
        **inputs,
    )


def _coverage(text: str) -> dict:
    issue = text.index("only with")
    temporal = text.index("retain")
    rows = (
        ("actor", "actor", "Clérk", "Clérk"),
        ("modal", "modality", "may", "P"),
        ("action", "action", "retain", "retain"),
        ("object", "object", "café records", "café records"),
        ("when", "temporal", "retain", "retain"),
    )
    built = _declaration(
        text,
        _occurrences(text, rows),
        unresolved=[
            {
                "code": "approval_necessity_unresolved",
                "span": _anchor(text, issue, len(text) - 1),
                "reason": "No implication direction or sufficiency is decided.",
            }
        ],
    )
    cut = issue
    declaration = built["declaration"]
    declaration["rules"][0]["qualifiers"]["temporal"] = {
        "op": "leaf",
        "occurrence_id": "when",
    }
    declaration["coverage"]["segments"] = [
        {
            "span": _anchor(text, 0, cut),
            "disposition": "represented",
            "rule_ids": ["r1"],
            "occurrence_ids": [row["occurrence_id"] for row in declaration["occurrences"]],
            "reason": "Lexical role proposals alone; no verified semantics.",
        },
        {
            "span": _anchor(text, cut, len(text)),
            "disposition": "unsupported",
            "rule_ids": [],
            "occurrence_ids": [],
            "reason": "Approval relation is explicitly unsupported.",
        },
    ]
    _seal(declaration)
    return _prepare_scope(text, built, _sampled(text))


def _reversed(text: str) -> dict:
    built = _declaration(
        text,
        _occurrences(
            text,
            (
                ("actor", "actor", "Clérk", "Clérk"),
                ("modal", "modality", "may", "P"),
                ("action", "action", "retain", "retain"),
                ("object", "object", "café records", "café records"),
            ),
        ),
        unresolved=[
            {
                "code": "approval_necessity_unresolved",
                "span": _anchor(text, text.index("only with"), len(text) - 1),
                "reason": "No implication direction or sufficiency is decided.",
            }
        ],
    )
    first = built["declaration"]["rules"][0]
    second = deepcopy(first)
    second["rule_id"] = "r2"
    built["declaration"]["rules"] = [first, second]
    built["declaration"]["statement_structure"] = {
        "op": "all",
        "operator_anchor": None,
        "children": [
            {"op": "rule", "rule_id": "r2"},
            {"op": "rule", "rule_id": "r1"},
        ],
    }
    built["declaration"]["coverage"]["segments"][0]["rule_ids"] = ["r1", "r2"]
    _seal(built["declaration"])
    return _prepare_scope(text, built, _sampled(text))


def _omitted_refused(text: str) -> None:
    built = _declaration(
        text,
        _occurrences(
            text,
            (
                ("actor", "actor", "Clérk", "Clérk"),
                ("modal", "modality", "may", "P"),
                ("action", "action", "retain", "retain"),
                ("object", "object", "café records", "café records"),
            ),
        ),
        unresolved=[
            {
                "code": "approval_necessity_unresolved",
                "span": _anchor(text, text.index("only with"), len(text) - 1),
                "reason": "No implication direction or sufficiency is decided.",
            }
        ],
    )
    extra = deepcopy(built["declaration"]["rules"][0])
    extra["rule_id"] = "r2"
    built["declaration"]["rules"].append(extra)
    _seal(built["declaration"])
    try:
        _prepare_scope(text, built, _sampled(text))
    except ValueError as exc:
        if "every rule exactly once" not in str(exc):
            raise RuntimeError("omitted rule failed for another reason") from exc
        return
    raise RuntimeError("statement omitted a declared rule")


def _typed_request(operator: str) -> dict:
    from ipfs_datasets_py.logic.syntax_core import ast, signatures

    object_sort = signatures.LogicSort("Object")

    class Builder:
        def __init__(self) -> None:
            self.count = 0

        def nid(self) -> str:
            self.count += 1
            return "fixture:node:" + str(self.count)

        def atom(self, name: str, *arguments: str) -> dict:
            return ast.mk_predicate(
                self.nid(),
                name,
                [ast.mk_variable(self.nid(), argument, object_sort) for argument in arguments],
            )

        def modal(self, letter: str, body: dict) -> dict:
            kind = {"O": "obligation", "P": "permission", "F": "forbidden"}[letter]
            return ast.mk_extension(
                self.nid(),
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

        def binary(self, kind: str, left: dict, right: dict) -> dict:
            return {
                "and": ast.mk_and,
                "or": ast.mk_or,
                "implies": ast.mk_implies,
                "iff": ast.mk_iff,
            }[kind](self.nid(), left, right)

        def quantify(self, body: dict, binders: list[tuple[str, str]]) -> dict:
            for name, kind in reversed(binders):
                body = (ast.mk_forall if kind == "forall" else ast.mk_exists)(
                    self.nid(),
                    [ast.Binder(name, object_sort)],
                    body,
                )
            return body

    builder = Builder()
    arguments = ("agent", "recipient")
    actions = builder.binary(
        "or",
        builder.atom("Assign", *arguments),
        builder.atom("Transfer", *arguments),
    )
    approval = builder.atom("Approve", "approver", "recipient")
    body = builder.binary("implies", builder.modal(operator, actions), approval)
    binders = [("agent", "forall"), ("recipient", "forall"), ("approver", "forall")]
    root = builder.quantify(body, binders)
    declarations = [
        ("Assign", arguments),
        ("Transfer", arguments),
        ("Approve", ("approver", "recipient")),
    ]
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
    expression = ast.TypedExpression("fixture:expression", root, signature)
    relational = _relational()
    return {
        "schema": relational.REQUEST_SCHEMA,
        "interpretation_status": "authored_unreviewed_declaration",
        "expression": expression.to_dict(),
        "argument_roles": [
            {"symbol": name, "roles": list(roles)} for name, roles in declarations
        ],
        "source_binding": None,
    }


def _pointer(value, path: str):
    for piece in path.split("/")[1:]:
        value = value[int(piece)] if isinstance(value, list) else value[piece]
    return value


def _correspondences(relational_report: dict) -> tuple[list[dict], list[dict]]:
    paths = [row["typed_path"] for row in relational_report["structural_mapping"]]
    rows = []
    code = "parameter_roles_and_quantifiers_unreviewed"
    for mapped in relational_report["structural_mapping"]:
        kind, path = mapped["kind"], mapped["typed_path"]
        anchors: list[str] = []
        if kind == "variable":
            name = _pointer(relational_report["typed_expression"], path)["symbol"]
            if name == "recipient":
                anchors = ["/occurrences/0/anchor"]
            elif name == "approver":
                anchors = ["/unresolved/1/span"]
        elif kind == "predicate":
            symbol = _pointer(relational_report["typed_expression"], path)["symbol"]
            anchors = [
                "/unresolved/1/span" if symbol == "Approve" else "/occurrences/2/anchor"
            ]
        elif kind in {"extension", "legacy.deontic/v1"}:
            anchors = ["/occurrences/1/anchor"]
        elif kind in {"or", "and"}:
            anchors = ["/occurrences/2/anchor"]
        elif kind in {"implies", "iff", "not"}:
            anchors = ["/unresolved/1/span"]
        rows.append(
            {
                "typed_path": path,
                "source_anchor_paths": anchors,
                "disposition": "lexical_hypothesis" if anchors else "introduced_unresolved",
                "unresolved_codes": [code]
                + (["passive_subject_agent_unresolved"] if anchors else []),
                "rationale": (
                    "Caller hypothesis only; source roles, bindings, quantifiers "
                    "and modal interpretation remain unresolved."
                ),
            }
        )
    issues = [
        {
            "code": code,
            "rationale": (
                "All introduced parameters, predicate arities, quantifiers and "
                "role identities are unreviewed assumptions."
            ),
            "affected_typed_paths": paths,
        }
    ]
    return rows, issues


def join_byte_target_inputs(operator_text: str) -> dict[str, object]:
    text = "Clérks may be assigned or transferred to café duties only with approval."
    approval = text.index("only with")
    temporal = text.index("assigned")
    rows = (
        ("subject", "actor", "Clérks", "Clérks"),
        ("modal", "modality", "may", "P"),
        ("action", "action", "be assigned or transferred", "be assigned or transferred"),
        ("duties", "object", "to café duties", "to café duties"),
        ("when", "temporal", "assigned", "assigned"),
    )
    built = _declaration(
        text,
        _occurrences(text, rows),
        unresolved=[
            {
                "code": "passive_subject_agent_unresolved",
                "span": _anchor(text, 0, len("Clérks")),
                "reason": (
                    "A legacy actor slot records the passive subject; agent and "
                    "recipient identity remain unreviewed."
                ),
            },
            {
                "code": "approval_necessity_unresolved",
                "span": _anchor(text, approval, len(text) - 1),
                "reason": "The proposed modal and implication placements remain unreviewed.",
            },
        ],
    )
    declaration = built["declaration"]
    declaration["rules"][0]["body"] = {
        "actor": "subject",
        "modality": "modal",
        "action": "action",
        "object": "duties",
    }
    declaration["rules"][0]["qualifiers"]["temporal"] = {
        "op": "leaf",
        "occurrence_id": "when",
    }
    declaration["coverage"]["segments"][0]["reason"] = (
        "Complete fictional source retained without selecting a legal interpretation."
    )
    _seal(declaration)
    sampled = _sampled(text)
    scope_report = _prepare_scope(text, built, sampled)
    request_raw = _wire(built["request"])
    declaration_raw = _wire(declaration)
    sampling = _sampling()
    unit = sampled["unit"]
    replay = {
        "sampling_report": sampled["sampling"],
        "raw_html": sampled["raw"],
        "scope_request_raw": request_raw,
        "scope_declaration_raw": declaration_raw,
        "expected_sampling_report_sha256": sampling.source_sampling_report_sha256(
            sampled["sampling"]
        ),
        "selected_unit_ref": {
            "schema": unit["schema"],
            "record_cid": unit["record_cid"],
            "sha256": _digest(unit),
        },
        "expected_scope_request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "expected_scope_declaration_sha256": hashlib.sha256(declaration_raw).hexdigest(),
        **deepcopy(sampled["args"]),
    }
    relational_raw = _wire(_typed_request(operator_text))
    relational_report = _relational().prepare_relational_norm_v1(
        relational_raw,
        expected_request_sha256=hashlib.sha256(relational_raw).hexdigest(),
    )
    rows, issues = _correspondences(relational_report)
    join_request = {
        "schema": _join().REQUEST_SCHEMA,
        "interpretation_status": "unreviewed_correspondence_proposal",
        "component_id": "fictional:connected:one",
        "alternative_id": "hypothesis",
        "selected_unit_ref": deepcopy(replay["selected_unit_ref"]),
        "node_correspondences": rows,
        "introduced_unresolved": issues,
    }
    raw = _wire(join_request)
    report = _join().prepare_source_typed_norm_join_v1(
        scope_report,
        relational_report,
        raw,
        expected_join_request_sha256=hashlib.sha256(raw).hexdigest(),
        expected_scope_report_sha256=_scope().sampled_norm_scope_report_sha256(scope_report),
        expected_relational_report_sha256=_relational().relational_norm_report_sha256(
            relational_report
        ),
        scope_replay_inputs=replay,
        relational_request_raw=relational_raw,
        expected_relational_request_sha256=hashlib.sha256(relational_raw).hexdigest(),
    )
    return {
        "join_report": report,
        "join_sha256": _join().source_typed_norm_join_sha256(report),
        "replay": {
            "scope_report": scope_report,
            "relational_report": relational_report,
            "join_request_raw": raw,
            "expected_join_request_sha256": hashlib.sha256(raw).hexdigest(),
            "expected_scope_report_sha256": _scope().sampled_norm_scope_report_sha256(
                scope_report
            ),
            "expected_relational_report_sha256": (
                _relational().relational_norm_report_sha256(relational_report)
            ),
            "scope_replay_inputs": replay,
            "relational_request_raw": relational_raw,
            "expected_relational_request_sha256": hashlib.sha256(
                relational_raw
            ).hexdigest(),
        },
    }


def _join_source(operator_text: str) -> dict:
    report = join_byte_target_inputs(operator_text)["join_report"]
    if not isinstance(report, dict):
        raise RuntimeError("join report was not a mapping")
    return report


def _qualifier_absent(report: dict, key: str) -> bool:
    return report["scope_declaration"]["rules"][0]["qualifiers"][key] is None


def _measure() -> dict[str, object]:
    text = "Clérk may retain café records only with approval."
    covered = _coverage(text)
    if covered["authority"]["formal_norm_semantics_encoded"] is not False:
        raise RuntimeError("scope encoded formal norm semantics")
    if covered["literal_accounting_complete"] is not True:
        raise RuntimeError("literal accounting was not complete")
    if covered["declared_semantic_coverage"] != "partial":
        raise RuntimeError("semantic coverage was promoted")
    dispositions = [
        row["disposition"] for row in covered["scope_declaration"]["coverage"]["segments"]
    ]
    if dispositions != ["represented", "unsupported"]:
        raise RuntimeError("coverage was not represented or explicitly unsupported")
    qualifiers = covered["scope_declaration"]["rules"][0]["qualifiers"]
    temporal = qualifiers["temporal"]
    if (
        qualifiers["conditions"] is not None
        or qualifiers["exceptions"] is not None
        or temporal.get("op") != "leaf"
        or temporal.get("occurrence_id") != "when"
    ):
        raise RuntimeError("qualifier attachments were dropped or rewritten")
    unresolved = [row["code"] for row in covered["scope_declaration"]["unresolved"]]
    if unresolved != ["approval_necessity_unresolved"]:
        raise RuntimeError("approval issue was decided or dropped")
    if not any(
        "No approval necessity" in item for item in covered["limitations"]
    ):
        raise RuntimeError("approval necessity was no longer refused")
    polarity = next(
        row["canonical_symbol"]
        for row in covered["scope_declaration"]["occurrences"]
        if row["facet"] == "modality"
    )
    if polarity != "P":
        raise RuntimeError("caller polarity changed")
    reversed_report = _reversed(text)
    order_issues = [
        row["code"]
        for row in reversed_report["flat_compatibility"]["issues"]
        if row["code"] == "statement_and_occurrence_rule_order_differ"
    ]
    if order_issues != ["statement_and_occurrence_rule_order_differ"]:
        raise RuntimeError("reversed rule order was normalized or ignored")
    if reversed_report["flat_compatibility"]["normalization_approved"] is not False:
        raise RuntimeError("rule-order normalization was approved")
    rule_ids = [row["rule_id"] for row in reversed_report["scope_declaration"]["rules"]]
    statement_ids = [
        child["rule_id"]
        for child in reversed_report["scope_declaration"]["statement_structure"]["children"]
    ]
    if rule_ids != ["r1", "r2"] or statement_ids != ["r2", "r1"]:
        raise RuntimeError("reversed rule order was rewritten")
    if reversed_report["scope_validation"]["rule_count"] != 2:
        raise RuntimeError("rule count was not complete")
    _omitted_refused(text)
    joined = _join_source("P")
    changed = _join_source("O")
    joined_qualifiers = joined["source_scope_report"]["scope_declaration"]["rules"][0][
        "qualifiers"
    ]
    if (
        joined["source_meaning_selected"] is not False
        or any(joined["authority"].values())
        or joined_qualifiers["conditions"] is not None
        or joined_qualifiers["exceptions"] is not None
        or joined_qualifiers["temporal"] != {"occurrence_id": "when", "op": "leaf"}
    ):
        raise RuntimeError("join selected meaning or dropped an attachment")
    codes = [row["code"] for row in joined["original_unresolved"]]
    if codes != [
        "passive_subject_agent_unresolved",
        "approval_necessity_unresolved",
    ]:
        raise RuntimeError("join dropped an unresolved approval")
    modal = [
        row["modal_operator"]
        for row in joined["provenance_ledger"]
        if row["kind"] == "legacy.deontic/v1"
    ]
    changed_modal = [
        row["modal_operator"]
        for row in changed["provenance_ledger"]
        if row["kind"] == "legacy.deontic/v1"
    ]
    if modal != ["P"] or changed_modal != ["O"] or changed["provenance_ledger"] == joined[
        "provenance_ledger"
    ]:
        raise RuntimeError("caller polarity was rewritten")
    if changed["source_meaning_selected"] is not False:
        raise RuntimeError("changed polarity selected a meaning")
    return {
        "approval_necessity_decided": False,
        "caller_polarity": polarity,
        "changed_modal_operator": "O",
        "changed_polarity_distinct": True,
        "changed_source_meaning_selected": False,
        "conditions_attached": False,
        "coverage_dispositions": dispositions,
        "declared_semantic_coverage": covered["declared_semantic_coverage"],
        "exceptions_attached": False,
        "formal_norm_semantics_encoded": False,
        "join_approval_codes": codes,
        "join_authority_any_true": False,
        "join_conditions_attached": False,
        "join_exceptions_attached": False,
        "join_modal_operator": "P",
        "join_source_meaning_selected": False,
        "join_temporal_occurrence_id": "when",
        "literal_accounting_complete": True,
        "literal_accounting_scope": covered["literal_accounting_scope"],
        "normalization_approved": False,
        "omitted_rule_refused": True,
        "order_issue": "statement_and_occurrence_rule_order_differ",
        "reversed_rule_count": 2,
        "reversed_rule_ids": rule_ids,
        "reversed_statement_ids": statement_ids,
        "rule_count": covered["scope_validation"]["rule_count"],
        "semantic_completeness_claimed": False,
        "temporal_occurrence_id": "when",
        "unresolved_codes": unresolved,
    }
