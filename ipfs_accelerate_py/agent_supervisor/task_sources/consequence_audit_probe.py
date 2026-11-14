"""Re-check shared consequences without running Lake again.

The shared owner shares one definition only when scope, source, and physical
origin match. Another edition and another definition occurrence stay apart.
Each definition, norm, and body hypothesis is necessary. A failed entailment
is a countermodel, and the recorded fixture compilation does not establish law.
"""

from __future__ import annotations

import base64
import hashlib
import itertools
import json
import re
from dataclasses import replace
from pathlib import Path

from .dependency_block_resolution import EXPECTED_CONSEQUENCE_AUDIT_PROBE
from .shared_lake_build_probe import measure_shared_lake_build_probe

_TOOLCHAIN = "leanprover/lean4:v4.34.1"
_LIMITATION = (
    "No object, relation, condition, exception, temporal origin, "
    "external definition or legal sort resolution."
)
_LEGAL_TRUTH = (
    "The conclusion is the parameterized modal action; "
    "it does not establish actual action truth or legal truth."
)
_REFUSAL = (
    "canonical O/P/F rules require the deontic route; "
    "broader family equivalence is unverified"
)
_TRIVIAL_BOOLEAN = "bounded nontrivial Boolean operands required"
_AUTHORITY = (
    "source_semantics_verified",
    "admitted",
    "formalized",
    "proof_authority",
)


def measure_consequence_audit_probe() -> dict[str, object] | None:
    """Return the executed consequence audit, or None when a check drifts."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_CONSEQUENCE_AUDIT_PROBE:
        return None
    return measured


def _measure() -> dict[str, object]:
    recorded = measure_shared_lake_build_probe()
    if recorded is None:
        raise RuntimeError("recorded fixture compilation is unavailable")
    if (
        recorded.get("lean_compilation_verified") is not True
        or recorded.get("legal_acceptance") is not False
        or recorded.get("federal_corpus_acceptance") is not False
        or recorded.get("source_semantics_verified") is not False
        or recorded.get("proof_authority") is not False
        or recorded.get("sorry_or_admit_present") is not False
        or recorded.get("admitted") is not False
        or recorded.get("formalized") is not False
    ):
        raise RuntimeError("recorded compilation claims source-law fidelity")
    receipt_sha, axiom_count = _recorded_receipt()
    same_cases = [_candidate("O"), _candidate("P")]
    same = _prepare(same_cases)
    manifest, files = same.to_dict(), dict(same.files)
    labels = _norm_labels(manifest, files, same_cases)
    editions = [
        _candidate(edition=edition, norm_id=f"edition-{edition}")
        for edition in (2024, 2025)
    ]
    edition_manifest = _prepare(editions).to_dict()
    occurrences = [
        _candidate(
            duplicate_definition=True,
            definition_index=index,
            norm_id=f"definition-occurrence-{index}",
        )
        for index in (1, 2)
    ]
    occurrence_manifest = _prepare(occurrences).to_dict()
    entailment = _entailment()
    definition = same_cases[0][3]["definition"]
    body = definition["declaration"]["body"]
    if body.get("op") != "any" or len(body.get("operands") or []) != 2:
        raise RuntimeError("definition condition shape changed")
    if definition.get("explicit_universal_biconditional") is not True:
        raise RuntimeError("definition biconditional changed")
    limitations = same_cases[0][3]["limitations"]
    if _LIMITATION not in limitations:
        raise RuntimeError("condition and exception limit was dropped")
    route = _routes()
    scope_modules = len(edition_manifest["definition_modules"])
    scope_keys = len({row["sharing_key_cid"] for row in edition_manifest["definition_modules"]})
    occurrence_modules = len(occurrence_manifest["definition_modules"])
    equal_ast = (
        occurrences[0][2]["definition_record"]["native_ast"]
        == occurrences[1][2]["definition_record"]["native_ast"]
    )
    declarations_differ = (
        occurrences[0][2]["definition_record"]["declaration"]
        != occurrences[1][2]["definition_record"]["declaration"]
    )
    if not equal_ast or not declarations_differ or occurrence_modules != 2:
        raise RuntimeError("equal definitions at distinct occurrences were merged")
    if editions[0][0]["raw_html"] != editions[1][0]["raw_html"] or scope_modules != 2:
        raise RuntimeError("another edition shared a definition")
    negation_theorem = _negation_theorem(manifest, files)
    vacuous_present = _vacuous_present(manifest, files)
    dropped = _refused(same, same_cases, "drop_hypothesis")
    invented = _refused(same, same_cases, "axiom_false")
    vacuous_refused = _refused(same, same_cases, "definition_true")
    trivial_refused = _trivial_boolean_refused()
    negation_inferred = entailment["failed_entailment_with_action_true"] == 0
    failed_is_negation = (
        negation_inferred
        or negation_theorem
        or not invented
        or not dropped
        or entailment["failed_entailment_with_action_true"] < 1
    )
    for key in _AUTHORITY:
        if manifest[key] is not False:
            raise RuntimeError("shared preparation claims authority")
    if (
        route["source_semantics_verified"] is not False
        or route["admitted"] is not False
        or route["cross_family_equivalence_verified"] is not False
    ):
        raise RuntimeError("family route claims authority")
    permission_rewritten = route["labels"][1] == route["labels"][0]
    return {
        "admitted": False,
        "all_hypotheses_force_modal_action": entailment[
            "all_hypotheses_force_modal_action"
        ],
        "alternative_modalities_collapsed": (
            len(set(labels)) != 2 or len(set(route["labels"])) != 3
        ),
        "audit_lake_build_executed": False,
        "compilation_rerun": False,
        "condition_exception_limitation_retained": True,
        "condition_resolution_claimed": False,
        "cross_family_equivalence_verified": False,
        "definition_body_operand_count": len(body["operands"]),
        "definition_body_operator": body["op"],
        "distinct_occurrence_definition_modules": occurrence_modules,
        "dropped_hypothesis_refused": dropped,
        "each_hypothesis_necessary": entailment["each_hypothesis_necessary"],
        "equal_ast_not_merged": True,
        "exception_resolution_claimed": False,
        "explicit_universal_biconditional": True,
        "failed_entailment_with_action_true": entailment[
            "failed_entailment_with_action_true"
        ],
        "failed_proof_is_proof_of_negation": failed_is_negation,
        "federal_corpus_acceptance_claimed": False,
        "formalized": False,
        "hypotheses_present": ["Def", "Norm", "Body"],
        "inapplicable_category_forces_action": not entailment[
            "inapplicable_category_does_not_force_action"
        ],
        "inapplicable_category_negates_action": not entailment[
            "inapplicable_category_does_not_negate_action"
        ],
        "invented_axiom_refused": invented,
        "legal_acceptance_claimed": False,
        "negation_inferred": negation_inferred,
        "negation_theorem_present": negation_theorem,
        "non_deontic_refusal": route["refusal"],
        "operators_distinct": len(set(route["labels"])) == 3,
        "permission_rewritten_as_obligation": permission_rewritten,
        "preparation_backend_executed": manifest["backend_executed"],
        "preparation_lean_compilation_verified": manifest["lean_compilation_verified"],
        "proof_authority": False,
        "receipt_sha256": receipt_sha,
        "recorded_axiom_lists_empty": True,
        "recorded_lean_compilation_verified": True,
        "recorded_legal_truth_assumption": True,
        "recorded_norm_theorem_count": axiom_count,
        "shared_modal_labels": labels,
        "sharing_definition_modules": len(manifest["definition_modules"]),
        "sharing_norm_modules": len(manifest["norm_bindings"]),
        "sorry_or_admit_present": vacuous_present,
        "source_semantics_verified": False,
        "supported_routes": route["supported_routes"],
        "theorem_establishes_legal_truth": _LEGAL_TRUTH
        not in manifest["theorem_assumptions"],
        "trivial_boolean_refused": trivial_refused,
        "vacuous_true_refused": vacuous_refused,
        "wrong_scope_definition_modules": scope_modules,
        "wrong_scope_shared": scope_modules == 1,
        "wrong_scope_sharing_keys": scope_keys,
    }


def _recorded_receipt() -> tuple[str, int]:
    root = _datasets_root()
    path = root / "_lir1401_build" / "build-receipt.json"
    raw = path.read_bytes()
    receipt = json.loads(raw)
    assumptions = receipt.get("theorem_assumptions")
    axioms = receipt.get("theorem_axiom_dependencies")
    if type(assumptions) is not list or _LEGAL_TRUTH not in assumptions:
        raise RuntimeError("compilation assumptions changed")
    if (
        type(axioms) is not dict
        or len(axioms) != 2
        or any(value != [] for value in axioms.values())
    ):
        raise RuntimeError("compilation axiom dependencies changed")
    if receipt.get("lean_compilation_verified") is not True:
        raise RuntimeError("recorded compilation is not the fixture command")
    if (
        receipt.get("legal_acceptance") is True
        or receipt.get("federal_corpus_acceptance") is True
        or receipt.get("source_semantics_verified") is not False
        or receipt.get("proof_authority") is not False
        or receipt.get("admitted") is not False
        or receipt.get("formalized") is not False
    ):
        raise RuntimeError("compilation receipt claims source-law fidelity")
    combined = str(receipt.get("stdout") or "") + str(receipt.get("stderr") or "")
    if "sorry" in combined or "sorryAx" in combined or " admit" in combined:
        raise RuntimeError("recorded compilation contains sorry or admit")
    return hashlib.sha256(raw).hexdigest(), len(axioms)


def _norm_labels(manifest: dict, files: dict, cases: list) -> list[str]:
    labels = []
    for binding, case in zip(manifest["norm_bindings"], cases, strict=True):
        body = files[binding["module_path"]]
        modality = case[3]["norm_declaration"]["modality"]
        label = "deontic:" + modality
        if f'"{label}"' not in body:
            raise RuntimeError("modal label missing")
        if not all(
            re.search(r"\(h" + name + r"\s*:", body) for name in ("Def", "Norm", "Body")
        ):
            raise RuntimeError("consequence hypothesis missing")
        labels.append(label)
    if labels != ["deontic:O", "deontic:P"]:
        raise RuntimeError("shared modalities collapsed or changed")
    return labels


def _negation_theorem(manifest: dict, files: dict) -> bool:
    for binding in manifest["norm_bindings"]:
        body = files[binding["module_path"]]
        if "¬" in body or "Not " in body or "theorem negation" in body:
            return True
    return False


def _vacuous_present(manifest: dict, files: dict) -> bool:
    for binding in manifest["norm_bindings"]:
        body = files[binding["module_path"]]
        if re.search(r"\b(?:axiom|sorry|admit)\b|:=\s*True\b", body):
            return True
    return False


def _entailment() -> dict[str, object]:
    forced = True
    necessary = {"Def": False, "Norm": False, "Body": False}
    failed_with_action_true = 0
    does_not_force = False
    does_not_negate = False
    for category, registered, licensed, modal_action in itertools.product(
        (False, True), repeat=4
    ):
        hypotheses = {
            "Def": category == (registered or licensed),
            "Norm": (not category) or modal_action,
            "Body": registered or licensed,
        }
        if all(hypotheses.values()) and not modal_action:
            forced = False
        if not all(hypotheses.values()) and modal_action:
            failed_with_action_true += 1
        for omitted in necessary:
            if (not modal_action) and all(
                value for key, value in hypotheses.items() if key != omitted
            ):
                necessary[omitted] = True
        if (not category) and hypotheses["Norm"] and not hypotheses["Body"]:
            if modal_action:
                does_not_negate = True
            else:
                does_not_force = True
    return {
        "all_hypotheses_force_modal_action": forced and all(necessary.values()),
        "each_hypothesis_necessary": all(necessary.values()),
        "failed_entailment_with_action_true": failed_with_action_true,
        "inapplicable_category_does_not_force_action": does_not_force,
        "inapplicable_category_does_not_negate_action": does_not_negate,
    }


def _routes() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import family_qualification as qualification
    from ipfs_datasets_py.logic.autoformal import legal_family_routes as routes
    from ipfs_datasets_py.logic.intent_ir.formalize.modal_projections import _ast

    try:
        routes.route_canonical({}, {}, family="fol")
    except ValueError as exc:
        refusal = str(exc)
    else:
        raise RuntimeError("non-deontic canonical route was accepted")
    if refusal != _REFUSAL:
        raise RuntimeError("non-deontic refusal changed")
    matrix = routes.capability_matrix()
    labels = []
    for operator, formula in (
        ("O", "O(retain(entity:agency, entity:record))"),
        ("P", "P(retain(entity:agency, entity:record))"),
        ("F", "F(retain(entity:agency, entity:record))"),
    ):
        parsed = qualification._strict_tdfol(formula)[0]
        result = routes.prepare_native_fragment(
            family="deontic_fol",
            formula=formula,
            expected_ast=_ast(parsed),
        )
        if result["exact_native_ast_roundtrip"] is not True:
            raise RuntimeError("deontic fragment did not round-trip")
        for key in (
            "source_semantics_verified",
            "cross_family_equivalence_verified",
            "admitted",
            "formalized",
        ):
            if key in result and result[key] is not False:
                raise RuntimeError("deontic fragment claims authority")
        if result["source_semantics_verified"] is not False:
            raise RuntimeError("route verified source semantics")
        labels.append("deontic:" + operator)
    if matrix["source_semantics_verified"] is not False:
        raise RuntimeError("capability matrix verified source semantics")
    supported = matrix["canonical_legal"]["supported_routes"]
    if supported != ["deontic"]:
        raise RuntimeError("supported routes changed")
    return {
        "refusal": refusal,
        "labels": labels,
        "supported_routes": list(supported),
        "source_semantics_verified": False,
        "cross_family_equivalence_verified": result["cross_family_equivalence_verified"],
        "admitted": result["admitted"],
    }


def _refused(preparation, cases, damage: str) -> bool:
    from ipfs_datasets_py.logic.autoformal import corpus_contracts as contracts
    from ipfs_datasets_py.logic.autoformal import legal_corpus_shared_lake as shared
    from ipfs_datasets_py.utils.cid_utils import canonical_dag_json_bytes

    manifest, files = preparation.to_dict(), dict(preparation.files)
    definition_path = manifest["definition_modules"][0]["module_path"]
    norm_path = manifest["norm_bindings"][0]["module_path"]
    if damage == "definition_true":
        files[definition_path] = "def formula_0 : Prop := True\n"
    elif damage == "drop_hypothesis":
        files[norm_path] = files[norm_path].replace("hBody", "missingBody")
    elif damage == "axiom_false":
        files["Unlisted.lean"] = "axiom invented : False\n"
    else:
        raise RuntimeError(damage)
    manifest["file_sha256"] = {
        name: hashlib.sha256(body.encode()).hexdigest() for name, body in files.items()
    }
    changed = replace(
        preparation,
        files=tuple(sorted(files.items())),
        manifest_json=canonical_dag_json_bytes(
            contracts.seal_record(
                {
                    key: item
                    for key, item in manifest.items()
                    if key not in {"record_cid", "cid_scope"}
                }
            )
        ).decode(),
    )
    try:
        shared.validate_shared_definition_corpus(
            changed,
            candidate_inputs=[case[0] for case in cases],
            expected_population=[case[1] for case in cases],
            toolchain=_TOOLCHAIN,
        )
    except contracts.CorpusContractError:
        return True
    return False


def _trivial_boolean_refused() -> bool:
    try:
        _candidate(body_literals=("registered",))
    except ValueError as exc:
        return _TRIVIAL_BOOLEAN in str(exc)
    return False


def _prepare(cases):
    from ipfs_datasets_py.logic.autoformal import legal_corpus_shared_lake as shared

    return shared.prepare_shared_definition_corpus(
        [case[0] for case in cases],
        expected_population=[case[1] for case in cases],
        toolchain=_TOOLCHAIN,
    )


def _candidate(
    modality="O",
    family="deontic_fol",
    *,
    edition=2024,
    duplicate_definition=False,
    definition_index=1,
    norm_id=None,
    body_literals=("registered", "licensed"),
):
    import json as json_module

    from ipfs_datasets_py.logic.autoformal import corpus_definition_norm_adapter as adapter
    from ipfs_datasets_py.logic.autoformal import legal_source_definition_norm_bridge as bridge
    from ipfs_datasets_py.logic.autoformal import legal_statutory_context as context
    from ipfs_datasets_py.utils.cid_utils import canonical_dag_json_bytes

    second = (
        '<p class="statutory-body-1em">(2) The term "covered filer" means registered or licensed.</p>'
        if duplicate_definition
        else ""
    )
    raw = (
        '<meta charset="utf-8"><h3 class="section-head">§1. Authored shared café fixture</h3>'
        '<p class="statutory-body">(a) Définitions — In this section:</p>'
        '<p class="statutory-body-1em">(1) The term "covered filer" means registered or licensed.</p>'
        + second
        + '<p class="statutory-body">(b) Each covered filer shall register.</p>'
        '<p class="statutory-body">(c) Each covered filer may file.</p>'
        '<p class="statutory-body">(d) Each covered filer shall not disclose.</p>'
    ).encode()
    url = (
        f"https://www.govinfo.gov/content/pkg/USCODE-{edition}-title40/html/"
        f"USCODE-{edition}-title40-sec1.htm"
    )
    document = context.extract_document(
        raw, url=url, edition=edition, legal_id="usc:us:40:1"
    )
    text = document["document_text"]
    definition_start = document["paragraphs"][definition_index]["char_start"]

    def span(literal, start=0):
        left = text.index(literal, start)
        return {"char_start": left, "char_end": left + len(literal), "text": literal}

    declaration = {
        "declaration_id": "authored-shared-definition",
        "profile": bridge.definitions.PROFILE,
        "interpretation_status": "unreviewed_caller_declaration",
        "paragraph_index": definition_index,
        "scope_span": span("In this section:"),
        "definition_cue": span("means", definition_start),
        "head": span("covered filer", definition_start),
        "body": {
            "op": "any",
            "operands": [
                {"op": "atom", "span": span(literal, definition_start)}
                for literal in body_literals
            ],
        },
    }
    definition_request = {
        "document": document,
        "raw_html_base64": base64.b64encode(raw).decode(),
        "declaration": declaration,
        "family": "fol",
    }
    definition = bridge.definitions.prepare_definition(definition_request)
    norm_index = {"O": 2, "P": 3, "F": 4}[modality] + int(duplicate_definition)
    start = document["paragraphs"][norm_index]["char_start"]
    cue = bridge.CUES[modality]
    action = {"O": "register", "P": "file", "F": "disclose"}[modality]
    norm = {
        "norm_id": norm_id or "authored-shared-" + modality,
        "profile": bridge.PROFILE,
        "interpretation_status": "unreviewed_caller_declaration",
        **{
            key: document[key]
            for key in ("document_id", "document_text_sha256", "legal_id", "edition")
        },
        "paragraph_index": norm_index,
        "source_span": span(f"Each covered filer {cue} {action}.", start),
        "quantifier_span": span("Each", start),
        "category_span": span("covered filer", start),
        "modality_span": span(cue, start),
        "action_span": span(action, span(cue, start)["char_end"]),
        "modality": modality,
        "variable": "x",
        "definition_head_symbol": definition["native_ast"]["formula"]["left"]["name"],
        "activation_scope": bridge.ACTIVATION,
        "modal_context": None,
    }
    request = {
        "definition_request": definition_request,
        "norm_declaration": norm,
        "family": family,
        "source_provenance": {
            "kind": "authored_official_shape_fixture",
            "authenticity_verified": False,
            "independently_reviewed": False,
            "training_admitted": False,
        },
    }
    report = bridge.prepare_definition_norm(request)

    def wire(value):
        return json_module.dumps(
            value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False
        ).encode()

    request_raw, report_raw = wire(request), wire(report)
    extracted_sha = hashlib.sha256(text.encode()).hexdigest()
    record = adapter.adapt_definition_norm_bytes(
        report_raw,
        expected_report_sha256=hashlib.sha256(report_raw).hexdigest(),
        request_raw=request_raw,
        expected_request_sha256=hashlib.sha256(request_raw).hexdigest(),
        raw_html=raw,
        expected_raw_sha256=hashlib.sha256(raw).hexdigest(),
        expected_extracted_sha256=extracted_sha,
    )
    candidate_raw = canonical_dag_json_bytes(record)
    inputs = {
        "candidate_raw": candidate_raw,
        "report_raw": report_raw,
        "request_raw": request_raw,
        "raw_html": raw,
    }
    pins = {
        "candidate_sha256": hashlib.sha256(candidate_raw).hexdigest(),
        "candidate_cid": record["record_cid"],
        "graph_cid": record["graph"]["record_cid"],
        "request_sha256": hashlib.sha256(request_raw).hexdigest(),
        "report_sha256": hashlib.sha256(report_raw).hexdigest(),
        "raw_sha256": hashlib.sha256(raw).hexdigest(),
        "extracted_sha256": extracted_sha,
    }
    return inputs, pins, record, report


def _datasets_root() -> Path:
    current = Path(__file__).resolve()
    marker = Path("ipfs_datasets_py/logic/autoformal/legal_corpus_shared_lake.py")
    for parent in current.parents:
        for root in (parent / "ipfs_datasets", parent / "external" / "ipfs_datasets"):
            if (root / marker).is_file() and (
                root / "_lir1401_build" / "build-receipt.json"
            ).is_file():
                return root
    raise RuntimeError("shared consequence owners are not on disk")
