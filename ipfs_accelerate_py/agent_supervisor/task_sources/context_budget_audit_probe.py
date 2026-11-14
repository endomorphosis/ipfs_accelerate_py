"""Measure an authored context-budget audit without executing a decoder.

One available source split stays a definition pin. One unavailable other
split stays a context pin. The five named controls share one declared
decoder id and one source budget. A granted cross-split allowance is
refused. A mismatch returns None.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .dependency_block_resolution import EXPECTED_CONTEXT_BUDGET_PROBE

_TERM = "covered filer"
_SOURCE_DOCUMENT = "doc-A"
_OTHER_DOCUMENT = "doc-B"
_METADATA = {"interpretation_status": "unreviewed_caller_declaration"}
_MARKERS = (
    "Context, no-context, shuffled, unrelated-context and retrieval-only controls share one decoder and one source budget.",
    "Definition dependencies stay separate from context pins.",
    "Cross-split context allowance is not granted.",
    "No decoder was executed.",
)
_cached: dict[str, object] | None = None


def measure_context_budget_audit_probe() -> dict[str, object] | None:
    """Return the locked authored-fixture audit, or None when a check fails."""

    global _cached
    if _cached is not None:
        return dict(_cached)
    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_CONTEXT_BUDGET_PROBE:
        return None
    _cached = measured
    return dict(_cached)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _wire(value: dict) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    ).encode()


def _cid(label: str) -> str:
    from ipfs_datasets_py.utils.cid_utils import cid_for_dag_json

    return cid_for_dag_json({"authored_fixture": label})


def _symbols():
    from ipfs_datasets_py.logic.integration.reasoning import legal_ir_symbols

    return legal_ir_symbols


def _lookup():
    from ipfs_datasets_py.logic.autoformal import scoped_definition_resolution

    return scoped_definition_resolution


def _table(document: str) -> dict:
    symbols = _symbols()
    builder = symbols.LegalIRSymbolTableBuilder(
        symbol_table_id="authored-table-" + document
    )
    scope = builder.add_document_scope(document)
    builder.add_definition(
        _TERM,
        symbols.LegalIRSymbolKind.DEFINED_TERM,
        symbol_id="term:source",
        scope_id=scope.scope_id,
        document_id=document,
        metadata=dict(_METADATA),
    )
    return builder.to_symbol_table().to_dict()


def _partition(identity: str, document: str, *, unavailable: bool) -> dict:
    raw = None if unavailable else _wire(_table(document))
    return {
        "partition_id": identity,
        "release_id": "authored-release-2024",
        "edition": 2024,
        "document_id": document,
        "source_version_cid": _cid("source-" + document + "-2024"),
        "scope_cid": _cid("scope-" + document + "-2024"),
        "applicability_cid": _cid("declared-applicability"),
        "table_raw": raw,
    }


def _request(partition_ids: list[str]) -> dict:
    symbols = _symbols()
    reference = symbols.LegalIRSymbolReference(
        reference_id="authored-reference-A",
        name=_TERM,
        symbol_kind=symbols.LegalIRSymbolKind.DEFINED_TERM,
        scope_id="document:" + _SOURCE_DOCUMENT,
        document_id=_SOURCE_DOCUMENT,
        explicit_target_document_id="",
        explicit_scope_id="",
        source_node_ids=("authored-use-node",),
        span_ids=("authored-use-span",),
        metadata=dict(_METADATA),
    ).to_dict()
    return {
        "reference": reference,
        "partition_ids": partition_ids,
        "source_occurrence_cid": _cid("source-use-occurrence"),
        "interpretation_cid": _cid("unreviewed-interpretation"),
    }


def _pins(manifest: dict, kind: str) -> list[dict]:
    found = []
    for row in manifest["artifacts"]:
        found.extend(row["dependency_pins"][kind])
    return found


def _controls(audit, *, decoder_id: str | None = None, source_budget: str | None = None, drop_last: bool = False):
    names = audit.CONTROL_NAMES[:-1] if drop_last else audit.CONTROL_NAMES
    rows = []
    for name in names:
        rows.append(
            {
                "name": name,
                "decoder_id": audit.DECODER_ID if decoder_id is None else decoder_id,
                "source_budget": audit.SOURCE_BUDGET if source_budget is None else source_budget,
            }
        )
    if decoder_id is not None and rows:
        rows[-1]["decoder_id"] = decoder_id
    if source_budget is not None and rows:
        rows[-1]["source_budget"] = source_budget
    return rows


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import (
        legal_context_budget_audit_v1 as audit,
    )

    audit_text = Path(audit.__file__).read_text(encoding="utf-8")
    lookup_path = Path(audit.__file__).with_name("scoped_definition_resolution.py")
    example_path = (
        Path(audit.__file__).resolve().parents[1]
        / "formalization"
        / "autoencoder"
        / "legal_relational_decoder_example_v1.py"
    )
    owner_text = {
        "audit": audit_text,
        "lookup": lookup_path.read_text(encoding="utf-8"),
        "example": example_path.read_text(encoding="utf-8"),
    }
    for text in owner_text.values():
        if "huggingface" in text or "torch" in text or "requests." in text or "urllib" in text:
            raise RuntimeError("context budget source reaches training or a remote service")
        for sentence in _MARKERS:
            if sentence not in text:
                raise RuntimeError("context budget boundary is missing")
    lookup = _lookup()
    source = _partition("source-split", _SOURCE_DOCUMENT, unavailable=False)
    other = _partition("other-split", _OTHER_DOCUMENT, unavailable=True)
    expected = [
        {
            "partition_id": row["partition_id"],
            "table_sha256": None if row["table_raw"] is None else _sha(row["table_raw"]),
        }
        for row in (source, other)
    ]
    registry = lookup.build_definition_lookup_registry([source, other], expected)
    request = _request(["source-split", "other-split"])
    raw = _wire(request)
    resolved = lookup.resolve_definition_lookup(registry, raw, _sha(raw))
    manifest = lookup.build_definition_lookup_dependencies(
        registry,
        [resolved],
        [resolved["record_cid"]],
    )
    if manifest["schema"] != "legal-graph-dependency-manifest/v1":
        raise RuntimeError("definition dependency manifest schema changed")
    if resolved["schema"] != "legal-scoped-definition-lookup/v1":
        raise RuntimeError("definition lookup schema changed")
    if resolved["status"] != "unavailable" or resolved["selected_symbol_id"] is not None:
        raise RuntimeError("unavailable split selected a definition")
    for flag in (
        "admitted",
        "formalized",
        "governing_scope_verified",
        "semantic_definition_binding_verified",
        "context_closure_verified",
        "model_executed",
        "training_qualified",
        "proof_authority",
        "exhaustive_search_verified",
    ):
        if resolved[flag] is not False:
            raise RuntimeError("lookup fixture set an authority flag")
    definition_pins = _pins(manifest, "definition")
    context_pins = _pins(manifest, "context")
    definition_keys = sorted(pin["key"] for pin in definition_pins)
    context_keys = sorted(pin["key"] for pin in context_pins)
    if definition_keys != ["definition-table:source-split"]:
        raise RuntimeError("definition pin key was not the available source split")
    if context_keys != ["definition-table:other-split"]:
        raise RuntimeError("context pin key was not the unavailable other split")
    if set(definition_keys) & set(context_keys):
        raise RuntimeError("definition pins overlap context pins")
    if (
        len(definition_pins) != 1
        or type(definition_pins[0]["sha256"]) is not str
        or len(definition_pins[0]["sha256"]) != 64
    ):
        raise RuntimeError("definition pin hash was not recorded")
    if len(context_pins) != 1 or context_pins[0]["sha256"] is not None:
        raise RuntimeError("unavailable context pin was given a hash")
    overlapping_pins_refused = False
    try:
        audit.audit_context_budget(
            definition_keys,
            definition_keys,
            _controls(audit),
        )
    except audit.ContextBudgetAuditError as error:
        overlapping_pins_refused = "definition pins overlap context pins" in str(error)
    omitted_control_refused = False
    try:
        audit.audit_context_budget(
            definition_keys,
            context_keys,
            _controls(audit, drop_last=True),
        )
    except audit.ContextBudgetAuditError as error:
        omitted_control_refused = "controls are not explicit" in str(error)
    mismatched_decoder_refused = False
    try:
        audit.audit_context_budget(
            definition_keys,
            context_keys,
            _controls(audit, decoder_id="other-decoder"),
        )
    except audit.ContextBudgetAuditError as error:
        mismatched_decoder_refused = "controls do not share one decoder" in str(error)
    mismatched_budget_refused = False
    try:
        audit.audit_context_budget(
            definition_keys,
            context_keys,
            _controls(audit, source_budget="other-budget"),
        )
    except audit.ContextBudgetAuditError as error:
        mismatched_budget_refused = "controls do not share one source budget" in str(error)
    decoder_executed_refused = False
    try:
        audit.audit_context_budget(
            definition_keys,
            context_keys,
            _controls(audit),
            decoder_executed=True,
        )
    except audit.ContextBudgetAuditError as error:
        decoder_executed_refused = "decoder stays unexecuted" in str(error)
    positive_allowance_refused = False
    try:
        audit.audit_context_budget(
            definition_keys,
            context_keys,
            _controls(audit),
            cross_split_context_allowance_count=1,
        )
    except audit.ContextBudgetAuditError as error:
        positive_allowance_refused = (
            "cross-split context allowance is not granted" in str(error)
        )
    allowance_granted_refused = False
    try:
        audit.audit_context_budget(
            definition_keys,
            context_keys,
            _controls(audit),
            cross_split_context_allowance_granted=True,
        )
    except audit.ContextBudgetAuditError as error:
        allowance_granted_refused = (
            "cross-split context allowance is not granted" in str(error)
        )
    recorded = audit.audit_context_budget(
        definition_keys,
        context_keys,
        _controls(audit),
    )
    if (
        not overlapping_pins_refused
        or not omitted_control_refused
        or not mismatched_decoder_refused
        or not mismatched_budget_refused
        or not decoder_executed_refused
        or not positive_allowance_refused
        or not allowance_granted_refused
        or recorded.get("schema") != audit.SCHEMA
        or recorded.get("authored_fixture_only") is not True
        or recorded.get("definition_pin_keys") != definition_keys
        or recorded.get("context_pin_keys") != context_keys
        or recorded.get("definition_pins_separate_from_context_pins") is not True
        or recorded.get("control_names") != list(audit.CONTROL_NAMES)
        or recorded.get("decoder_id") != audit.DECODER_ID
        or recorded.get("source_budget") != audit.SOURCE_BUDGET
        or recorded.get("controls_share_decoder") is not True
        or recorded.get("controls_share_source_budget") is not True
        or recorded.get("decoder_executed") is not False
        or recorded.get("cross_split_context_allowance_count") != 0
        or recorded.get("cross_split_context_allowance_granted") is not False
        or recorded.get("model_executed") is not False
        or recorded.get("training_executed") is not False
        or recorded.get("worker_launched") is not False
        or recorded.get("federal_corpus_scored") is not False
        or recorded.get("legal_meaning_established") is not False
        or recorded.get("legal_acceptance_claimed") is not False
        or recorded.get("admitted") is not False
        or recorded.get("formalized") is not False
    ):
        raise RuntimeError("context budget audit did not keep the named boundary")
    return {
        "admitted": False,
        "allowance_granted_refused": True,
        "authored_fixture_only": True,
        "context_pin_keys": ["definition-table:other-split"],
        "context_pin_sha256_missing": True,
        "control_names": [
            "context",
            "no-context",
            "shuffled",
            "unrelated-context",
            "retrieval-only",
        ],
        "controls_share_decoder": True,
        "controls_share_source_budget": True,
        "cross_split_context_allowance_count": 0,
        "cross_split_context_allowance_granted": False,
        "decoder_executed": False,
        "decoder_executed_refused": True,
        "decoder_id": "declared-relational-decoder",
        "definition_pin_keys": ["definition-table:source-split"],
        "definition_pin_sha256_recorded": True,
        "definition_pins_separate_from_context_pins": True,
        "federal_corpus_scored": False,
        "formalized": False,
        "legal_acceptance_claimed": False,
        "legal_meaning_established": False,
        "lookup_status": "unavailable",
        "mismatched_budget_refused": True,
        "mismatched_decoder_refused": True,
        "model_executed": False,
        "omitted_control_refused": True,
        "overlapping_pins_refused": True,
        "positive_allowance_refused": True,
        "schema": "legal-context-budget-audit/v1",
        "selected_symbol_id_absent": True,
        "source_budget": "authored-source-budget",
        "training_executed": False,
        "worker_launched": False,
    }
