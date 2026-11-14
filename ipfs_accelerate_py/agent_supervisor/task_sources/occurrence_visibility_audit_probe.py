"""Measure visible occurrences without creating a governing-definition edge.

One selected occurrence stays unsupported. One selected occurrence stays
missing. The retrieved source-occurrence locator stays on the lookup
record and off every dependency edge. The producer packet is written
before a separate evaluation packet and receives no evaluation reference.
A mismatch returns None.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path

from .dependency_block_resolution import EXPECTED_OCCURRENCE_VISIBILITY_PROBE

_TERM = "covered filer"
_SOURCE_DOCUMENT = "doc-A"
_OTHER_DOCUMENT = "doc-B"
_METADATA = {"interpretation_status": "unreviewed_caller_declaration"}
_EVALUATION_MARKERS = (
    "Account for every expected occurrence and submitted declared observation.",
    "every observation must be retained exactly once",
    "Unsupported is an explicit caller abstention category; rejected typed outputs do not establish a syntax-versus-family-capability taxonomy.",
)
_LOOKUP_MARKERS = (
    "It does not discover governing scopes or legal\nprecedence.",
    "Retrieved locator evidence is not automatically a governing-definition edge.",
    "A retrieved source occurrence stays on the lookup record.",
)
_AUDIT_MARKERS = (
    "Every selected occurrence stays visible.",
    "Unsupported and missing outcomes stay visible.",
    "Retrieved locator evidence is not automatically a governing-definition edge.",
    "The producer packet is written before the evaluation packet.",
    "The producer packet receives no evaluation reference.",
)
_cached: dict[str, object] | None = None


def measure_occurrence_visibility_probe() -> dict[str, object] | None:
    """Return the locked authored-fixture audit, or None when a check fails."""

    global _cached
    if _cached is not None:
        return dict(_cached)
    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_OCCURRENCE_VISIBILITY_PROBE:
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


def _transport(value: dict) -> bytes:
    return json.dumps(
        value,
        sort_keys=True,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8") + b"\n"


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


def _request(partition_ids: list[str], locator_cid: str) -> dict:
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
        "source_occurrence_cid": locator_cid,
        "interpretation_cid": _cid("unreviewed-interpretation"),
    }


def _batch(owner) -> dict:
    selected = "occurrence:selected"
    unresolved = "occurrence:unresolved"
    rows = []
    for identity in (selected, unresolved):
        rows.append(
            {
                "id": identity,
                "source_sha256": _sha(("source:" + identity).encode("utf-8")),
                "context_sha256": _sha(("context:" + identity).encode("utf-8")),
            }
        )
    population = {
        "schema": "legal-relational-batch-population/v1",
        "partition": "authored-controls",
        "selection_sha256": _sha(b"complete hand-authored occurrence selection"),
        "rows": rows,
    }
    population_raw = _transport(population)
    population_pin = _sha(population_raw)
    references = {
        "schema": "legal-relational-batch-references/v1",
        "population_sha256": population_pin,
        "rows": [
            {
                "id": row["id"],
                "status": "unavailable",
                "request_utf8": None,
                "request_sha256": None,
                "report": None,
                "report_sha256": None,
                "reason": "No authored reference supplied",
            }
            for row in rows
        ],
    }
    selected_row = rows[0]
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
                "id": selected_row["id"],
                "source_sha256": selected_row["source_sha256"],
                "context_sha256": selected_row["context_sha256"],
                "status": "unsupported",
                "teacher_forcing": False,
                "target_access": False,
                "request_utf8": None,
                "request_sha256": None,
                "reason": "Explicit authored control disposition",
            }
        ],
    }
    references_raw = _transport(references)
    predictions_raw = _transport(predictions)
    report = owner.prepare_relational_batch_evaluation_v1(
        population_raw,
        references_raw,
        predictions_raw,
        expected_population_sha256=population_pin,
        expected_references_sha256=_sha(references_raw),
        expected_predictions_sha256=_sha(predictions_raw),
    )
    return report


def _forbidden(text: str) -> bool:
    return "huggingface" in text or "torch" in text or "requests." in text or "urllib" in text


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import (
        legal_occurrence_visibility_audit_v1 as audit,
    )
    from ipfs_datasets_py.logic.autoformal import (
        legal_relational_batch_evaluation_v1 as evaluation,
    )
    from ipfs_datasets_py.logic.autoformal import (
        legal_training_metric_report_v1 as packets,
    )

    audit_text = Path(audit.__file__).read_text(encoding="utf-8")
    evaluation_text = Path(evaluation.__file__).read_text(encoding="utf-8")
    lookup_path = Path(audit.__file__).with_name("scoped_definition_resolution.py")
    lookup_text = lookup_path.read_text(encoding="utf-8")
    packet_text = Path(packets.__file__).read_text(encoding="utf-8")
    for text in (audit_text, evaluation_text, lookup_text, packet_text):
        if _forbidden(text):
            raise RuntimeError("occurrence visibility source reaches training or a remote service")
    for sentence in _AUDIT_MARKERS:
        if sentence not in audit_text:
            raise RuntimeError("occurrence visibility boundary is missing")
    for sentence in _EVALUATION_MARKERS:
        if sentence not in evaluation_text:
            raise RuntimeError("evaluation occurrence boundary is missing")
    for sentence in _LOOKUP_MARKERS:
        if sentence not in lookup_text:
            raise RuntimeError("lookup locator boundary is missing")
    report = _batch(evaluation)
    occurrences = report["occurrences"]
    if [row["id"] for row in occurrences] != ["occurrence:selected", "occurrence:unresolved"]:
        raise RuntimeError("selected occurrences were not retained")
    by_id = {row["id"]: row for row in occurrences}
    if by_id["occurrence:selected"]["status"] != "unsupported":
        raise RuntimeError("unsupported outcome was not visible")
    if by_id["occurrence:selected"]["reasons"] != ["declared_unsupported"]:
        raise RuntimeError("unsupported outcome was not visible")
    if by_id["occurrence:unresolved"]["status"] != "missing":
        raise RuntimeError("missing outcome was not visible")
    if by_id["occurrence:unresolved"]["reasons"] != ["missing_prediction"]:
        raise RuntimeError("missing outcome was not visible")
    authority = report["authority"]
    if (
        report["metric_reference_access"] is not True
        or authority.get("accepted") is not False
        or authority.get("governing_use_verified") is not False
        or authority.get("training_executed") is not False
        or authority.get("model_executed") is not False
        or authority.get("admitted") is not False
        or authority.get("formalized") is True
        or report["training_mask"] != 0
        or report["evaluation_gold_mask"] != 0
        or report["generation"]["kind"] != "authored_control"
    ):
        raise RuntimeError("evaluation report claimed acceptance or hid a flag")
    lookup = _lookup()
    locator_cid = _cid("source-use-occurrence")
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
    request = _request(["source-split", "other-split"], locator_cid)
    raw = _wire(request)
    resolved = lookup.resolve_definition_lookup(registry, raw, _sha(raw))
    manifest = lookup.build_definition_lookup_dependencies(
        registry,
        [resolved],
        [resolved["record_cid"]],
    )
    if resolved["source_occurrence_cid"] != locator_cid:
        raise RuntimeError("retrieved locator left the lookup record")
    if resolved["status"] != "unavailable" or resolved["selected_symbol_id"] is not None:
        raise RuntimeError("unavailable route selected a governing definition")
    if (
        resolved["governing_scope_verified"] is not False
        or resolved["semantic_definition_binding_verified"] is not False
        or resolved["admitted"] is not False
        or resolved["formalized"] is not False
    ):
        raise RuntimeError("lookup fixture verified a governing scope")
    edges = manifest["edges"]
    pin_kinds = sorted({kind for row in manifest["artifacts"] for kind in row["dependency_pins"]})
    for edge in edges:
        blob = edge["dependent_id"] + "\n" + edge["dependency_id"]
        if locator_cid in blob or "governing-definition" in blob:
            raise RuntimeError("retrieved locator became a governing-definition edge")
    if "governing-definition" in pin_kinds:
        raise RuntimeError("retrieved locator became a governing-definition edge")
    visible = [{"id": row["id"], "status": row["status"]} for row in occurrences]
    selected_ids = ["occurrence:selected", "occurrence:unresolved"]
    dropped_occurrence_refused = False
    try:
        audit.audit_occurrence_visibility(
            [*selected_ids, "occurrence:hidden"],
            visible,
            edges,
            locator_cid=locator_cid,
            locator_on_lookup=locator_cid,
            pin_kinds=pin_kinds,
            governing_scope_verified=False,
            semantic_definition_binding_verified=False,
            producer_written_before_evaluation=True,
            separate_packet_owners=True,
            producer_evaluation_reference_access=False,
            evaluation_metric_reference_access=True,
        )
    except audit.OccurrenceVisibilityAuditError as error:
        dropped_occurrence_refused = "selected occurrence is not visible" in str(error)
    hidden = [dict(row) for row in visible]
    hidden[0]["status"] = "compared"
    hidden_unsupported_refused = False
    try:
        audit.audit_occurrence_visibility(
            selected_ids,
            hidden,
            edges,
            locator_cid=locator_cid,
            locator_on_lookup=locator_cid,
            pin_kinds=pin_kinds,
            governing_scope_verified=False,
            semantic_definition_binding_verified=False,
            producer_written_before_evaluation=True,
            separate_packet_owners=True,
            producer_evaluation_reference_access=False,
            evaluation_metric_reference_access=True,
        )
    except audit.OccurrenceVisibilityAuditError as error:
        hidden_unsupported_refused = "unsupported outcome is not visible" in str(error)
    governing_edge_refused = False
    try:
        audit.audit_occurrence_visibility(
            selected_ids,
            visible,
            [*edges, {"dependent_id": "definition-lookup:fixture", "dependency_id": "governing-definition"}],
            locator_cid=locator_cid,
            locator_on_lookup=locator_cid,
            pin_kinds=pin_kinds,
            governing_scope_verified=False,
            semantic_definition_binding_verified=False,
            producer_written_before_evaluation=True,
            separate_packet_owners=True,
            producer_evaluation_reference_access=False,
            evaluation_metric_reference_access=True,
        )
    except audit.OccurrenceVisibilityAuditError as error:
        governing_edge_refused = "retrieved locator is a governing-definition edge" in str(error)
    locator_edge_refused = False
    try:
        audit.audit_occurrence_visibility(
            selected_ids,
            visible,
            [*edges, {"dependent_id": "definition-lookup:fixture", "dependency_id": locator_cid}],
            locator_cid=locator_cid,
            locator_on_lookup=locator_cid,
            pin_kinds=pin_kinds,
            governing_scope_verified=False,
            semantic_definition_binding_verified=False,
            producer_written_before_evaluation=True,
            separate_packet_owners=True,
            producer_evaluation_reference_access=False,
            evaluation_metric_reference_access=True,
        )
    except audit.OccurrenceVisibilityAuditError as error:
        locator_edge_refused = "retrieved locator is a governing-definition edge" in str(error)
    governing_scope_refused = False
    try:
        audit.audit_occurrence_visibility(
            selected_ids,
            visible,
            edges,
            locator_cid=locator_cid,
            locator_on_lookup=locator_cid,
            pin_kinds=pin_kinds,
            governing_scope_verified=True,
            semantic_definition_binding_verified=False,
            producer_written_before_evaluation=True,
            separate_packet_owners=True,
            producer_evaluation_reference_access=False,
            evaluation_metric_reference_access=True,
        )
    except audit.OccurrenceVisibilityAuditError as error:
        governing_scope_refused = "governing scope stays unverified" in str(error)
    with tempfile.TemporaryDirectory(prefix="lir2002-") as directory:
        root = Path(directory)
        producer_reference_refused = False
        try:
            packets.materialize_separated_packets(
                root,
                [{"id": "producer-1", "observation": "producer", "evaluation_reference": "fixture-reference"}],
                [{"id": "evaluation-1", "observation": "evaluation", "evaluation_reference": "fixture-reference"}],
            )
        except packets.TrainingMetricReportError as error:
            producer_reference_refused = "evaluation reference" in str(error)
        if (root / packets.PRODUCER_NAME).exists() or (root / packets.EVALUATION_NAME).exists():
            raise RuntimeError("refused producer packet was written")
        sealed = packets.materialize_separated_packets(
            root,
            [{"id": "producer-1", "observation": "producer-observation"}],
            [{"id": "evaluation-1", "observation": "evaluation-observation", "evaluation_reference": "fixture-reference"}],
        )
        producer_text = (root / packets.PRODUCER_NAME).read_text(encoding="utf-8")
        evaluation_text_packet = (root / packets.EVALUATION_NAME).read_text(encoding="utf-8")
        producer_body = json.loads(producer_text)
    if (
        not producer_reference_refused
        or "evaluation_reference" in producer_body
        or producer_body.get("evaluation_reference_access") is not False
        or sealed.get("producer_owner") != packets.PRODUCER_OWNER
        or sealed.get("evaluation_owner") != packets.EVALUATION_OWNER
        or sealed.get("producer_written_before_evaluation") is not True
        or sealed.get("separate_packet_owners") is not True
        or sealed.get("producer_evaluation_reference_access") is not False
        or sealed.get("training_executed") is not False
        or sealed.get("admitted") is not False
        or sealed.get("formalized") is not False
        or "post-freeze-evaluation" not in evaluation_text_packet
    ):
        raise RuntimeError("separated packets were not measured")
    recorded = audit.audit_occurrence_visibility(
        selected_ids,
        visible,
        edges,
        locator_cid=locator_cid,
        locator_on_lookup=resolved["source_occurrence_cid"],
        pin_kinds=pin_kinds,
        governing_scope_verified=resolved["governing_scope_verified"],
        semantic_definition_binding_verified=resolved["semantic_definition_binding_verified"],
        producer_written_before_evaluation=sealed["producer_written_before_evaluation"],
        separate_packet_owners=sealed["separate_packet_owners"],
        producer_evaluation_reference_access=sealed["producer_evaluation_reference_access"],
        evaluation_metric_reference_access=report["metric_reference_access"],
    )
    if (
        not dropped_occurrence_refused
        or not hidden_unsupported_refused
        or not governing_edge_refused
        or not locator_edge_refused
        or not governing_scope_refused
        or recorded.get("schema") != audit.SCHEMA
        or recorded.get("authored_fixture_only") is not True
        or recorded.get("selected_occurrence_count") != 2
        or recorded.get("selected_occurrences_visible") is not True
        or recorded.get("unsupported_outcome_visible") is not True
        or recorded.get("missing_outcome_visible") is not True
        or recorded.get("unresolved_outcome_status") != "missing"
        or recorded.get("locator_retained_on_lookup") is not True
        or recorded.get("locator_promoted_to_edge") is not False
        or recorded.get("governing_definition_edge_count") != 0
        or recorded.get("governing_scope_verified") is not False
        or recorded.get("semantic_definition_binding_verified") is not False
        or recorded.get("producer_written_before_evaluation") is not True
        or recorded.get("separate_packet_owners") is not True
        or recorded.get("producer_evaluation_reference_access") is not False
        or recorded.get("evaluation_metric_reference_access") is not True
        or recorded.get("decoder_executed") is not False
        or recorded.get("training_executed") is not False
        or recorded.get("model_executed") is not False
        or recorded.get("worker_launched") is not False
        or recorded.get("federal_corpus_scored") is not False
        or recorded.get("legal_meaning_established") is not False
        or recorded.get("legal_acceptance_claimed") is not False
        or recorded.get("admitted") is not False
        or recorded.get("formalized") is not False
    ):
        raise RuntimeError("occurrence visibility audit did not keep the named boundary")
    return {
        "admitted": False,
        "authored_fixture_only": True,
        "decoder_executed": False,
        "dropped_occurrence_refused": True,
        "evaluation_metric_reference_access": True,
        "evaluation_owner": "post-freeze-evaluation",
        "federal_corpus_scored": False,
        "formalized": False,
        "governing_definition_edge_count": 0,
        "governing_edge_refused": True,
        "governing_scope_refused": True,
        "governing_scope_verified": False,
        "hidden_unsupported_refused": True,
        "legal_acceptance_claimed": False,
        "legal_meaning_established": False,
        "locator_edge_refused": True,
        "locator_promoted_to_edge": False,
        "locator_retained_on_lookup": True,
        "missing_outcome_visible": True,
        "model_executed": False,
        "producer_evaluation_reference_access": False,
        "producer_owner": "primary-training-inference",
        "producer_reference_refused": True,
        "producer_written_before_evaluation": True,
        "schema": "legal-occurrence-visibility-audit/v1",
        "selected_occurrence_count": 2,
        "selected_occurrences_visible": True,
        "semantic_definition_binding_verified": False,
        "separate_packet_owners": True,
        "training_executed": False,
        "unresolved_outcome_status": "missing",
        "unsupported_outcome_visible": True,
        "worker_launched": False,
    }
