"""Decide when a dependency-satisfied block can move without inventing work.

An empty blocked reason, or the authored packet and admission sentence, is a
dependency gate. A publication hold can be discharged from dependency evidence
that already separates qualification, engineering gain, and statutory
generalization. A review hold can be discharged only when every criterion
allows an explicit blocker. A reviewer-identity criterion is discharged only
when the dependency packet records blank answer slots and names no reviewer.
A profile comparison is recorded from owner source limits when that review
supplied no meanings. A representation task is recorded with no extension
when that comparison selected nothing and the current codec still round-trips
one formula, refuses unsupported forms, and keeps its fixed profile. A
scoped graph task is recorded when that same empty extension is still true
and the current graph contract retains companions and ordered arguments,
keeps equal spellings apart, and refuses person and event nodes. A family
route task is recorded when that extension is still empty and the current
deontic route keeps obligation and permission distinct, pins its import
closure, and runs no Lake build. A family-lowering task is recorded when
that route limit is already stored and the current relational profile
preserves structure, signature, and its deontic annotation, while
counted-frame text and DCEC ground fragments stay on separate supports.
A definition-correspondence task is recorded when that scoped graph stayed
unmerged and the current owners keep the two-unary alias unproved, retain
every declared argument and qualifier slot on a non-admitted graph, refuse
discarded qualifier text, and keep unresolved alternatives on separate
branches. A lookup task is recorded when that correspondence stayed
unproved and the current owners retain every searched partition, route,
empty table, unavailable placeholder, and competing candidate, while a
nearest-scope hit does not verify a governing scope or a justified
binding. A dependency-closure task is recorded when that lookup stayed
unverified and the current owners report an explicit missing-dependency
frontier for the declared population, while a new competing definition
makes the former unique resolution stale and ambiguous instead of
unioning the candidates. legal_dependency_inventory_exhaustive stays
false. A meaning-material task is recorded when that closure stayed
non-exhaustive and the current owners account for the selected text as
represented or explicitly unsupported, keep every rule referenced once,
retain a reversed rule order as an unapproved difference, and preserve
null conditions and exceptions, a temporal leaf, unresolved approval,
and the caller polarity. A byte-target task is recorded when that
codec stayed a 259-symbol byte export with no semantic-action extension
and the current example keeps source/context as the only model input,
leaves labels and embedding metadata separate, and fixes both eligibility
masks at zero before any fitting. A complete-interpretation review is
recorded only as an unreviewed representation-phase boundary when that
meaning stayed semantically incomplete and the graph-review owner still
refuses an accepted view. No independent review is supplied, no omission
audit is claimed, and representation learning does not authorize legal
acceptance. A graph round-trip is recorded when that unreviewed
boundary is already stored and the current candidate graph keeps
closed endpoints, role ordinals, source and interpretation companions,
and distinct alternatives through a bundle reload. Review receipts stay
separately addressed and no assertion is accepted. A snapshot seal is
recorded when that round-trip stayed unaccepted and the current bundle
commits its own payload files under an independently supplied manifest
hash, refuses an incomplete bundle, and leaves external source,
interpretation, and evidence payloads unauthenticated. A missing declared
dependency stays explicit and snapshot custody does not promote candidate
assertions or unsupported imports. A shared-module task is recorded
when that custody stayed unpromoted and the current owners reuse
library target legal, keep each declaration's graph, interpretation,
and source identity, split the corpus into several module files, and
run no Lake build or admission. A comparison cell is recorded when the
width-control specification granted no call and no authority and the
current producer returns the locked canonical 8D sparse projection, the
locked 384D settings receipt, or the locked 768D complete-encoder
identity. Those calls do not admit a meaning, score a federal corpus, or
start training. A modality hypothesis is not selected from a retained
training recipe. This module does not write status, set admission flags,
or promote gold.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence

COMPLETED_DEPENDENCY_STATUSES = frozenset(
    {"completed", "complete", "done", "skipped"}
)
DEPENDENCY_RESOLUTION_OPERATION = (
    "dependency_block_resolved_from_current_evidence"
)
PUBLICATION_NON_PROMOTION_OPERATION = (
    "dependency_block_resolved_publication_non_promotion"
)
EXPLICIT_BLOCKER_OPERATION = "dependency_block_resolved_explicit_blocker"
REVIEW_EXPLICIT_BLOCKER_OPERATION = (
    "dependency_block_resolved_review_explicit_blocker"
)
PROFILE_LIMIT_OPERATION = (
    "dependency_block_resolved_profile_limits_from_owners"
)
CODEC_INTACT_OPERATION = (
    "dependency_block_resolved_codec_intact_no_extension"
)
SCOPED_GRAPH_OPERATION = (
    "dependency_block_resolved_scoped_graph_distinctions"
)
FAMILY_ROUTE_OPERATION = "dependency_block_resolved_family_route_limits"
FAMILY_LOWERING_OPERATION = (
    "dependency_block_resolved_family_lowering_boundaries"
)
DEFINITION_CORRESPONDENCE_OPERATION = (
    "dependency_block_resolved_definition_correspondence"
)
LOOKUP_PARTITION_OPERATION = (
    "dependency_block_resolved_lookup_partition_retention"
)
DECLARED_CLOSURE_OPERATION = (
    "dependency_block_resolved_declared_closure_frontier"
)
DECLARED_MEANING_OPERATION = (
    "dependency_block_resolved_declared_meaning_retention"
)
BYTE_TARGET_OPERATION = (
    "dependency_block_resolved_byte_target_eligibility_boundary"
)
GRAPH_ROUNDTRIP_OPERATION = (
    "dependency_block_resolved_graph_roundtrip_boundaries"
)
SNAPSHOT_CUSTODY_OPERATION = (
    "dependency_block_resolved_snapshot_custody_boundaries"
)
SHARED_MODULE_OPERATION = (
    "dependency_block_resolved_shared_module_boundaries"
)
V2_OWNER_ENVELOPE_OPERATION = (
    "dependency_block_resolved_v2_owner_envelope_boundaries"
)
TYPED_DECODER_OPERATION = (
    "dependency_block_resolved_typed_decoder_controls"
)
SHARED_LAKE_BUILD_OPERATION = (
    "dependency_block_resolved_shared_lake_fixture_build"
)
CONSEQUENCE_AUDIT_OPERATION = (
    "dependency_block_resolved_nonvacuous_consequence_boundaries"
)
INVALIDATION_KEYS_OPERATION = (
    "dependency_block_resolved_invalidation_key_boundaries"
)
INCREMENTAL_LAKE_AGREEMENT_OPERATION = (
    "dependency_block_resolved_incremental_lake_agreement"
)
RELEASE_COORDINATOR_OPERATION = (
    "dependency_block_resolved_release_coordinator_boundaries"
)
MODALITY_CAUSAL_COMPARISON_OPERATION = (
    "dependency_block_resolved_modality_causal_comparison_boundaries"
)
MODALITY_ERROR_REPORT_OPERATION = (
    "dependency_block_resolved_modality_error_report_boundaries"
)
TRAINING_METRIC_OPERATION = (
    "dependency_block_resolved_training_metric_boundaries"
)
WIDTH_CONTROL_OPERATION = (
    "dependency_block_resolved_width_control_boundaries"
)
CONTEXT_BUDGET_OPERATION = (
    "dependency_block_resolved_context_budget_boundaries"
)
OCCURRENCE_VISIBILITY_OPERATION = (
    "dependency_block_resolved_occurrence_visibility_boundaries"
)
LABEL_MASK_OPERATION = (
    "dependency_block_resolved_label_mask_boundaries"
)
TEACHER_IDENTITY_OPERATION = (
    "dependency_block_resolved_teacher_identity_boundaries"
)
POPULATION_COVERAGE_OPERATION = (
    "dependency_block_resolved_population_coverage_boundaries"
)
STORAGE_CONCURRENCY_OPERATION = (
    "dependency_block_resolved_storage_concurrency_boundaries"
)
SPARSE8_COMPARISON_OPERATION = (
    "dependency_block_resolved_sparse8_comparison_cell"
)
NATIVE384_COMPARISON_OPERATION = (
    "dependency_block_resolved_native384_comparison_cell"
)
NATIVE768_COMPARISON_OPERATION = (
    "dependency_block_resolved_native768_comparison_cell"
)
UNREVIEWED_REPRESENTATION_OPERATION = (
    "dependency_block_resolved_unreviewed_representation_phase"
)
RESOLUTION_OBLIGATION_OPERATION = (
    "dependency_block_resolved_resolution_obligation"
)
RESOLUTION_PARENT_OPERATION = (
    "dependency_block_resolved_resolution_obligations"
)
RESOLUTION_WORK_OPERATION = "dependency_block_resolved_resolution_work"
RESTATEMENT_RETIRED_OPERATION = "dependency_block_restatement_retired"
_RESOLUTION_OPERATIONS = frozenset(
    {
        DEPENDENCY_RESOLUTION_OPERATION,
        PUBLICATION_NON_PROMOTION_OPERATION,
        EXPLICIT_BLOCKER_OPERATION,
        REVIEW_EXPLICIT_BLOCKER_OPERATION,
        PROFILE_LIMIT_OPERATION,
        CODEC_INTACT_OPERATION,
        SCOPED_GRAPH_OPERATION,
        FAMILY_ROUTE_OPERATION,
        FAMILY_LOWERING_OPERATION,
        DEFINITION_CORRESPONDENCE_OPERATION,
        LOOKUP_PARTITION_OPERATION,
        DECLARED_CLOSURE_OPERATION,
        DECLARED_MEANING_OPERATION,
        BYTE_TARGET_OPERATION,
        GRAPH_ROUNDTRIP_OPERATION,
        SNAPSHOT_CUSTODY_OPERATION,
        SHARED_MODULE_OPERATION,
        V2_OWNER_ENVELOPE_OPERATION,
        TYPED_DECODER_OPERATION,
        SHARED_LAKE_BUILD_OPERATION,
        CONSEQUENCE_AUDIT_OPERATION,
        INVALIDATION_KEYS_OPERATION,
        INCREMENTAL_LAKE_AGREEMENT_OPERATION,
        RELEASE_COORDINATOR_OPERATION,
        MODALITY_CAUSAL_COMPARISON_OPERATION,
        MODALITY_ERROR_REPORT_OPERATION,
        TRAINING_METRIC_OPERATION,
        WIDTH_CONTROL_OPERATION,
        CONTEXT_BUDGET_OPERATION,
        OCCURRENCE_VISIBILITY_OPERATION,
        LABEL_MASK_OPERATION,
        TEACHER_IDENTITY_OPERATION,
        POPULATION_COVERAGE_OPERATION,
        STORAGE_CONCURRENCY_OPERATION,
        SPARSE8_COMPARISON_OPERATION,
        NATIVE384_COMPARISON_OPERATION,
        NATIVE768_COMPARISON_OPERATION,
        UNREVIEWED_REPRESENTATION_OPERATION,
        RESOLUTION_OBLIGATION_OPERATION,
        RESOLUTION_PARENT_OPERATION,
        RESOLUTION_WORK_OPERATION,
        RESTATEMENT_RETIRED_OPERATION,
    }
)

EXPECTED_DEFINITION_CORRESPONDENCE_PROBE: dict[str, object] = {
    "alias_assumed_not_proved": True,
    "alias_counted_as_operative_norm": False,
    "alias_norm_assertion_count": 0,
    "alias_semantics": "explicit_unreviewed_unary_biconditional_assumption",
    "alternative_ids": ["hypothesis", "hypothesis-b"],
    "alternatives_share_edge": False,
    "argument_count_per_branch": 6,
    "argument_roles": [
        "agent",
        "recipient",
        "agent",
        "recipient",
        "approver",
        "recipient",
    ],
    "arguments_missing_from_graph": 0,
    "body_roles": ["definition_body/0000", "definition_body/0001"],
    "body_roles_on_graph": True,
    "graph_admitted": False,
    "graph_edge_count": 26,
    "graph_formalized": False,
    "graph_node_count": 29,
    "graph_norm_assertion_count": 0,
    "graph_source_semantics_verified": False,
    "graph_view": "candidate",
    "native_build_executed": False,
    "qualifier_facets": ["conditions", "exceptions", "temporal"],
    "qualifier_refusal": (
        "complete norm paragraph coverage differs; extra text cannot be discarded"
    ),
    "qualifier_slots_missing_from_graph": 0,
    "qualifier_slots_per_branch": 3,
    "qualifier_text_discarded": False,
    "qualifier_values_null": True,
    "source_anchor_counts": [0, 1, 0, 1, 1, 1],
    "source_identity_asserted": False,
    "unresolved_codes": [
        "approval_necessity_unresolved",
        "parameter_roles_and_quantifiers_unreviewed",
        "passive_subject_agent_unresolved",
    ],
    "unresolved_on_graph": True,
}
_BOILERPLATE_REASON = (
    "requires a reviewed current job packet, exact eligible inputs "
    "and fresh purpose-specific native admission."
)
_SUBSTANTIVE_STAGES = frozenset({"semantic_review", "publication"})
_SUBSTANTIVE_PHRASES = (
    "cannot invent",
    "invent review",
    "fabricat",
    "reviewer identity",
    "actual independent",
    "independent review",
    "independent acceptance",
    "legal/formal reviewer",
    "gold/teacher",
    "gold or teacher",
    "no automatic runtime",
)
_AUTHORITY_FLAGS = (
    "admitted",
    "formalized",
    "gold_or_teacher_admission_granted",
    "constitution_roundtrip_ok",
)


def _text(value: object) -> str:
    return str(value or "").strip()


def _stage_packets(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        parts = value.split(",")
    elif isinstance(value, (list, tuple)):
        parts = value
    else:
        return ()
    return tuple(
        item for item in (_text(part).lower() for part in parts) if item
    )


def dependency_release_decision(
    *,
    status: object,
    body: Mapping[str, object] | None,
    dependency_statuses: Sequence[object],
    active_block: bool,
    criteria: Sequence[object] = (),
) -> str:
    """Return ``release`` or a stable reason the task stays unchanged."""

    if _text(status).lower() != "blocked":
        return "not_blocked"
    if active_block:
        return "active_block"
    if not dependency_statuses:
        return "no_dependencies"
    if any(
        _text(item).lower() not in COMPLETED_DEPENDENCY_STATUSES
        for item in dependency_statuses
    ):
        return "dependencies_open"
    record = body if isinstance(body, Mapping) else {}
    for flag in _AUTHORITY_FLAGS:
        if record.get(flag) is True:
            return "authority_flag_hold"
    stage = _text(record.get("resource_stage")).lower()
    packets = _stage_packets(record.get("required_stage_packets"))
    if stage in _SUBSTANTIVE_STAGES or _SUBSTANTIVE_STAGES.intersection(packets):
        return "substantive_review_or_publication_hold"
    reason = _text(record.get("blocked_reason") or record.get("blocked reason"))
    haystack = " ".join((reason, *(_text(item) for item in criteria))).lower()
    for phrase in _SUBSTANTIVE_PHRASES:
        if phrase in haystack:
            return "substantive_authored_hold"
    normalized = " ".join(reason.lower().split())
    if normalized and normalized != _BOILERPLATE_REASON:
        return "unclassified_blocked_reason"
    return "release"


def catalogue_resolution_decision(
    *,
    status: object,
    body: Mapping[str, object] | None,
    dependency_statuses: Sequence[object],
    active_block: bool,
    criteria: Sequence[object] = (),
) -> str:
    """DuckDB scientific tasks opt in with manual completion and a packet flag.

    Other blocked tasks keep the caller's existing status machine. Markdown
    boards use :func:`dependency_release_decision` directly.
    """

    record = body if isinstance(body, Mapping) else {}
    if record.get("execution_packet_required") is not True:
        return "packet_not_required"
    if _text(record.get("completion")).lower() != "manual":
        return "not_manual"
    return dependency_release_decision(
        status=status,
        body=record,
        dependency_statuses=dependency_statuses,
        active_block=active_block,
        criteria=criteria,
    )


def markdown_blocked_task_can_reopen(
    metadata: Mapping[str, object] | None,
    *,
    depends_on_count: int,
) -> bool:
    """True when a markdown block is only an authored dependency gate."""

    if not isinstance(metadata, Mapping) or depends_on_count < 1:
        return False
    packets = metadata.get(
        "required stage packets", metadata.get("required_stage_packets", ())
    )
    decision = dependency_release_decision(
        status="blocked",
        body={
            "blocked_reason": _text(metadata.get("blocked reason")),
            "resource_stage": _text(
                metadata.get("resource stage") or metadata.get("resource_stage")
            ),
            "required_stage_packets": list(_stage_packets(packets)),
        },
        dependency_statuses=tuple("completed" for _ in range(depends_on_count)),
        active_block=False,
    )
    return decision == "release"


_RESTATEMENT_PREFIX = "Produce a recorded result that satisfies:"
_SHARED_PREDICATE = re.compile(
    r"^(?P<body>.+)\s+(?P<pred>(?:remain|remains|stay|stays)\s+\S.*)$",
    re.IGNORECASE,
)
_DENOMINATOR_MARKERS = (
    "full_selected_structure_diagnostics",
    "comparable_pair_structure_diagnostics",
    (
        "Full selected and comparable-pair rates are separate descriptive "
        "structural diagnostics; neither is reviewed accuracy or training "
        "qualification."
    ),
)
_SCOPE_MARKERS = (
    (
        "Generation dimension/checkpoint/encoder/settings fields and "
        "no-target flags are declarations only; no model, numerical "
        "execution, target-withholding or latent-only inference is "
        "authenticated."
    ),
    "neither is reviewed accuracy or training qualification.",
)
_METRIC_MARKERS = {
    "component_effects": "component_effects_remain_separate: true",
    "seed_uncertainty": "seed_uncertainty_remains_separate: true",
    "named_losses": "named_losses_remain_separate: true",
}
_DISTINCT_CELLS_SENTENCE = (
    "Cited comparison cells record distinct dimensions and distinct operations."
)
_OWNER_METRIC_SENTENCE = {
    "component_effects": (
        "The evaluation owner records component_effects_remain_separate: true "
        "beside the structure diagnostics."
    ),
    "seed_uncertainty": (
        "The evaluation owner records seed_uncertainty_remains_separate: true "
        "beside the structure diagnostics."
    ),
    "named_losses": (
        "The evaluation owner records named_losses_remain_separate: true "
        "beside the structure diagnostics."
    ),
}
_PREVIOUS_ARTIFACTS_SENTENCE = (
    "The coordinator receipt keeps previously published artifacts selected "
    "and records that remote upload did not advance the release."
)
_PARTIAL_UPLOAD_SENTENCE = (
    "A recorded partial or uncertain upload keeps the previous coherent "
    "release selected."
)
_REFERENCED_ARTIFACTS_SENTENCE = (
    "The coordinator receipt keeps referenced artifacts published before "
    "the receipt."
)
_FAILED_MODELS_SENTENCE = (
    "Failed models stay preserved as their own recorded set."
)
_MATERIAL_MEANING_SENTENCE = (
    "Accepted components record complete material meaning."
)
_PER_USE_BINDING_SENTENCE = (
    "Per-use bindings are recorded on the accepted components."
)
_BUILD_EVIDENCE_SENTENCE = (
    "Real build evidence is recorded for the accepted components."
)
_FACT_RULES = (
    (
        "material meaning",
        "legal_meaning_established",
        _MATERIAL_MEANING_SENTENCE,
        "recorded_legal_meaning",
    ),
    (
        "per-use binding",
        "semantic_definition_binding_verified",
        _PER_USE_BINDING_SENTENCE,
        "recorded_per_use_binding",
    ),
    (
        "build evidence",
        "lake_build_executed",
        _BUILD_EVIDENCE_SENTENCE,
        "recorded_build_evidence",
    ),
)
_METRIC_RECORDED_FLAGS = {
    "component_effects": "component_effects_recorded",
    "seed_uncertainty": "seed_uncertainty_recorded",
    "named_losses": "named_losses_recorded",
}


def _underlying_criterion(criterion: object) -> str:
    text = " ".join(_text(criterion).split())
    if text.startswith(_RESTATEMENT_PREFIX):
        text = text[len(_RESTATEMENT_PREFIX) :].strip()
    return text


def _split_criterion(text: str) -> list[str]:
    """Split one criterion into sentences and shared-predicate list items."""

    sentences: list[str] = []
    for chunk in text.split(";"):
        chunk = " ".join(chunk.split()).strip()
        if not chunk:
            continue
        for piece in re.split(r"(?<=[.])\s+(?=[A-Z])", chunk):
            piece = piece.strip()
            if piece:
                sentences.append(piece)
    result: list[str] = []
    for sentence in sentences:
        result.extend(_shared_predicate_items(sentence) or [sentence])
    return result


def _shared_predicate_items(sentence: str) -> list[str] | None:
    match = _SHARED_PREDICATE.match(sentence.strip())
    if match is None or "," not in match.group("body"):
        return None
    raw = [part.strip() for part in match.group("body").split(",") if part.strip()]
    if len(raw) < 2:
        return None
    last = raw[-1]
    if last.lower().startswith("and "):
        raw[-1] = last[4:].strip()
    else:
        parts = re.split(r"\s+and\s+", last, maxsplit=1, flags=re.IGNORECASE)
        if len(parts) == 2 and parts[0].strip() and parts[1].strip():
            raw[-1] = parts[0].strip()
            raw.append(parts[1].strip())
    items = [item for item in raw if item]
    if len(items) < 2:
        return None
    predicate = " ".join(match.group("pred").split())
    obligations: list[str] = []
    for item in items:
        obligation = f"{item} {predicate}"
        if not obligation.endswith("."):
            obligation += "."
        obligations.append(obligation)
    return obligations


def _metric_name(text: str) -> str | None:
    norm = _normalized(text)
    if "denominator" in norm or "separate" not in norm:
        return None
    hits = [
        name
        for name, phrase in (
            ("component_effects", "component effect"),
            ("seed_uncertainty", "seed uncertainty"),
            ("named_losses", "named loss"),
        )
        if phrase in norm
    ]
    if len(hits) != 1:
        return None
    return hits[0]


def _obligation_kind(text: str) -> str | None:
    norm = _normalized(text)
    exact = {
        _normalized(_DISTINCT_CELLS_SENTENCE): "distinct_comparison_records",
        _normalized(_PREVIOUS_ARTIFACTS_SENTENCE): "previous_artifacts_retained",
        _normalized(_PARTIAL_UPLOAD_SENTENCE): "partial_upload_policy_recorded",
        _normalized(_REFERENCED_ARTIFACTS_SENTENCE): "referenced_artifacts_kept",
        _normalized(_FAILED_MODELS_SENTENCE): "failed_models_preserved",
    }
    for metric, sentence in _OWNER_METRIC_SENTENCE.items():
        exact[_normalized(sentence)] = f"owner_metric_{metric}"
    for _phrase, _key, sentence, kind in _FACT_RULES:
        exact[_normalized(sentence)] = kind
    if norm in exact:
        return exact[norm]
    if (
        "denominator" in norm
        and "separate" in norm
        and "comparable" in norm
        and "component effect" not in norm
        and "seed uncertainty" not in norm
        and "named loss" not in norm
    ):
        return "denominator_separation"
    metric = _metric_name(text)
    if metric is not None:
        return metric
    if "source-conditioned" in norm and "latent bottleneck" in norm:
        return "source_conditioned_scope"
    if "remote" in norm and "bod" in norm and "hash" in norm:
        return "remote_body_match"
    return None


def _criterion_obligations(criteria: Sequence[object]) -> list[dict[str, str]]:
    unique: list[str] = []
    seen: set[str] = set()
    for criterion in criteria:
        text = _underlying_criterion(criterion)
        if not text:
            continue
        for obligation in _split_criterion(text):
            key = _normalized(obligation)
            if not key or key in seen:
                continue
            seen.add(key)
            unique.append(obligation)
    return [
        {"obligation": item, "kind": _obligation_kind(item) or "unresolved"}
        for item in unique
    ]


def obligation_slug(text: str, index: int) -> str:
    """Stable short suffix for one obligation task id."""

    stop = {
        "the",
        "a",
        "an",
        "and",
        "or",
        "of",
        "to",
        "that",
        "this",
        "remain",
        "remains",
        "stay",
        "stays",
        "separate",
        "selected",
        "its",
        "without",
        "can",
        "for",
        "from",
    }
    words = re.findall(r"[a-z0-9]+", _normalized(text))
    kept = [word for word in words if word not in stop][:4]
    slug = "-".join(kept) or f"obligation-{index}"
    return f"{index:02d}-{slug}"[:64]


def resolution_obligations(
    *,
    criteria: Sequence[object],
    body: Mapping[str, object] | None = None,
) -> list[dict[str, str]]:
    """Split one blocked goal into obligations the release can track.

    A single criterion that no resolver can attempt is left unchanged.
    A restatement of that criterion is not created. Two or more clauses
    become tasks even when none has a specialized resolver. Obligations
    that are already the resolution work, or a goal already decomposed,
    produce nothing.
    """

    record = body if isinstance(body, Mapping) else {}
    if (
        record.get("resolution_subgoal") is True
        or record.get("resolution_decomposed") is True
        or record.get("resolution_work") is True
    ):
        return []
    obligations = _criterion_obligations(criteria)
    if len(obligations) < 2:
        return []
    if all(
        item["kind"] == "unresolved" for item in obligations
    ) and _known_builder_criteria(criteria):
        return []
    return obligations


def _known_builder_criteria(criteria: Sequence[object]) -> bool:
    """True when an existing release builder owns the acceptance as a whole."""

    matchers = (
        codec_criteria_match,
        scoped_graph_criteria_match,
        family_route_criteria_match,
        family_lowering_criteria_match,
        definition_correspondence_criteria_match,
        lookup_partition_criteria_match,
        declared_closure_criteria_match,
        byte_target_criteria_match,
        declared_meaning_criteria_match,
        unreviewed_representation_criteria_match,
        graph_roundtrip_criteria_match,
        snapshot_custody_criteria_match,
        shared_module_criteria_match,
        v2_owner_envelope_criteria_match,
        typed_decoder_criteria_match,
        shared_lake_build_criteria_match,
        consequence_audit_criteria_match,
        invalidation_keys_criteria_match,
        incremental_lake_agreement_criteria_match,
        release_coordinator_criteria_match,
        modality_causal_comparison_criteria_match,
        modality_error_report_criteria_match,
        training_metric_criteria_match,
        width_control_criteria_match,
        context_budget_criteria_match,
        occurrence_visibility_criteria_match,
        label_mask_criteria_match,
        teacher_identity_criteria_match,
        population_coverage_criteria_match,
        storage_concurrency_criteria_match,
        sparse8_criteria_match,
        native384_criteria_match,
        native768_criteria_match,
    )
    return any(matcher(criteria) for matcher in matchers)


def _owner_with_markers(
    owners: Mapping[str, Mapping[str, object]] | None,
    markers: Sequence[str],
) -> Mapping[str, object] | None:
    if not isinstance(owners, Mapping):
        return None
    for record in owners.values():
        if not isinstance(record, Mapping):
            continue
        text = record.get("text")
        if isinstance(text, str) and all(marker in text for marker in markers):
            return record
    return None


def _remote_body_match_record(
    owners: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object] | None:
    if not isinstance(owners, Mapping):
        return None
    for record in owners.values():
        if not isinstance(record, Mapping):
            continue
        text = record.get("text")
        if not isinstance(text, str) or not text.lstrip().startswith("{"):
            continue
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            continue
        if not isinstance(data, dict):
            continue
        matched = data.get("records_with_current_advertised_lfs_match")
        total = data.get("asset_record_count")
        if (
            data.get("byte_exact_payload_verification_executed") is not True
            or data.get("payload_downloaded") is not True
            or isinstance(matched, bool)
            or isinstance(total, bool)
            or not isinstance(matched, int)
            or not isinstance(total, int)
            or total <= 0
            or matched != total
        ):
            continue
        return {
            "asset_record_count": total,
            "records_matched": matched,
            "byte_exact_payload_verification_executed": True,
            "payload_downloaded": True,
            "owner_path": _text(record.get("path")),
            "owner_sha256": _text(record.get("sha256")),
        }
    return None


def _source_conditioned_call_without_bottleneck(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    called = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if (
                mapping.get("mean_coordinate_mse_is_semantic_superiority") is True
                or mapping.get("latent_bottleneck_benefit_proven") is True
                or mapping.get("legal_acceptance_claimed") is True
            ):
                return False
            if mapping.get("model_executed") is True or mapping.get("producer_called") is True:
                called = True
    return called


def _obligation_evidence(
    *,
    obligation: str,
    kind: str,
    record: Mapping[str, object],
    extra: Mapping[str, object],
    dependency_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    return {
        "operation": RESOLUTION_OBLIGATION_OPERATION,
        "obligation": obligation,
        "obligation_kind": kind,
        "satisfied": True,
        "owner_path": _text(record.get("path")),
        "owner_sha256": _text(record.get("sha256")),
        "training_executed": False,
        "legal_acceptance_claimed": False,
        "federal_corpus_scored": False,
        "cited_dependency_evidence": _cited_evidence(dependency_evidence),
        **dict(extra),
        **_authority_false(),
    }


def resolution_obligation_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object] | None:
    """Record one obligation when the owner or join already satisfies it.

    Absence of a measurement is not satisfaction. Component effects, seed
    uncertainty, and named losses stay unresolved until the owner or a
    dependency records each one as its own separated fact. Distinct
    comparison cells do not record those metrics. A remote body match is
    satisfied only when byte-exact verification downloaded every required
    body.
    """

    if len(tuple(criteria)) != 1 or _any_flag_true(dependency_evidence):
        return None
    obligation = _underlying_criterion(next(iter(criteria)))
    kind = _obligation_kind(obligation)
    if kind == "denominator_separation":
        record = _owner_with_markers(owners, _DENOMINATOR_MARKERS)
        if record is None:
            return None
        return _obligation_evidence(
            obligation=obligation,
            kind=kind,
            record=record,
            extra={
                "selected_n_and_comparable_k_separate": True,
                "reviewed_accuracy": False,
                "component_effects_recorded": False,
                "seed_uncertainty_recorded": False,
                "named_losses_recorded": False,
            },
            dependency_evidence=dependency_evidence,
        )
    if kind in _METRIC_MARKERS:
        record = _owner_with_markers(
            owners,
            (
                _METRIC_MARKERS[kind],
                "full_selected_structure_diagnostics",
                "comparable_pair_structure_diagnostics",
            ),
        )
        if record is not None:
            return _obligation_evidence(
                obligation=obligation,
                kind=kind,
                record=record,
                extra={
                    "metric": kind,
                    "metric_remains_separate": True,
                    "reviewed_accuracy": False,
                },
                dependency_evidence=dependency_evidence,
            )
        if _separated_metric_measurement(kind, dependency_evidence) is None:
            return None
        return _obligation_evidence(
            obligation=obligation,
            kind=kind,
            record={"path": "", "sha256": ""},
            extra={
                "metric": kind,
                "metric_remains_separate": True,
                _METRIC_RECORDED_FLAGS[kind]: True,
                "reviewed_accuracy": False,
                "pooled_score_recorded": False,
            },
            dependency_evidence=dependency_evidence,
        )
    if kind == "source_conditioned_scope":
        record = _owner_with_markers(owners, _SCOPE_MARKERS)
        if record is None or not _source_conditioned_call_without_bottleneck(
            dependency_evidence
        ):
            return None
        return _obligation_evidence(
            obligation=obligation,
            kind=kind,
            record=record,
            extra={
                "source_conditioned_scope_qualified": True,
                "usefulness_measured": False,
                "latent_bottleneck_benefit_proven": False,
                "reviewed_accuracy": False,
            },
            dependency_evidence=dependency_evidence,
        )
    if kind == "remote_body_match":
        matched = _remote_body_match_record(owners)
        if matched is None:
            return None
        record = {
            "path": matched["owner_path"],
            "sha256": matched["owner_sha256"],
        }
        return _obligation_evidence(
            obligation=obligation,
            kind=kind,
            record=record,
            extra=matched,
            dependency_evidence=dependency_evidence,
        )
    return _recorded_fact_obligation(
        obligation,
        dependency_evidence,
    )


def _obligation_negated(text: str) -> bool:
    norm = f" {_normalized(text)} "
    return any(token in norm for token in (" not ", " no ", " without ", " never "))


def _matching_facts(obligation: str) -> list[tuple[str, str, str, str]]:
    if _obligation_negated(obligation):
        return []
    norm = _normalized(obligation)
    return [rule for rule in _FACT_RULES if rule[0] in norm]


def _fact_is_recorded(
    key: str,
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when one dependency body stores the positive fact.

    A false flag, a nested citation, or an authority flag does not count.
    Distinct comparison cells keep component effects, seed uncertainty, and
    named losses false, so those receipts do not satisfy this check.
    """

    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        body = item.get("body")
        if not isinstance(body, Mapping):
            continue
        if (
            body.get("legal_acceptance_claimed") is True
            or body.get("federal_corpus_acceptance_claimed") is True
        ):
            continue
        if body.get(key) is True:
            return True
    return False


def _separated_metric_measurement(
    kind: str,
    dependency_evidence: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    """Return a measurement that keeps one metric separate.

    The receipt must name the metric, record it, and refuse reviewed
    accuracy and a pooled score. Cells that only keep dimensions apart
    do not qualify.
    """

    flag = _METRIC_RECORDED_FLAGS.get(kind)
    if flag is None or _any_flag_true(dependency_evidence):
        return None
    for item in dependency_evidence:
        body = item.get("body")
        if not isinstance(body, Mapping):
            continue
        if body.get("legal_acceptance_claimed") is True:
            continue
        if (
            body.get("metric") == kind
            and body.get("metric_remains_separate") is True
            and body.get(flag) is True
            and body.get("reviewed_accuracy") is not True
            and body.get("pooled_score_recorded") is not True
            and body.get("cells_recorded_separately") is not True
        ):
            return body
    return None


def _recorded_fact_obligation(
    obligation: str,
    dependency_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    facts = _matching_facts(obligation)
    if not facts or any(
        not _fact_is_recorded(rule[1], dependency_evidence) for rule in facts
    ):
        return None
    return _obligation_evidence(
        obligation=obligation,
        kind="unresolved",
        record={"path": "", "sha256": ""},
        extra={
            "recorded_facts": [rule[1] for rule in facts],
            "reviewed_accuracy": False,
        },
        dependency_evidence=dependency_evidence,
    )


def recorded_fact_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Complete a high-level goal whose every clause is already recorded.

    Each clause must name one or more known positive facts, and every fact
    must be true on a dependency. A clause with no known fact, a negated
    clause, or a false flag leaves the goal blocked. No admission is granted.
    """

    if _known_builder_criteria(criteria) or _any_flag_true(dependency_evidence):
        return None
    obligations = _criterion_obligations(criteria)
    if not obligations:
        return None
    fact_kinds = {rule[3] for rule in _FACT_RULES}
    recorded: list[str] = []
    for item in obligations:
        if item["kind"] != "unresolved" and item["kind"] not in fact_kinds:
            return None
        facts = _matching_facts(item["obligation"])
        if not facts:
            return None
        for rule in facts:
            if not _fact_is_recorded(rule[1], dependency_evidence):
                return None
            recorded.append(rule[1])
    cited = _cited_evidence(dependency_evidence)
    if not cited:
        return None
    return {
        "operation": RESOLUTION_PARENT_OPERATION,
        "obligations_satisfied": [item["obligation"] for item in obligations],
        "recorded_facts": recorded,
        "satisfied": True,
        "training_executed": False,
        "legal_acceptance_claimed": False,
        "federal_corpus_scored": False,
        "reviewed_accuracy": False,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


def resolution_follow_on_work(kind: str, obligation: str) -> list[dict[str, str]]:
    """Concrete tasks for one obligation the current record does not satisfy.

    Each task names one fact. A task that only repeats the obligation is not
    created. The caller leaves the obligation blocked until every fact exists.
    """

    if kind in _METRIC_MARKERS:
        return [
            {
                "obligation": _DISTINCT_CELLS_SENTENCE,
                "kind": "distinct_comparison_records",
            },
            {
                "obligation": _OWNER_METRIC_SENTENCE[kind],
                "kind": f"owner_metric_{kind}",
            },
        ]
    norm = _normalized(obligation)
    if (
        kind == "unresolved"
        and "partial" in norm
        and "upload" in norm
        and "previous" in norm
    ):
        return [
            {
                "obligation": _PREVIOUS_ARTIFACTS_SENTENCE,
                "kind": "previous_artifacts_retained",
            },
            {
                "obligation": _PARTIAL_UPLOAD_SENTENCE,
                "kind": "partial_upload_policy_recorded",
            },
        ]
    if kind == "unresolved" and "preserve" in norm and "failed" in norm:
        return [
            {
                "obligation": _REFERENCED_ARTIFACTS_SENTENCE,
                "kind": "referenced_artifacts_kept",
            },
            {
                "obligation": _FAILED_MODELS_SENTENCE,
                "kind": "failed_models_preserved",
            },
        ]
    facts = _matching_facts(obligation) if kind == "unresolved" else []
    if len(facts) >= 2:
        return [
            {"obligation": sentence, "kind": fact_kind}
            for _phrase, _key, sentence, fact_kind in facts
        ]
    return []


def _comparison_cell_records(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> list[tuple[int, str]] | None:
    found: list[tuple[int, str]] = []
    seen: set[tuple[int, str]] = set()
    for item in dependency_evidence:
        body = item.get("body")
        if not isinstance(body, Mapping):
            continue
        if _any_flag_true((item,)):
            return None
        if body.get("legal_acceptance_claimed") is True:
            return None
        dimension = body.get("dimension")
        operation = body.get("operation")
        called = body.get("model_executed") is True or body.get("producer_called") is True
        if (
            isinstance(dimension, bool)
            or not isinstance(dimension, int)
            or dimension <= 0
            or not isinstance(operation, str)
            or not operation.strip()
            or not called
        ):
            continue
        pair = (dimension, operation.strip())
        if pair in seen:
            continue
        seen.add(pair)
        found.append(pair)
    if len(found) < 2:
        return None
    if len({item[0] for item in found}) != len(found):
        return None
    if len({item[1] for item in found}) != len(found):
        return None
    return found


def _coordinator_retention(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> Mapping[str, object] | None:
    if _any_flag_true(dependency_evidence):
        return None
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if (
                mapping.get("referenced_artifacts_published_before_receipt") is True
                and mapping.get("remote_upload_executed") is False
                and mapping.get("unpublished_artifact_refused") is True
                and mapping.get("legal_acceptance_claimed") is False
                and mapping.get("federal_corpus_acceptance_claimed") is False
            ):
                return mapping
    return None


def restatement_retirement(
    body: Mapping[str, Any] | None,
    owners: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, Any] | None:
    """Retire a blocked restatement the release can neither execute nor satisfy.

    A superseded missing-record task repeats its parent. An owner-marker task
    asks a pinned evaluation owner to contain a line the measurement path
    replaces. Retirement is not satisfaction: the parent stays blocked until
    its own recorded result exists.
    """

    record = body if isinstance(body, Mapping) else {}
    superseded = record.get("superseded_by")
    if (
        record.get("missing_record_subgoal") is True
        and record.get("resolution_decomposed") is True
        and isinstance(superseded, list)
        and bool(superseded)
    ):
        return _retirement_evidence(
            "superseded_missing_record",
            [str(item) for item in superseded],
        )
    kind = str(record.get("obligation_kind") or "")
    if record.get("resolution_work") is not True or not kind.startswith("owner_metric_"):
        return None
    if not isinstance(owners, Mapping) or not owners:
        return None
    if not any(
        isinstance(item, Mapping) and isinstance(item.get("text"), str) and item.get("text")
        for item in owners.values()
    ):
        return None
    marker = _METRIC_MARKERS.get(kind.removeprefix("owner_metric_"))
    if not isinstance(marker, str) or _owner_with_markers(owners, (marker,)) is not None:
        return None
    return _retirement_evidence("pinned_owner_marker_absent", [marker])


def _retirement_evidence(reason: str, detail: list[str]) -> dict[str, Any]:
    return {
        "operation": RESTATEMENT_RETIRED_OPERATION,
        "satisfied": False,
        "retirement_reason": reason,
        "retired_detail": detail,
        "reviewed_accuracy": False,
        "legal_acceptance_claimed": False,
        "training_executed": False,
        "federal_corpus_scored": False,
        **_authority_false(),
    }


def resolution_work_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]] | None,
) -> dict[str, object] | None:
    """Record one follow-on fact when that fact is already stored.

    Distinct comparison cells do not record component effects, seed
    uncertainty, or named losses. A coordinator receipt that kept the
    previously published artifacts does not record a partial upload or a
    preserved failed-model set.
    """

    if len(tuple(criteria)) != 1 or _any_flag_true(dependency_evidence):
        return None
    obligation = _underlying_criterion(next(iter(criteria)))
    kind = _obligation_kind(obligation)
    if kind == "distinct_comparison_records":
        cells = _comparison_cell_records(dependency_evidence)
        if cells is None:
            return None
        return _obligation_evidence(
            obligation=obligation,
            kind=kind,
            record={"path": "", "sha256": ""},
            extra={
                "operation": RESOLUTION_WORK_OPERATION,
                "cells_recorded_separately": True,
                "dimensions": [item[0] for item in cells],
                "operations": [item[1] for item in cells],
                "component_effects_recorded": False,
                "seed_uncertainty_recorded": False,
                "named_losses_recorded": False,
                "reviewed_accuracy": False,
                "pooled_score_recorded": False,
            },
            dependency_evidence=dependency_evidence,
        )
    if kind in {
        "owner_metric_component_effects",
        "owner_metric_seed_uncertainty",
        "owner_metric_named_losses",
    }:
        metric = kind.removeprefix("owner_metric_")
        record = _owner_with_markers(
            owners,
            (
                _METRIC_MARKERS[metric],
                "full_selected_structure_diagnostics",
                "comparable_pair_structure_diagnostics",
            ),
        )
        if record is None:
            return None
        return _obligation_evidence(
            obligation=obligation,
            kind=kind,
            record=record,
            extra={
                "operation": RESOLUTION_WORK_OPERATION,
                "metric": metric,
                "metric_remains_separate": True,
                "reviewed_accuracy": False,
            },
            dependency_evidence=dependency_evidence,
        )
    if kind in {"previous_artifacts_retained", "referenced_artifacts_kept"}:
        retained = _coordinator_retention(dependency_evidence)
        if retained is None:
            return None
        return _obligation_evidence(
            obligation=obligation,
            kind=kind,
            record={"path": "", "sha256": ""},
            extra={
                "operation": RESOLUTION_WORK_OPERATION,
                "previous_artifacts_retained": True,
                "referenced_artifacts_published_before_receipt": True,
                "remote_upload_executed": False,
                "remote_upload_advanced_release": False,
                "unpublished_artifact_refused": True,
                "partial_upload_outcome_recorded": False,
                "failed_models_preserved": False,
            },
            dependency_evidence=dependency_evidence,
        )
    if kind == "partial_upload_policy_recorded":
        if _any_flag_true(dependency_evidence):
            return None
        for item in dependency_evidence:
            for mapping in _mappings(item.get("body")):
                if (
                    mapping.get("partial_or_uncertain_upload_retained_previous_release")
                    is True
                    and mapping.get("previous_release_retained") is True
                    and mapping.get("legal_acceptance_claimed") is False
                ):
                    return _obligation_evidence(
                        obligation=obligation,
                        kind=kind,
                        record={"path": "", "sha256": ""},
                        extra={
                            "operation": RESOLUTION_WORK_OPERATION,
                            "partial_upload_outcome_recorded": True,
                            "previous_release_retained": True,
                        },
                        dependency_evidence=dependency_evidence,
                    )
        return None
    if kind == "failed_models_preserved":
        for item in dependency_evidence:
            for mapping in _mappings(item.get("body")):
                if (
                    mapping.get("failed_models_preserved") is True
                    and mapping.get("legal_acceptance_claimed") is False
                ):
                    return _obligation_evidence(
                        obligation=obligation,
                        kind=kind,
                        record={"path": "", "sha256": ""},
                        extra={
                            "operation": RESOLUTION_WORK_OPERATION,
                            "failed_models_preserved": True,
                        },
                        dependency_evidence=dependency_evidence,
                    )
        return None
    if kind.startswith("recorded_"):
        rule = next((item for item in _FACT_RULES if item[3] == kind), None)
        if rule is None or not _fact_is_recorded(rule[1], dependency_evidence):
            return None
        return _obligation_evidence(
            obligation=obligation,
            kind=kind,
            record={"path": "", "sha256": ""},
            extra={
                "operation": RESOLUTION_WORK_OPERATION,
                rule[1]: True,
                "reviewed_accuracy": False,
            },
            dependency_evidence=dependency_evidence,
        )
    return None


def resolved_parent_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Complete a decomposed goal only when every obligation is satisfied.

    A blocked sibling obligation leaves the parent unresolved. The receipt
    grants no admission, review, training, or legal acceptance.
    """

    obligations = _criterion_obligations(criteria)
    if not obligations or _any_flag_true(dependency_evidence):
        return None
    found: dict[str, Mapping[str, object]] = {}
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != RESOLUTION_OBLIGATION_OPERATION:
                continue
            if mapping.get("satisfied") is not True:
                continue
            key = _normalized(mapping.get("obligation"))
            if key:
                found[key] = mapping
    if any(_normalized(item["obligation"]) not in found for item in obligations):
        return None
    cited = _cited_evidence(dependency_evidence)
    if not cited:
        return None
    return {
        "operation": RESOLUTION_PARENT_OPERATION,
        "obligations_satisfied": [item["obligation"] for item in obligations],
        "satisfied": True,
        "training_executed": False,
        "legal_acceptance_claimed": False,
        "federal_corpus_scored": False,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


def _normalized(value: object) -> str:
    return " ".join(_text(value).lower().split())


def _criterion_disposition(criterion: object) -> str:
    """Classify one acceptance sentence. Unknown sentences stay unresolved."""

    text = _normalized(criterion)
    if (
        "reviewer identity" in text
        or "actual reviewer" in text
        or "independent review" in text
        or "independent acceptance" in text
    ):
        return "reviewer_required"
    if any(
        phrase in text
        for phrase in (
            "or explicit blocker",
            "or preserve a blocker",
            "or qualify an explicit extension",
        )
    ):
        return "explicit_blocker"
    if (
        "distinguish" in text
        and "qualification" in text
        and "generalization" in text
    ):
        return "distinction"
    if "no automatic" in text and any(
        phrase in text for phrase in ("gold", "promotion", "teacher")
    ):
        return "non_promotion"
    return "other"


def _mappings(value: object):
    if isinstance(value, Mapping):
        yield value
        for child in value.values():
            yield from _mappings(child)
    elif isinstance(value, (list, tuple)):
        for child in value:
            yield from _mappings(child)


def _stored_value(evidence: Sequence[Mapping[str, object]], key: str) -> object:
    for item in evidence:
        for mapping in _mappings(item.get("body")):
            if key not in mapping:
                continue
            found = mapping[key]
            if isinstance(found, bool) or isinstance(found, str):
                return found
            if isinstance(found, int):
                return found
    return None


def _any_flag_true(evidence: Sequence[Mapping[str, object]]) -> bool:
    for item in evidence:
        for mapping in _mappings(item.get("body")):
            for flag in _AUTHORITY_FLAGS:
                if mapping.get(flag) is True:
                    return True
    return False


def _authority_false() -> dict[str, object]:
    return {
        "completion_authority": False,
        "admitted": False,
        "formalized": False,
        "gold_or_teacher_admission_granted": False,
        "constitution_roundtrip_ok": False,
        "native_admission": False,
        "Lean_admitted": False,
        "training_authorized": False,
        "new_training_launched": False,
        "launcher_invoked": False,
        "reviewer_claimed": False,
        "decision_fabricated": False,
        "runtime_default_promoted": False,
        "teacher_promoted": False,
    }


_PROFILE_MARKERS = {
    "relational": (
        "1 <= symbol.arity <= 8",
        "ordered distinct bounded argument roles required",
        "only explicitly bound Object variables supported",
        "exactly binary connective required",
        "constants/functions/other terms unsupported",
        '"O": "obligation"',
    ),
    "counted_frame": (
        '"semantics": "arbitrary_index_without_order"',
        "No nonempty membership, temporal aggregation, deontic operator or source meaning is asserted.",
        "TimeFrame is an arbitrary index with no order, duration, simultaneity or calendar interpretation.",
        "Each object is counted by built-in identity within the same event/time pair, without occurrence multiplicity.",
        '"maximum": 3',
    ),
    "family": (
        '"supported_routes": ["deontic"]',
        '"cross_family_equivalence_verified": False',
        '"all_logic_families_supported": False',
        "no temporal operators",
        "no deontic operators",
    ),
}


def _profile_criteria(criteria: Sequence[object]) -> bool:
    texts = tuple(_normalized(item) for item in criteria)
    if len(texts) < 2:
        return False
    preserve = tuple(
        text for text in texts if "preserve" in text and "arity" in text and "time" in text
    )
    limits = tuple(
        text for text in texts if "limits" in text and "explicit" in text and "counted-frame" in text
    )
    return bool(preserve) and bool(limits) and len(preserve) + len(limits) == len(texts)


def _review_blocker_without_meanings(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != REVIEW_EXPLICIT_BLOCKER_OPERATION:
                continue
            identity = mapping.get("reviewer_identity")
            blanks = identity.get("blank_answer_slots") if isinstance(identity, Mapping) else None
            if (
                isinstance(identity, Mapping)
                and identity.get("identity_asserted") is False
                and isinstance(blanks, int)
                and not isinstance(blanks, bool)
                and blanks > 0
            ):
                return True
    return False


def profile_limit_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
) -> dict[str, object] | None:
    """Record owner profile limits when the review supplied no meanings.

    The comparison admits nothing. It quotes structural limits that are
    already written in the owner sources and records that the selected
    meaning set is empty.
    """

    if not _profile_criteria(criteria) or not _review_blocker_without_meanings(
        dependency_evidence
    ):
        return None
    if set(owners) != {"relational", "counted_frame", "family"}:
        return None
    limits: dict[str, object] = {}
    owner_sha256: dict[str, object] = {}
    for role, markers in _PROFILE_MARKERS.items():
        record = owners.get(role)
        if not isinstance(record, Mapping):
            return None
        text = record.get("text")
        digest = record.get("sha256")
        path = record.get("path")
        if not isinstance(text, str) or not isinstance(digest, str) or not isinstance(path, str):
            return None
        if not digest or any(marker not in text for marker in markers):
            return None
        limits[role] = list(markers)
        owner_sha256[role] = {"path": path, "sha256": digest}
    cited = [
        {
            "task_cid": _text(item.get("task_cid")),
            "digest": _text(item.get("digest")),
        }
        for item in dependency_evidence
        if _text(item.get("digest"))
    ]
    if not cited:
        return None
    return {
        "operation": PROFILE_LIMIT_OPERATION,
        "selected_meaning_count": 0,
        "meanings_approved": False,
        "review_disposition": "explicit_blocker",
        "preserved_owner_limits": limits,
        "counted_frame_semantics_separate": True,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


_CODEC_MARKERS = {
    "ast": (
        "opaque string/bytes payloads are rejected",
        "opaque unversioned payloads are rejected",
    ),
    "relational": (
        "TypedExpression must survive exact serialized reconstruction",
        "constants/functions/other terms unsupported",
        "unsupported relational node: ",
    ),
    "join": (
        "unsupported derived typed provenance kind",
    ),
}
EXPECTED_CODEC_PROBE = {
    "vocabulary_size": 259,
    "semantic_action_language": False,
    "grammar_constrained_generation": False,
    "source_binding": "unsupported_in_authored_profile",
    "empty_status": "incomplete",
    "empty_reason": "missing_bos_and_eos",
    "zero_status": "rejected",
    "zero_reason": "invalid_frame",
    "positive_replay": True,
    "typed_expression_retained": True,
    "exact_native_ast_roundtrip": True,
    "relational_authority_all_false": True,
    "admissions": 0,
    "extensions_added": 0,
    "source_bound": False,
    "native_build_executed": False,
    "human_reviews": 0,
    "node_count": 16,
    "quantifier_count": 3,
    "modal_node_count": 1,
    "predicate_occurrence_count": 3,
    "source_binding_refusal": (
        "source binding unsupported: authored fragment is not a statutory interpretation"
    ),
    "opaque_payload_refusal": (
        "LogicExtensionNode.payload must be a structured mapping; "
        "opaque string/bytes payloads are rejected"
    ),
    "reviewed_status_refusal": "authored unreviewed request schema required",
}


def codec_criteria_match(criteria: Sequence[object]) -> bool:
    """True when every criterion is one of the two representation sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    if len(texts) != 2:
        return False
    correspondence = any(
        "lossless" in text and "correspondence" in text and "unsupported refusals" in text
        for text in texts
    )
    intact = any(
        "codecs" in text
        and "intact" in text
        and "opaque atoms" in text
        and "omitted text" in text
        for text in texts
    )
    return correspondence and intact


def _empty_profile_comparison(dependency_evidence: Sequence[Mapping[str, object]]) -> bool:
    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != PROFILE_LIMIT_OPERATION:
                continue
            if (
                mapping.get("selected_meaning_count") == 0
                and mapping.get("meanings_approved") is False
                and mapping.get("review_disposition") == "explicit_blocker"
            ):
                return True
    return False


def codec_intact_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record no extension when the review selected no meanings.

    The probe must be the current codec and AST measurement. A selected
    meaning, a changed codec, or a missing refusal leaves the task blocked.
    """

    if (
        not codec_criteria_match(criteria)
        or not _empty_profile_comparison(dependency_evidence)
        or set(owners) != set(_CODEC_MARKERS)
        or probe != EXPECTED_CODEC_PROBE
    ):
        return None
    limits: dict[str, object] = {}
    owner_sha256: dict[str, object] = {}
    for role, markers in _CODEC_MARKERS.items():
        record = owners.get(role)
        if not isinstance(record, Mapping):
            return None
        text = record.get("text")
        digest = record.get("sha256")
        path = record.get("path")
        if not isinstance(text, str) or not isinstance(digest, str) or not isinstance(path, str):
            return None
        if not digest or any(marker not in text for marker in markers):
            return None
        limits[role] = list(markers)
        owner_sha256[role] = {"path": path, "sha256": digest}
    cited = [
        {
            "task_cid": _text(item.get("task_cid")),
            "digest": _text(item.get("digest")),
        }
        for item in dependency_evidence
        if _text(item.get("digest"))
    ]
    if not cited:
        return None
    return {
        "operation": CODEC_INTACT_OPERATION,
        "extensions_added": 0,
        "meanings_requiring_extension": 0,
        "meanings_approved": False,
        "selected_meaning_count": 0,
        "review_disposition": "explicit_blocker",
        "vocabulary_size": EXPECTED_CODEC_PROBE["vocabulary_size"],
        "semantic_action_language": False,
        "grammar_constrained_generation": False,
        "codec_source_binding": EXPECTED_CODEC_PROBE["source_binding"],
        "empty_frame": {
            "status": EXPECTED_CODEC_PROBE["empty_status"],
            "reason": EXPECTED_CODEC_PROBE["empty_reason"],
        },
        "zero_id": {
            "status": EXPECTED_CODEC_PROBE["zero_status"],
            "reason": EXPECTED_CODEC_PROBE["zero_reason"],
        },
        "positive_replay": True,
        "typed_expression_retained": True,
        "exact_native_ast_roundtrip": True,
        "relational_authority_all_false": True,
        "admissions": 0,
        "source_bound": False,
        "native_build_executed": False,
        "human_reviews": 0,
        "node_count": EXPECTED_CODEC_PROBE["node_count"],
        "quantifier_count": EXPECTED_CODEC_PROBE["quantifier_count"],
        "modal_node_count": EXPECTED_CODEC_PROBE["modal_node_count"],
        "predicate_occurrence_count": EXPECTED_CODEC_PROBE["predicate_occurrence_count"],
        "refusals": [
            {
                "case": "source_binding",
                "message": EXPECTED_CODEC_PROBE["source_binding_refusal"],
            },
            {
                "case": "opaque_payload",
                "message": EXPECTED_CODEC_PROBE["opaque_payload_refusal"],
            },
            {
                "case": "reviewed_status",
                "message": EXPECTED_CODEC_PROBE["reviewed_status_refusal"],
            },
        ],
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


def _review_explicit_blocker_evidence(
    criteria: Sequence[object],
    dispositions: Sequence[str],
    dependency_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Record a blank review packet as an explicit blocker, naming nobody."""

    blanks = _stored_value(dependency_evidence, "blank_answer_slots")
    remainder = set(dispositions) - {"reviewer_required"}
    cited = [
        {
            "task_cid": _text(item.get("task_cid")),
            "digest": _text(item.get("digest")),
        }
        for item in dependency_evidence
        if _text(item.get("digest"))
    ]
    if (
        not cited
        or not remainder
        or not remainder <= {"explicit_blocker"}
        or not isinstance(blanks, int)
        or isinstance(blanks, bool)
        or blanks <= 0
    ):
        return None
    return {
        "operation": REVIEW_EXPLICIT_BLOCKER_OPERATION,
        "reviewer_identity": {
            "status": "not_supplied",
            "identity_asserted": False,
            "blank_answer_slots": blanks,
        },
        "criteria": [
            {
                "ordinal": index,
                "disposition": (
                    "identity_not_supplied"
                    if disposition == "reviewer_required"
                    else "explicit_blocker"
                ),
                "criterion": _text(criterion),
            }
            for index, (criterion, disposition) in enumerate(
                zip(criteria, dispositions)
            )
        ],
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


def substantive_resolution_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
) -> dict[str, object] | None:
    """Build a non-admission evidence body from stored dependency facts.

    A publication hold is discharged only when every criterion is a
    distinction or a non-promotion prohibition and the dependency evidence
    already separates implementation qualification, engineering gain, and
    statutory generalization. A review hold is discharged when every
    criterion allows an explicit blocker. A reviewer-identity criterion is
    discharged only together with those blocker criteria, and only when the
    dependency packet already records blank answer slots. No reviewer is
    named and no approval is granted.
    """

    dispositions = tuple(_criterion_disposition(item) for item in criteria)
    if not dispositions:
        return None
    if _any_flag_true(dependency_evidence):
        return None
    if "reviewer_required" in dispositions:
        return _review_explicit_blocker_evidence(
            criteria, dispositions, dependency_evidence
        )
    cited = [
        {
            "task_cid": _text(item.get("task_cid")),
            "digest": _text(item.get("digest")),
        }
        for item in dependency_evidence
        if _text(item.get("digest"))
    ]
    if set(dispositions) <= {"distinction", "non_promotion"} and {
        "distinction",
        "non_promotion",
    } <= set(dispositions):
        criteria_verified = _stored_value(dependency_evidence, "criteria_verified")
        model_forwards = _stored_value(dependency_evidence, "model_forwards")
        training_executed = _stored_value(dependency_evidence, "training_executed")
        gained = _stored_value(dependency_evidence, "full_ir_gained_or_lost")
        winner = _stored_value(dependency_evidence, "winner_selected")
        heldout = _stored_value(dependency_evidence, "heldout_uscode_fidelity")
        correspondence = _stored_value(
            dependency_evidence, "official_source_correspondence"
        )
        qualified = isinstance(criteria_verified, int) and not isinstance(
            criteria_verified, bool
        ) and (model_forwards == 0 or training_executed is False)
        gain_separated = gained == 0 or winner is False
        generalization_separated = heldout is False or correspondence is False
        if not cited or not (
            qualified and gain_separated and generalization_separated
        ):
            return None
        qualification: dict[str, object] = {"criteria_verified": criteria_verified}
        if model_forwards == 0:
            qualification["model_forwards"] = 0
        if training_executed is False:
            qualification["training_executed"] = False
        gain: dict[str, object] = {}
        if gained == 0:
            gain["full_ir_gained_or_lost"] = 0
        if winner is False:
            gain["winner_selected"] = False
        generalization: dict[str, object] = {}
        if heldout is False:
            generalization["heldout_uscode_fidelity"] = False
        if correspondence is False:
            generalization["official_source_correspondence"] = False
        return {
            "operation": PUBLICATION_NON_PROMOTION_OPERATION,
            "implementation_qualification": qualification,
            "matched_engineering_gain": gain,
            "statutory_generalization": generalization,
            "cited_dependency_evidence": cited,
            **_authority_false(),
        }
    if set(dispositions) == {"explicit_blocker"} and cited:
        return {
            "operation": EXPLICIT_BLOCKER_OPERATION,
            "criteria": [
                {
                    "ordinal": index,
                    "disposition": "explicit_blocker",
                    "criterion": _text(criterion),
                }
                for index, criterion in enumerate(criteria)
            ],
            "cited_dependency_evidence": cited,
            **_authority_false(),
        }
    return None


def _cited_evidence(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> list[dict[str, str]]:
    return [
        {
            "task_cid": _text(item.get("task_cid")),
            "digest": _text(item.get("digest")),
        }
        for item in dependency_evidence
        if _text(item.get("digest"))
    ]


def _codec_added_no_extension(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when a dependency already recorded the intact 259-symbol codec."""

    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != CODEC_INTACT_OPERATION:
                continue
            if (
                mapping.get("extensions_added") == 0
                and mapping.get("meanings_requiring_extension") == 0
                and mapping.get("meanings_approved") is False
                and mapping.get("vocabulary_size") == 259
            ):
                return True
    return False


def _owner_limits(
    owners: Mapping[str, Mapping[str, object]],
    markers: Mapping[str, Sequence[str]],
) -> tuple[dict[str, object], dict[str, object]] | None:
    if set(owners) != set(markers):
        return None
    limits: dict[str, object] = {}
    owner_sha256: dict[str, object] = {}
    for role, required in markers.items():
        record = owners.get(role)
        if not isinstance(record, Mapping):
            return None
        text = record.get("text")
        digest = record.get("sha256")
        path = record.get("path")
        if not isinstance(text, str) or not isinstance(digest, str) or not isinstance(path, str):
            return None
        if not digest or any(marker not in text for marker in required):
            return None
        limits[role] = list(required)
        owner_sha256[role] = {"path": path, "sha256": digest}
    return limits, owner_sha256


_GRAPH_MARKERS = {
    "graph": (
        "nor merges equivalent entities nor qualifies legal meaning",
        "role edges require an explicit ordinal",
        "duplicate role argument position",
    ),
    "join": (
        "The existing source and relational owners retain their separate contracts.",
        "Membership object/event/time roles, equality endpoints and lexical binding are syntax facts, not reviewed legal identities or temporal scopes.",
        "does not encode the complete statutory norm, its modality, participants, duties or applicability.",
    ),
}
EXPECTED_SCOPED_GRAPH_PROBE = {
    "node_count": 9,
    "edge_count": 7,
    "equal_spelling_term_senses": 2,
    "equal_spelling_entities": 1,
    "distinct_sense_ids": 2,
    "senses_merged": False,
    "entity_remains_entity": True,
    "person_node_type": False,
    "event_node_type": False,
    "person_nodes": 0,
    "event_nodes": 0,
    "person_node_refused": "unsupported graph node type",
    "event_node_refused": "unsupported graph node type",
    "companions_retained": True,
    "source_version_retained": True,
    "interpretation_retained": True,
    "ordered_role_ordinals": "HAS_ACTION:0,HAS_ACTOR:0,HAS_OBJECT:0,HAS_OBJECT:1",
    "duplicate_role_refused": "duplicate role argument position",
    "action_edge_count": 1,
    "view": "candidate",
    "graph_admitted": False,
    "graph_formalized": False,
    "graph_source_semantics_verified": False,
}


def scoped_graph_criteria_match(criteria: Sequence[object]) -> bool:
    """True when the two criteria are the companion and non-merge sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    if len(texts) != 2:
        return False
    companions = any(
        "source/version/interpretation companions" in text and "ordered arguments" in text
        for text in texts
    )
    distinction = any(
        "equal spelling" in text and "ground persons" in text and "actual events" in text
        for text in texts
    )
    return companions and distinction


def scoped_graph_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record graph distinctions when the codec extension stayed empty.

    The probe must show separate equal spellings, retained companions and
    ordered arguments, and refused person and event nodes. A merged sense,
    an accepted person or event node, or a changed contract leaves the task
    blocked.
    """

    checked = _owner_limits(owners, _GRAPH_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    if (
        not scoped_graph_criteria_match(criteria)
        or not _codec_added_no_extension(dependency_evidence)
        or checked is None
        or probe != EXPECTED_SCOPED_GRAPH_PROBE
        or not cited
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": SCOPED_GRAPH_OPERATION,
        "companions_retained": True,
        "source_version_retained": True,
        "interpretation_retained": True,
        "ordered_role_ordinals": EXPECTED_SCOPED_GRAPH_PROBE["ordered_role_ordinals"],
        "equal_spelling_term_senses": 2,
        "equal_spelling_entities": 1,
        "senses_merged": False,
        "entity_remains_entity": True,
        "person_node_type": False,
        "event_node_type": False,
        "person_nodes": 0,
        "event_nodes": 0,
        "category_implies_person": False,
        "normative_action_establishes_event": False,
        "action_edge_count": 1,
        "person_node_refused": EXPECTED_SCOPED_GRAPH_PROBE["person_node_refused"],
        "event_node_refused": EXPECTED_SCOPED_GRAPH_PROBE["event_node_refused"],
        "duplicate_role_refused": EXPECTED_SCOPED_GRAPH_PROBE["duplicate_role_refused"],
        "graph_view": "candidate",
        "graph_admitted": False,
        "graph_formalized": False,
        "graph_source_semantics_verified": False,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


_FAMILY_MARKERS = {
    "routes": (
        "does not make it equivalent to the legal rule.",
        "broader family equivalence is unverified",
        "Quantifiers/cognition/events are deliberately outside this route.",
        '"supported_routes": ["deontic"]',
    ),
    "lake": (
        "Kernel checking does not establish law.",
        "preparation confers no build authority.",
        "Every child's explicit Interpretation, carrier, time, definition and alias premises remain unchanged.",
        "The conclusion is the parameterized modal action; it does not establish actual action truth or legal truth.",
    ),
}
EXPECTED_FAMILY_ROUTE_PROBE = {
    "obligation_label": "deontic:O",
    "permission_label": "deontic:P",
    "prohibition_label": "deontic:F",
    "obligation_operator": "O",
    "permission_operator": "P",
    "prohibition_operator": "F",
    "operators_distinct": True,
    "permission_rewritten_as_obligation": False,
    "exact_native_ast_roundtrip": True,
    "source_bound": False,
    "source_semantics_verified": False,
    "cross_family_equivalence_verified": False,
    "backend_executed": False,
    "all_logic_families_supported": False,
    "admitted": False,
    "supported_routes": ["deontic"],
    "source_semantics_matrix": False,
    "non_deontic_refusal": (
        "canonical O/P/F rules require the deontic route; "
        "broader family equivalence is unverified"
    ),
    "temporal_in_deontic_refusal": "temporal_operator_outside_deontic_fol",
    "distinct_operator_assumption": (
        "Deontic/cognitive operators are distinct interpretation parameters; "
        "no modal axioms or source truth are asserted."
    ),
    "no_source_truth_asserted": True,
    "lake_build_executed": False,
    "producer_pin_count": 17,
    "producer_pin_sha256": (
        "ddeb47f67bd8ee96317cee69c7d00c2ba0d65d271e76901c872ae41bba1a20a0"
    ),
}


def family_route_criteria_match(criteria: Sequence[object]) -> bool:
    """True when the two criteria are the deontic-limit and closure sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    if len(texts) != 2:
        return False
    deontic = any(
        "obligations are not actual actions" in text
        and "permissions are not duties" in text
        and "explicit hypotheses" in text
        for text in texts
    )
    closure = any(
        "import closure" in text
        and "axiom policy" in text
        and "cross-family equivalence" in text
        for text in texts
    )
    return deontic and closure


def family_route_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record route limits when the codec extension stayed empty.

    The probe must keep obligation and permission as different modal labels,
    refuse a non-deontic relabeling, and record no Lake build. Identifying
    permission with obligation, or a changed import pin, leaves the task
    blocked.
    """

    checked = _owner_limits(owners, _FAMILY_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    if (
        not family_route_criteria_match(criteria)
        or not _codec_added_no_extension(dependency_evidence)
        or checked is None
        or probe != EXPECTED_FAMILY_ROUTE_PROBE
        or not cited
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": FAMILY_ROUTE_OPERATION,
        "obligation_label": "deontic:O",
        "permission_label": "deontic:P",
        "prohibition_label": "deontic:F",
        "operators_distinct": True,
        "permission_rewritten_as_obligation": False,
        "modal_action_establishes_actual_action_truth": False,
        "no_source_truth_asserted": True,
        "premises_recorded_as_explicit_assumptions": True,
        "temporal_operator_outside_deontic": True,
        "events_outside_ground_route": True,
        "supported_routes": ["deontic"],
        "cross_family_equivalence_verified": False,
        "all_logic_families_supported": False,
        "non_deontic_refusal": EXPECTED_FAMILY_ROUTE_PROBE["non_deontic_refusal"],
        "distinct_operator_assumption": EXPECTED_FAMILY_ROUTE_PROBE[
            "distinct_operator_assumption"
        ],
        "lake_build_executed": False,
        "producer_pin_count": EXPECTED_FAMILY_ROUTE_PROBE["producer_pin_count"],
        "producer_pin_sha256": EXPECTED_FAMILY_ROUTE_PROBE["producer_pin_sha256"],
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


_FAMILY_LOWERING_MARKERS = {
    "relational": (
        "This variables-only fragment does not translate statutory text.",
        "independent interpretation parameters.",
        "ordered distinct bounded argument roles required",
        "exact O/P/F payload and features required",
    ),
    "counted": (
        "TimeFrame is an arbitrary index with no order, duration, simultaneity or calendar interpretation.",
        "recognized_typed_FOL_to_specialized_Lean_text",
        "native_TDFOL_route_used",
        "Only the existing fixed authored counted frame is represented",
    ),
    "routes": (
        "does not make it equivalent to the legal rule.",
        "Quantifiers/cognition/events are deliberately outside this route.",
        "calendar_to_FOL_or_LTL_without_clock_translation",
        "DCEC_quantified_or_cognitive_or_event_semantics",
    ),
}
EXPECTED_FAMILY_LOWERING_PROBE = {
    "structural_mapping_rows": 20,
    "structural_paths_unique": True,
    "node_count": 16,
    "quantifier_count": 3,
    "modal_node_count": 1,
    "predicate_occurrence_count": 3,
    "signature_preserved": True,
    "annotation_operator": "P",
    "annotation_kind": "legacy.deontic/v1",
    "exact_native_ast_roundtrip": True,
    "source_bound": False,
    "source_translation_performed": False,
    "relational_authority_all_false": True,
    "extra_annotation_refused": "valid exact TypedExpression required",
    "signature_mismatch_refused": "ordered distinct bounded argument roles required",
    "counted_profile": "authored-at-most-three-distinct-per-event-and-time/v1",
    "counted_lowering_route": "recognized_typed_FOL_to_specialized_Lean_text",
    "counted_native_tdfol_route_used": False,
    "counted_print_stable": True,
    "counted_nodes": 49,
    "counted_equality_occurrences": 6,
    "counted_membership_occurrences": 4,
    "counted_source_bound": False,
    "counted_native_build_executed": False,
    "counted_authority_all_false": True,
    "counted_calendar_interpretation": False,
    "dcec_family": "dcec",
    "dcec_printed": "O(retain(Agency, Record))",
    "dcec_exact_native_ast_roundtrip": True,
    "dcec_source_bound": False,
    "dcec_cross_family_equivalence_verified": False,
    "dcec_backend_executed": False,
    "dcec_admitted": False,
    "quantified_dcec_refused": (
        "DCEC route requires nonempty ground predicate arguments"
    ),
    "cognitive_dcec_refused": (
        "DCEC route requires nonempty ground predicate arguments"
    ),
    "equality_in_dcec_refused": "legacy.unknown_character",
    "calendar_translation_unsupported": (
        "calendar_to_FOL_or_LTL_without_clock_translation"
    ),
    "cognitive_translation_unsupported": (
        "DCEC_quantified_or_cognitive_or_event_semantics"
    ),
    "equality_translated_into_dcec": False,
    "profiles_share_support_boundary": False,
    "non_deontic_refusal": (
        "canonical O/P/F rules require the deontic route; "
        "broader family equivalence is unverified"
    ),
}


def family_lowering_criteria_match(criteria: Sequence[object]) -> bool:
    """True when the two criteria are the profile and support-boundary sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    if len(texts) != 2:
        return False
    preservation = any(
        "lossless structure" in text
        and "signature" in text
        and "annotation preservation" in text
        and "exact profiles" in text
        for text in texts
    )
    boundaries = any(
        "counted-frame" in text
        and "dcec" in text
        and "separate support boundaries" in text
        for text in texts
    )
    return preservation and boundaries


def _family_route_limited(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when a dependency already recorded the unpromoted deontic route."""

    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != FAMILY_ROUTE_OPERATION:
                continue
            if (
                mapping.get("lake_build_executed") is False
                and mapping.get("cross_family_equivalence_verified") is False
                and mapping.get("permission_rewritten_as_obligation") is False
                and mapping.get("supported_routes") == ["deontic"]
            ):
                return True
    return False


def family_lowering_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record separate lowering supports when the deontic route stayed limited.

    The probe must preserve relational structure, signature, and the deontic
    annotation, keep counted-frame equality out of DCEC, and refuse calendar,
    quantified, and cognitive translations. A shared support boundary or a
    changed probe leaves the task blocked.
    """

    checked = _owner_limits(owners, _FAMILY_LOWERING_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    if (
        not family_lowering_criteria_match(criteria)
        or not _family_route_limited(dependency_evidence)
        or checked is None
        or probe != EXPECTED_FAMILY_LOWERING_PROBE
        or not cited
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": FAMILY_LOWERING_OPERATION,
        "structure_preserved": True,
        "signature_preserved": True,
        "annotation_preserved": True,
        "annotation_operator": "P",
        "annotation_kind": "legacy.deontic/v1",
        "exact_native_ast_roundtrip": True,
        "extra_annotation_refused": EXPECTED_FAMILY_LOWERING_PROBE[
            "extra_annotation_refused"
        ],
        "signature_mismatch_refused": EXPECTED_FAMILY_LOWERING_PROBE[
            "signature_mismatch_refused"
        ],
        "counted_profile": EXPECTED_FAMILY_LOWERING_PROBE["counted_profile"],
        "counted_lowering_route": EXPECTED_FAMILY_LOWERING_PROBE[
            "counted_lowering_route"
        ],
        "counted_native_tdfol_route_used": False,
        "counted_equality_occurrences": 6,
        "counted_calendar_interpretation": False,
        "dcec_family": "dcec",
        "dcec_printed": EXPECTED_FAMILY_LOWERING_PROBE["dcec_printed"],
        "quantified_dcec_refused": EXPECTED_FAMILY_LOWERING_PROBE[
            "quantified_dcec_refused"
        ],
        "cognitive_dcec_refused": EXPECTED_FAMILY_LOWERING_PROBE[
            "cognitive_dcec_refused"
        ],
        "equality_in_dcec_refused": "legacy.unknown_character",
        "equality_translated_into_dcec": False,
        "calendar_translation_unsupported": EXPECTED_FAMILY_LOWERING_PROBE[
            "calendar_translation_unsupported"
        ],
        "cognitive_translation_unsupported": EXPECTED_FAMILY_LOWERING_PROBE[
            "cognitive_translation_unsupported"
        ],
        "profiles_share_support_boundary": False,
        "cross_family_equivalence_verified": False,
        "lake_build_executed": False,
        "source_bound": False,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


_DEFINITION_CORRESPONDENCE_MARKERS = {
    "definition_use": (
        "A whole local predicate is aliased by an unreviewed assumption; no identity or legal implication is proved.",
        "all body roles and contexts remain retained.",
        "explicit_unreviewed_unary_biconditional_assumption",
    ),
    "definition_norm": (
        "qualifiers, objects and extra modals unsupported",
        "extra text cannot be discarded",
        "bounded literal category name without hidden qualifiers required",
    ),
    "join": (
        "All original unresolved source issues remain unchanged, including those touched by a typed hypothesis.",
        "Predicate argument roles, lexical binding, quantifier order and modal/connective structure are syntax facts, not resolved legal entities.",
        "admitted graph, native execution or qualified target is supplied.",
    ),
}


def definition_correspondence_criteria_match(criteria: Sequence[object]) -> bool:
    """True when the two criteria are the alias and correspondence sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    if len(texts) != 2:
        return False
    alias = any(
        "two-unary-definition alias" in text
        and "complete operative norm" in text
        for text in texts
    )
    survival = any(
        "every argument and qualifier survives" in text
        and "source/ast/graph correspondence" in text
        and "unresolved alternatives retain separate branches" in text
        for text in texts
    )
    return alias and survival


def _scoped_graph_distinguished(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when a dependency kept equal spellings and refused person/event nodes."""

    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != SCOPED_GRAPH_OPERATION:
                continue
            if (
                mapping.get("senses_merged") is False
                and mapping.get("category_implies_person") is False
                and mapping.get("normative_action_establishes_event") is False
                and mapping.get("graph_admitted") is False
            ):
                return True
    return False


def definition_correspondence_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record retained arguments, qualifier slots, and separate alternatives.

    The probe must keep the two-unary alias unproved, place every declared
    argument and qualifier slot on a non-admitted graph, refuse discarded
    qualifier text, and keep unresolved alternatives on separate branches.
    A promoted alias, a dropped slot, a shared branch edge, or a changed
    probe leaves the task blocked.
    """

    checked = _owner_limits(owners, _DEFINITION_CORRESPONDENCE_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    if (
        not definition_correspondence_criteria_match(criteria)
        or not _scoped_graph_distinguished(dependency_evidence)
        or checked is None
        or probe != EXPECTED_DEFINITION_CORRESPONDENCE_PROBE
        or not cited
    ):
        return None
    limits, owner_sha256 = checked
    measured = EXPECTED_DEFINITION_CORRESPONDENCE_PROBE
    return {
        "operation": DEFINITION_CORRESPONDENCE_OPERATION,
        "alias_counted_as_operative_norm": False,
        "alias_assumed_not_proved": True,
        "alias_semantics": measured["alias_semantics"],
        "alias_norm_assertion_count": 0,
        "body_roles": list(measured["body_roles"]),
        "body_roles_on_graph": True,
        "alternative_ids": list(measured["alternative_ids"]),
        "alternatives_share_edge": False,
        "argument_count_per_branch": 6,
        "argument_roles": list(measured["argument_roles"]),
        "source_anchor_counts": list(measured["source_anchor_counts"]),
        "arguments_missing_from_graph": 0,
        "qualifier_facets": list(measured["qualifier_facets"]),
        "qualifier_slots_per_branch": 3,
        "qualifier_values_null": True,
        "qualifier_slots_missing_from_graph": 0,
        "qualifier_text_discarded": False,
        "qualifier_refusal": measured["qualifier_refusal"],
        "unresolved_codes": list(measured["unresolved_codes"]),
        "unresolved_on_graph": True,
        "graph_node_count": 29,
        "graph_edge_count": 26,
        "graph_norm_assertion_count": 0,
        "graph_view": "candidate",
        "graph_admitted": False,
        "graph_formalized": False,
        "graph_source_semantics_verified": False,
        "native_build_executed": False,
        "source_identity_asserted": False,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_LOOKUP_PARTITION_PROBE: dict[str, object] = {
    "accepted_lookup_supplied": False,
    "admitted": False,
    "binding_scope": (
        "complete_supplied_tables_conservative_not_exact_search_trace"
    ),
    "competing_candidate_ids": ["term:first", "term:second"],
    "competing_selection_made": False,
    "competing_status": "ambiguous",
    "empty_definition_count": 0,
    "empty_partition_availability": "available",
    "empty_selection_made": False,
    "empty_status": "unresolved",
    "exhaustive_search_verified": False,
    "farther_candidate_retained": True,
    "farther_symbol_selected": False,
    "formalized": False,
    "governing_definition_use_supplied": False,
    "governing_scope_verified": False,
    "governing_use_verified": False,
    "justified_binding_count": 0,
    "nearest_scope_path": ["section:near", "document:doc-A", "global"],
    "nearest_selected_symbol_id": "term:near",
    "nearest_status": "resolved",
    "registry_binding": (
        "explicit_query_route_and_complete_supplied_partition_tables"
    ),
    "route_availability": ["available", "unavailable"],
    "route_local_owner_status": "resolved",
    "route_local_owner_symbol_ids": ["term:local"],
    "route_partition_ids": ["local", "external"],
    "route_selection_admitted": False,
    "route_status": "unavailable",
    "searched_availability": ["available", "unavailable", "available"],
    "searched_partition_count": 3,
    "searched_partition_ids": [
        "searched-empty",
        "searched-missing",
        "searched-local",
    ],
    "semantic_definition_binding_verified": False,
    "source_semantics_verified": False,
    "unavailable_owner_resolution_present": False,
    "unavailable_partition_availability": "unavailable",
    "unavailable_selection_made": False,
    "unavailable_status": "unavailable",
}
_LOOKUP_PARTITION_MARKERS = {
    "lookup": (
        "including empty tables and\nunavailable placeholders.",
        "It does not discover governing scopes or legal\nprecedence.",
        "All supplied query partitions are conservative dependencies, not an\n"
        "exact trace of the resolver's searches.",
        "Equal unknown editions do not establish governing compatibility.",
    ),
    "scope": (
        "This adapter preserves a caller's existing normative-scope declaration.",
        "No approval necessity, cardinality frame, purpose, antecedent, "
        "plural identity, applicability or governing definition use is decided.",
        '"accepted_lookup": None, "governing_definition_use": None,',
    ),
}


def lookup_partition_criteria_match(criteria: Sequence[object]) -> bool:
    """True when the two criteria are the retention and nearest-scope sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    if len(texts) != 2:
        return False
    retained = any(
        "searched partitions" in text
        and "unavailable/empty" in text
        and "competing candidates" in text
        for text in texts
    )
    nearest = any(
        "nearest-scope lookup" in text
        and "governs the law" in text
        and "justified bindings" in text
        for text in texts
    )
    return retained and nearest


def _definition_correspondence_retained(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when a dependency kept the alias unproved on a non-admitted graph."""

    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != DEFINITION_CORRESPONDENCE_OPERATION:
                continue
            if (
                mapping.get("alias_counted_as_operative_norm") is False
                and mapping.get("graph_admitted") is False
                and mapping.get("qualifier_text_discarded") is False
                and mapping.get("alternatives_share_edge") is False
                and mapping.get("source_identity_asserted") is False
            ):
                return True
    return False


def lookup_partition_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record retained partitions and the refusal to govern from nearest scope.

    The probe must keep empty and unavailable states distinct, retain every
    searched route and competing candidate, and leave the nearest lexical hit
    unverified as a governing scope or a justified binding. A collapsed
    candidate, a dropped partition, a verified governing use, or a changed
    probe leaves the task blocked.
    """

    checked = _owner_limits(owners, _LOOKUP_PARTITION_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    if (
        not lookup_partition_criteria_match(criteria)
        or not _definition_correspondence_retained(dependency_evidence)
        or checked is None
        or probe != EXPECTED_LOOKUP_PARTITION_PROBE
        or not cited
    ):
        return None
    limits, owner_sha256 = checked
    measured = EXPECTED_LOOKUP_PARTITION_PROBE
    return {
        "operation": LOOKUP_PARTITION_OPERATION,
        "binding_scope": measured["binding_scope"],
        "registry_binding": measured["registry_binding"],
        "nearest_status": measured["nearest_status"],
        "nearest_selected_symbol_id": measured["nearest_selected_symbol_id"],
        "nearest_scope_path": list(measured["nearest_scope_path"]),
        "farther_symbol_selected": False,
        "governing_scope_verified": False,
        "semantic_definition_binding_verified": False,
        "justified_binding_count": 0,
        "competing_status": measured["competing_status"],
        "competing_candidate_ids": list(measured["competing_candidate_ids"]),
        "competing_selection_made": False,
        "farther_candidate_retained": True,
        "empty_status": measured["empty_status"],
        "empty_selection_made": False,
        "empty_partition_availability": measured["empty_partition_availability"],
        "empty_definition_count": 0,
        "unavailable_status": measured["unavailable_status"],
        "unavailable_owner_resolution_present": False,
        "unavailable_partition_availability": measured[
            "unavailable_partition_availability"
        ],
        "unavailable_selection_made": False,
        "route_partition_ids": list(measured["route_partition_ids"]),
        "route_availability": list(measured["route_availability"]),
        "route_status": measured["route_status"],
        "route_local_owner_status": measured["route_local_owner_status"],
        "route_local_owner_symbol_ids": list(
            measured["route_local_owner_symbol_ids"]
        ),
        "route_selection_admitted": False,
        "searched_partition_ids": list(measured["searched_partition_ids"]),
        "searched_availability": list(measured["searched_availability"]),
        "searched_partition_count": 3,
        "governing_use_verified": False,
        "accepted_lookup_supplied": False,
        "governing_definition_use_supplied": False,
        "admitted": False,
        "formalized": False,
        "source_semantics_verified": False,
        "exhaustive_search_verified": False,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_DECLARED_CLOSURE_PROBE: dict[str, object] = {
    "admitted": False,
    "candidate_ids": ["term:first", "term:new"],
    "closure_is_exhaustive_legal_theory": False,
    "coverage_scope": "exact_declared_artifacts_pins_and_edges_only",
    "duplicate_identity_refused": True,
    "former_lookup_stale": True,
    "former_partition_stale": True,
    "former_selected_symbol_id": "term:first",
    "former_status": "resolved",
    "frontier_stale_artifact_ids": ["rule", "doctrine"],
    "global_consistency_verified": False,
    "interpretations_unioned": False,
    "invalidation_schema": "legal-graph-reverse-invalidation/v1",
    "legal_dependency_inventory_exhaustive": False,
    "manifest_schema": "legal-graph-dependency-manifest/v1",
    "missing_dependencies": [
        {"dependency_id": "missing-source", "dependent_id": "rule"}
    ],
    "missing_dependency_retained": True,
    "population_complete": True,
    "proof_authority": False,
    "revised_selection_absent": True,
    "revised_status": "ambiguous",
    "source_semantics_verified": False,
    "stale_count": 2,
}
_DECLARED_CLOSURE_MARKERS = {
    "dependencies": (
        "Missing targets and unbound pins are explicit and prevent reuse.",
        "does not infer legal dependencies, establish an exhaustive legal theory, or\n"
        "prove that unchanged artifacts are semantically correct.",
        "A missing target is\n"
        "    retained as an open dependency rather than silently removed.",
        "legal_dependency_inventory_exhaustive",
        "mean no dependencies were declared, not that legal completeness was proved.",
    ),
    "lookup": (
        "globally duplicate definition or alias identity",
        "do not replace the owner outcome with a new search.",
    ),
}


def declared_closure_criteria_match(criteria: Sequence[object]) -> bool:
    """True when the two criteria are the frontier and non-union sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    if len(texts) != 2:
        return False
    frontier = any(
        "complete closure" in text
        and "blocked frontier" in text
        and "selected component" in text
        for text in texts
    )
    competing = any(
        "conflicting interpretations" in text
        and "indiscriminately unioned" in text
        and "competing definitions" in text
        and "unique resolutions" in text
        for text in texts
    )
    return frontier and competing


def _lookup_partition_retained(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when a dependency retained partitions without governing the law."""

    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != LOOKUP_PARTITION_OPERATION:
                continue
            if (
                mapping.get("governing_scope_verified") is False
                and mapping.get("semantic_definition_binding_verified") is False
                and mapping.get("competing_selection_made") is False
                and mapping.get("route_selection_admitted") is False
                and mapping.get("justified_binding_count") == 0
            ):
                return True
    return False


def declared_closure_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record a declared missing-dependency frontier and an unmerged competitor.

    The probe must keep the missing target, refuse an exhaustive legal
    inventory, and show that a new competing definition makes the former
    unique resolution stale and ambiguous. A unioned interpretation, an
    exhaustive inventory, a dropped frontier, or a changed probe leaves the
    task blocked.
    """

    checked = _owner_limits(owners, _DECLARED_CLOSURE_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    measured = EXPECTED_DECLARED_CLOSURE_PROBE
    if (
        not declared_closure_criteria_match(criteria)
        or not _lookup_partition_retained(dependency_evidence)
        or checked is None
        or probe != measured
        or not cited
        or measured["legal_dependency_inventory_exhaustive"] is not False
        or measured["closure_is_exhaustive_legal_theory"] is not False
        or measured["interpretations_unioned"] is not False
        or measured["proof_authority"] is not False
        or measured["admitted"] is not False
        or measured["population_complete"] is not True
        or measured["coverage_scope"]
        != "exact_declared_artifacts_pins_and_edges_only"
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": DECLARED_CLOSURE_OPERATION,
        "closure_is_exhaustive_legal_theory": False,
        "legal_dependency_inventory_exhaustive": False,
        "population_complete": True,
        "coverage_scope": measured["coverage_scope"],
        "manifest_schema": measured["manifest_schema"],
        "invalidation_schema": measured["invalidation_schema"],
        "missing_dependency_retained": True,
        "missing_dependencies": [
            dict(item) for item in measured["missing_dependencies"]
        ],
        "frontier_stale_artifact_ids": list(measured["frontier_stale_artifact_ids"]),
        "global_consistency_verified": False,
        "source_semantics_verified": False,
        "proof_authority": False,
        "former_status": measured["former_status"],
        "former_selected_symbol_id": measured["former_selected_symbol_id"],
        "revised_status": measured["revised_status"],
        "revised_selection_absent": True,
        "candidate_ids": list(measured["candidate_ids"]),
        "interpretations_unioned": False,
        "former_partition_stale": True,
        "former_lookup_stale": True,
        "stale_count": 2,
        "duplicate_identity_refused": True,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_DECLARED_MEANING_PROBE: dict[str, object] = {
    "approval_necessity_decided": False,
    "caller_polarity": "P",
    "changed_modal_operator": "O",
    "changed_polarity_distinct": True,
    "changed_source_meaning_selected": False,
    "conditions_attached": False,
    "coverage_dispositions": ["represented", "unsupported"],
    "declared_semantic_coverage": "partial",
    "exceptions_attached": False,
    "formal_norm_semantics_encoded": False,
    "join_approval_codes": [
        "passive_subject_agent_unresolved",
        "approval_necessity_unresolved",
    ],
    "join_authority_any_true": False,
    "join_conditions_attached": False,
    "join_exceptions_attached": False,
    "join_modal_operator": "P",
    "join_source_meaning_selected": False,
    "join_temporal_occurrence_id": "when",
    "literal_accounting_complete": True,
    "literal_accounting_scope": (
        "every selected source character assigned a caller coverage "
        "disposition; not semantic completeness"
    ),
    "normalization_approved": False,
    "omitted_rule_refused": True,
    "order_issue": "statement_and_occurrence_rule_order_differ",
    "reversed_rule_count": 2,
    "reversed_rule_ids": ["r1", "r2"],
    "reversed_statement_ids": ["r2", "r1"],
    "rule_count": 1,
    "semantic_completeness_claimed": False,
    "temporal_occurrence_id": "when",
    "unresolved_codes": ["approval_necessity_unresolved"],
}
_DECLARED_MEANING_MARKERS = {
    "scope": (
        "Full literal accounting includes unresolved and unsupported text; it is not complete semantic coverage.",
        "every selected source character assigned a caller coverage disposition; not semantic completeness",
        "No approval necessity, cardinality frame, purpose, antecedent, plural identity, applicability or governing definition use is decided.",
    ),
    "join": (
        "All original unresolved source issues remain unchanged, including those touched by a typed hypothesis.",
        "Predicate argument roles, lexical binding, quantifier order and modal/connective structure are syntax facts, not resolved legal entities.",
        "it does not establish natural-language semantics.",
    ),
}


EXPECTED_BYTE_TARGET_PROBE: dict[str, object] = {
    "ast_action_head_added": False,
    "byte_offset": 3,
    "codec_mutation_refused": True,
    "codec_profile": "authored-object-relational-deontic/v1",
    "eligibility_granted": False,
    "embedding_reference_declaration_only": True,
    "evaluation_gold_mask": 0,
    "evaluation_gold_qualified": False,
    "example_profile": "candidate-source-separated-relational-byte-target/v1",
    "example_replayed": True,
    "example_status": "candidate_target_prepared",
    "exposure_status_caller_declared": True,
    "fitting_executed": False,
    "grammar_constrained_generation": False,
    "model_executed": False,
    "model_input_fields": ["context", "source_text"],
    "semantic_action_language": False,
    "separate_from_model_input": True,
    "source_binding": "unsupported_in_authored_profile",
    "split": "unassigned",
    "target_ends_with_eos": True,
    "target_starts_with_bos": True,
    "teacher_qualified": False,
    "training_executed": False,
    "training_gold_qualified": False,
    "training_mask": 0,
    "training_qualified": False,
    "vocabulary_size": 259,
}
_BYTE_TARGET_MARKERS = {
    "codec": (
        "not a semantic action",
        "language, grammar-constrained decoder, source translator or training admission.",
    ),
    "example": (
        "This adapter creates neither reviewed meaning nor training/gold eligibility.",
        "The exact model input contains source/context only; labels, alternative declarations and embedding metadata are separate.",
        "A candidate byte target is not a reviewed meaning, semantic-action decoder or training-admission decision.",
    ),
}


def byte_target_criteria_match(criteria: Sequence[object]) -> bool:
    """True when the criteria are the byte-target and pre-fit policy sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    if len(texts) != 2:
        return False
    export = any(
        "fixed byte target" in text
        and "codec/example export" in text
        and "ast-action" in text
        and "own task" in text
        for text in texts
    )
    policy = any(
        "training/evaluation eligibility" in text
        and "permitted context" in text
        and "untouched-component" in text
        and "before fitting" in text
        for text in texts
    )
    return export and policy


def byte_target_eligibility_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record the fixed byte export and the zero eligibility masks.

    The AST-action head stays off this codec. Eligibility, gold, and fitting
    stay false. A changed probe or an extended codec leaves the task blocked.
    """

    checked = _owner_limits(owners, _BYTE_TARGET_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    measured = EXPECTED_BYTE_TARGET_PROBE
    if (
        not byte_target_criteria_match(criteria)
        or not _codec_added_no_extension(dependency_evidence)
        or checked is None
        or probe != measured
        or not cited
        or measured["semantic_action_language"] is not False
        or measured["ast_action_head_added"] is not False
        or measured["eligibility_granted"] is not False
        or measured["training_mask"] != 0
        or measured["evaluation_gold_mask"] != 0
        or measured["fitting_executed"] is not False
        or measured["training_executed"] is not False
        or measured["training_qualified"] is not False
        or measured["evaluation_gold_qualified"] is not False
        or measured["vocabulary_size"] != 259
        or measured["codec_mutation_refused"] is not True
        or measured["separate_from_model_input"] is not True
        or measured["model_input_fields"] != ["context", "source_text"]
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": BYTE_TARGET_OPERATION,
        "vocabulary_size": 259,
        "semantic_action_language": False,
        "grammar_constrained_generation": False,
        "ast_action_head_added": False,
        "codec_mutation_refused": True,
        "codec_profile": measured["codec_profile"],
        "example_profile": measured["example_profile"],
        "example_status": measured["example_status"],
        "example_replayed": True,
        "source_binding": measured["source_binding"],
        "model_input_fields": list(measured["model_input_fields"]),
        "separate_from_model_input": True,
        "split": "unassigned",
        "training_mask": 0,
        "evaluation_gold_mask": 0,
        "eligibility_granted": False,
        "training_qualified": False,
        "training_gold_qualified": False,
        "evaluation_gold_qualified": False,
        "teacher_qualified": False,
        "training_executed": False,
        "model_executed": False,
        "fitting_executed": False,
        "embedding_reference_declaration_only": True,
        "exposure_status_caller_declared": True,
        "byte_offset": 3,
        "target_starts_with_bos": True,
        "target_ends_with_eos": True,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


def declared_meaning_criteria_match(criteria: Sequence[object]) -> bool:
    """True when the criteria are the coverage and attachment sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    if len(texts) != 2:
        return False
    coverage = any(
        "meaning-bearing material" in text
        and "explicitly unsupported" in text
        and "rule counts" in text
        and "order" in text
        for text in texts
    )
    attachment = any(
        "conditions" in text
        and "exceptions" in text
        and "necessary approval" in text
        and "polarity" in text
        and "temporal attachment" in text
        for text in texts
    )
    return coverage and attachment


def _declared_closure_retained(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when a dependency kept an explicit frontier without a legal theory."""

    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != DECLARED_CLOSURE_OPERATION:
                continue
            if (
                mapping.get("legal_dependency_inventory_exhaustive") is False
                and mapping.get("closure_is_exhaustive_legal_theory") is False
                and mapping.get("interpretations_unioned") is False
                and mapping.get("proof_authority") is False
            ):
                return True
    return False


def declared_meaning_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record represented-or-unsupported coverage and preserved attachments.

    The probe must keep unsupported text explicit, refuse a missing rule,
    retain a reversed rule order without approving normalization, and preserve
    the temporal leaf, unresolved approval, and caller polarity. A decided
    approval, a selected meaning, or a changed probe leaves the task blocked.
    """

    checked = _owner_limits(owners, _DECLARED_MEANING_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    measured = EXPECTED_DECLARED_MEANING_PROBE
    if (
        not declared_meaning_criteria_match(criteria)
        or not _declared_closure_retained(dependency_evidence)
        or checked is None
        or probe != measured
        or not cited
        or measured["formal_norm_semantics_encoded"] is not False
        or measured["semantic_completeness_claimed"] is not False
        or measured["approval_necessity_decided"] is not False
        or measured["normalization_approved"] is not False
        or measured["join_source_meaning_selected"] is not False
        or measured["coverage_dispositions"] != ["represented", "unsupported"]
        or measured["literal_accounting_complete"] is not True
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": DECLARED_MEANING_OPERATION,
        "literal_accounting_complete": True,
        "literal_accounting_scope": measured["literal_accounting_scope"],
        "declared_semantic_coverage": "partial",
        "semantic_completeness_claimed": False,
        "formal_norm_semantics_encoded": False,
        "coverage_dispositions": list(measured["coverage_dispositions"]),
        "rule_count": 1,
        "omitted_rule_refused": True,
        "reversed_rule_ids": list(measured["reversed_rule_ids"]),
        "reversed_statement_ids": list(measured["reversed_statement_ids"]),
        "reversed_rule_count": 2,
        "order_issue": measured["order_issue"],
        "normalization_approved": False,
        "conditions_attached": False,
        "exceptions_attached": False,
        "temporal_occurrence_id": measured["temporal_occurrence_id"],
        "unresolved_codes": list(measured["unresolved_codes"]),
        "approval_necessity_decided": False,
        "caller_polarity": "P",
        "join_conditions_attached": False,
        "join_exceptions_attached": False,
        "join_temporal_occurrence_id": measured["join_temporal_occurrence_id"],
        "join_approval_codes": list(measured["join_approval_codes"]),
        "join_modal_operator": "P",
        "changed_modal_operator": "O",
        "changed_polarity_distinct": True,
        "join_source_meaning_selected": False,
        "changed_source_meaning_selected": False,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_GRAPH_ROUNDTRIP_PROBE: dict[str, object] = {
    "accepted_assertion_count": 0,
    "accepted_snapshot_refused": "v1 graph snapshots are candidate-only",
    "admitted": False,
    "alternative_identities_distinct": True,
    "alternative_ids": ["A", "B"],
    "alternatives_share_identity": False,
    "ast_node_type": False,
    "bundle_roundtrip": True,
    "candidate_assertion_count": 2,
    "candidate_view": "candidate",
    "cross_alternative_refused": "edge crosses mutually exclusive interpretations",
    "dangling_endpoint_refused": (
        "dangling graph endpoint; external imports unsupported in v1"
    ),
    "duplicate_role_ordinal_refused": "duplicate role argument position",
    "edge_count": 3,
    "endpoints_closed": True,
    "formalized": False,
    "graph_to_ast_embedded": False,
    "interpretation_companion_retained": True,
    "manifest_authority_false": True,
    "multiplicity_ordinals": [0, 1],
    "node_count": 7,
    "review_node_type": False,
    "review_receipt_authenticated": False,
    "review_receipt_retained": True,
    "role_ordinals": ["HAS_ACTOR:0", "HAS_OBJECT:0", "HAS_OBJECT:1"],
    "source_companion_retained": True,
    "source_semantics_verified": False,
}
EXPECTED_UNREVIEWED_REPRESENTATION_PROBE: dict[str, object] = {
    "accepted_supported_pilot": False,
    "accepted_view_eligible": False,
    "declarations_authenticate_rich_norm": False,
    "downstream_lean_kernel_admission_required": True,
    "downstream_semantic_validation_required": True,
    "human_identity_verified": False,
    "human_independence_verified": False,
    "independent_review_supplied": False,
    "legal_fidelity_claimed": False,
    "material_omission_audit_performed": False,
    "pilot_readme_present": False,
    "representation_learning_authorizes_legal_acceptance": False,
    "representation_phase": "self_supervised_autoformalization",
    "signature_establishes_legal_truth": False,
    "source_custody_is_legal_fidelity": False,
    "structural_validity_is_legal_fidelity": False,
}
_UNREVIEWED_REPRESENTATION_CRITERIA = frozenset(
    {
        "zero known material omission/addition in the accepted supported pilot.",
        (
            "unresolved material blocks this component; source custody and "
            "structural validity are not legal fidelity."
        ),
        (
            "require the exact supported complete interpretation review "
            "domain; existing source-response or unary-definition "
            "declarations alone cannot authenticate a rich norm/use subject."
        ),
    }
)


def unreviewed_representation_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the three complete-interpretation review sentences only."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 3 and frozenset(texts) == _UNREVIEWED_REPRESENTATION_CRITERIA


def _declared_meaning_unaccepted(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when declared meaning stayed partial and unapproved."""

    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != DECLARED_MEANING_OPERATION:
                continue
            if (
                mapping.get("semantic_completeness_claimed") is False
                and mapping.get("approval_necessity_decided") is False
                and mapping.get("join_source_meaning_selected") is False
            ):
                return True
    return False


def unreviewed_representation_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record that representation learning supplied no independent review.

    The declared-meaning dependency must already refuse semantic
    completeness, approval, and a selected join meaning. The live
    graph-review owner must still refuse an accepted view. No omission
    audit is claimed and no legal meaning is accepted.
    """

    if (
        not unreviewed_representation_criteria_match(criteria)
        or not _declared_meaning_unaccepted(dependency_evidence)
        or not isinstance(probe, Mapping)
    ):
        return None
    digest = probe.get("review_owner_sha256")
    measured = {
        key: probe.get(key)
        for key in EXPECTED_UNREVIEWED_REPRESENTATION_PROBE
    }
    if (
        measured != EXPECTED_UNREVIEWED_REPRESENTATION_PROBE
        or not isinstance(digest, str)
        or len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        return None
    cited = _cited_evidence(dependency_evidence)
    if not cited:
        return None
    return {
        "operation": UNREVIEWED_REPRESENTATION_OPERATION,
        **EXPECTED_UNREVIEWED_REPRESENTATION_PROBE,
        "review_owner_sha256": digest,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


_GRAPH_ROUNDTRIP_CRITERIA = frozenset(
    {
        (
            "exact endpoints, multiplicity, role ordinals, "
            "source/interpretation/review companions and graph-to-ast "
            "structure survive."
        ),
        (
            "accepted and candidate assertions remain separate; alternative "
            "theories retain distinct identity."
        ),
    }
)
_GRAPH_ROUNDTRIP_MARKERS = {
    "graph": (
        "closed endpoints",
        "dangling graph endpoint; external imports unsupported in v1",
        "role edges require an explicit ordinal",
        "duplicate role argument position",
        "alternative group and identity must be supplied together",
        "edge crosses mutually exclusive interpretations",
        "v1 graph snapshots are candidate-only",
        "Semantic ASTs and review receipts remain separately content-addressed inputs.",
        "source_occurrence_cids",
        "interpretation_cids",
    ),
    "exchange": (
        "Parquet records do not reconstruct the exact sealed snapshot",
        "No dataset upload or legal admission occurs here.",
        "writer readback differs from input snapshot",
        "graph bundle cannot claim authority",
    ),
}


def graph_roundtrip_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the two closed-graph round-trip sentences only."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _GRAPH_ROUNDTRIP_CRITERIA


def _unreviewed_representation_retained(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when representation learning stayed unreviewed and unaccepted."""

    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != UNREVIEWED_REPRESENTATION_OPERATION:
                continue
            if (
                mapping.get("independent_review_supplied") is False
                and mapping.get("accepted_view_eligible") is False
                and mapping.get("legal_fidelity_claimed") is False
                and mapping.get(
                    "representation_learning_authorizes_legal_acceptance"
                )
                is False
            ):
                return True
    return False


def graph_roundtrip_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record a candidate graph round-trip that accepts no assertion.

    Endpoints, role multiplicity, source and interpretation companions, and
    distinct alternatives must survive the bundle reload. Review receipts
    stay separately addressed. An admitted snapshot or a changed probe
    leaves the task blocked.
    """

    checked = _owner_limits(owners, _GRAPH_ROUNDTRIP_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    if (
        not graph_roundtrip_criteria_match(criteria)
        or not _unreviewed_representation_retained(dependency_evidence)
        or checked is None
        or probe != EXPECTED_GRAPH_ROUNDTRIP_PROBE
        or not cited
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": GRAPH_ROUNDTRIP_OPERATION,
        **EXPECTED_GRAPH_ROUNDTRIP_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_SNAPSHOT_CUSTODY_PROBE: dict[str, object] = {
    "admitted": False,
    "bundle_files": [
        "edges.parquet",
        "manifest.json",
        "nodes.parquet",
        "snapshot.json",
    ],
    "bundle_records_present": True,
    "bundle_roundtrip": True,
    "candidate_assertions_promoted": False,
    "candidate_view": "candidate",
    "coverage_scope": "exact_declared_artifacts_pins_and_edges_only",
    "dependency_admitted": False,
    "edge_count": 0,
    "existing_directory_refused": "could not write local graph bundle:",
    "external_payloads_in_bundle": 0,
    "external_referenced_records_authenticated": False,
    "external_references_verified": False,
    "formalized": False,
    "incomplete_bundle_refused": "incomplete bundle or unknown filenames",
    "interpretation_reference_retained": True,
    "legal_dependency_inventory_exhaustive": False,
    "manifest_identity_refused": (
        "dependency manifest differs from independently expected identity"
    ),
    "missing_dependency_count": 1,
    "missing_dependency_id": "missing-record",
    "node_count": 3,
    "payload_count": 3,
    "population_complete": True,
    "promoted_manifest_refused": (
        "dependency manifest differs from exact regeneration"
    ),
    "proof_authority": False,
    "reusable_under_declared_dependencies": [],
    "snapshot_view": "candidate",
    "source_reference_retained": True,
    "source_semantics_verified": False,
    "stale_artifact_ids": ["snapshot"],
    "wrong_manifest_refused": "bundle file SHA256 differs: manifest.json",
}
_SNAPSHOT_CUSTODY_CRITERIA = frozenset(
    {
        "all referenced records are present and independently committed.",
        (
            "snapshot custody does not promote candidate legal assertions or "
            "unsupported imports."
        ),
    }
)
_SNAPSHOT_CUSTODY_MARKERS = {
    "exchange": (
        "Failed attempts retain their payloads for inspection, without a committed",
        "Existing directories/files are never overwritten or reused.",
        "incomplete bundle or unknown filenames",
        "payloads, source occurrences or evidence merely named by those contents.",
        "graph bundle cannot claim authority",
        "external_references_verified",
        "externally supplied manifest SHA256",
    ),
    "dependencies": (
        "Replay a manifest only under an independently retained content pin.",
        "dependency manifest differs from independently expected identity",
        "retained as an open dependency rather than silently removed.",
        "observations must account for every artifact; use null for unavailable",
        "does not infer legal dependencies, establish an exhaustive legal theory",
        "exact_declared_artifacts_pins_and_edges_only",
        "legal_dependency_inventory_exhaustive",
    ),
}


def snapshot_custody_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the two snapshot-custody sentences only."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _SNAPSHOT_CUSTODY_CRITERIA


def _graph_roundtrip_unaccepted(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the stored graph round-trip accepted no assertion."""

    if _any_flag_true(dependency_evidence):
        return False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != GRAPH_ROUNDTRIP_OPERATION:
                continue
            if (
                mapping.get("accepted_assertion_count") == 0
                and mapping.get("graph_to_ast_embedded") is False
                and mapping.get("review_receipt_authenticated") is False
                and mapping.get("admitted") is False
                and mapping.get("formalized") is False
                and mapping.get("bundle_roundtrip") is True
                and mapping.get("endpoints_closed") is True
                and mapping.get("alternative_identities_distinct") is True
            ):
                return True
    return False


def snapshot_custody_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record snapshot custody that promotes no candidate assertion.

    The bundle's own payload files must be present and pinned by an
    independently supplied manifest hash. External source, interpretation,
    and evidence payloads stay unauthenticated. A missing declared
    dependency stays explicit and is not reused.
    """

    checked = _owner_limits(owners, _SNAPSHOT_CUSTODY_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    if (
        not snapshot_custody_criteria_match(criteria)
        or not _graph_roundtrip_unaccepted(dependency_evidence)
        or checked is None
        or probe != EXPECTED_SNAPSHOT_CUSTODY_PROBE
        or not cited
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": SNAPSHOT_CUSTODY_OPERATION,
        **EXPECTED_SNAPSHOT_CUSTODY_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_SHARED_MODULE_PROBE: dict[str, object] = {
    "admitted": False,
    "backend_executed": False,
    "declaration_graph_roots_retained": True,
    "declaration_interpretation_roots_retained": True,
    "declaration_source_roots_retained": True,
    "definition_module_count": 1,
    "definition_modules": [
        "LegalCorpus.Definition847d1b03df9cd4352db8236d91eecfa8d6872745349372db52a55f017a877698",
    ],
    "external_definition_resolution": False,
    "file_count": 8,
    "formalized": False,
    "graph_count": 2,
    "independently_reviewed": False,
    "interpretation_count": 2,
    "lake_build_executed": False,
    "lean_compilation_verified": False,
    "lean_file_count": 5,
    "library_roots": ["LegalCorpus"],
    "library_target": "legal",
    "norm_module_count": 2,
    "norm_modules": [
        "LegalCorpus.Norma5e2e324f23f53a6946c5bd6515b69093592ffb38655ef01cdd2e13a0664e2e9",
        "LegalCorpus.Normb8fb10dca08ce667b0323b0d22f1adc50822f1bda26afb9a97a166357148b3ec",
    ],
    "one_file_corpus": False,
    "preparation_only": True,
    "project_name": "legal_corpus_shared",
    "proof_authority": False,
    "qualified_declarations_are_module_local": True,
    "root_file_is_import_list": True,
    "shared_definition_deduplication": True,
    "source_semantics_verified": False,
    "source_version_count": 1,
    "training_qualified": False,
    "v2_build_eligible": True,
    "v2_kernel_proofs_checked": True,
    "v2_max_files": 9,
    "v2_toolchain_probed": True,
    "v2_unpinned_toolchain_refused": (
        "exact version-pinned Lean toolchain string required"
    ),
}
_SHARED_MODULE_CRITERIA = frozenset(
    {
        (
            "reuse the existing legal lake library target; every declaration "
            "maps to graph/interpretation/source roots."
        ),
        "one coherent module corpus does not require one enormous file.",
    }
)
_SHARED_MODULE_MARKERS = {
    "shared": (
        '[[lean_lib]]\\nname = "legal"\\nroots = ["LegalCorpus"]',
        '"interpretation_cid": candidate["report_record"]["record_cid"]',
        "source_occurrence_cid",
        "unsupported or oversized complete shared Lean project",
        "Share exact explicit definitions while preserving every original norm.",
    ),
    "definition_use": (
        "No process, toolchain probe, build or admission occurs here.",
        "exact distinct native module population required",
        '[[lean_lib]]\\nname = "legal"',
        '"processes_executed": 0',
        '"build_eligible": False',
    ),
}


def shared_module_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the two shared-module sentences only."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _SHARED_MODULE_CRITERIA


def _snapshot_custody_unpromoted(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when stored snapshot custody authenticated nothing and ran no Lake."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != SNAPSHOT_CUSTODY_OPERATION:
                continue
            if (
                mapping.get("candidate_assertions_promoted") is True
                or mapping.get("lake_build_executed") is True
                or mapping.get("external_referenced_records_authenticated") is True
                or mapping.get("external_references_verified") is True
                or mapping.get("legal_dependency_inventory_exhaustive") is True
                or mapping.get("proof_authority") is True
            ):
                return False
            if (
                mapping.get("external_referenced_records_authenticated") is False
                and mapping.get("candidate_assertions_promoted") is False
                and mapping.get("external_references_verified") is False
                and mapping.get("legal_dependency_inventory_exhaustive") is False
                and mapping.get("proof_authority") is False
                and mapping.get("admitted") is False
                and mapping.get("formalized") is False
                and mapping.get("bundle_roundtrip") is True
                and mapping.get("bundle_records_present") is True
                and mapping.get("missing_dependency_count") == 1
                and mapping.get("population_complete") is True
                and mapping.get("coverage_scope")
                == "exact_declared_artifacts_pins_and_edges_only"
            ):
                found = True
            else:
                return False
    return found


def shared_module_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record a shared module corpus that reuses the legal library target.

    Each declaration keeps its graph, interpretation, and source identity.
    The corpus stays several module files. v2_build_eligible,
    v2_kernel_proofs_checked, and v2_toolchain_probed are true only while
    those owner flags stay false. preparation_only stays true and
    lake_build_executed stays false. A Lake build, a promoted snapshot, or
    a changed probe leaves the task blocked. This is not kernel admission.
    """

    checked = _owner_limits(owners, _SHARED_MODULE_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not shared_module_criteria_match(criteria)
        or not _snapshot_custody_unpromoted(dependency_evidence)
        or checked is None
        or probe != EXPECTED_SHARED_MODULE_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": SHARED_MODULE_OPERATION,
        **EXPECTED_SHARED_MODULE_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_V2_OWNER_ENVELOPE_PROBE: dict[str, object] = {
    "admission_build_eligible": False,
    "admitted": False,
    "binary_sha256_recorded": False,
    "build_max_modules": 6,
    "build_profile": "bounded-v2-definition-use-conditional-build/v1",
    "command_argv_tail": ["--offline", "--no-cache", "build", "legal"],
    "command_bound_before_execution": True,
    "context_closure_verified": False,
    "expected_artifact_population_bound_before_execution": True,
    "formalized": False,
    "hypotheses_bound_before_execution": True,
    "independent_binary_pin_required_before_execution": True,
    "lake_build_executed": False,
    "library_target": "legal",
    "native_max_modules": 6,
    "native_profile": "replayed-v2-definition-use-conditional-native-build/v1",
    "owners_interchangeable": False,
    "profile_selected": False,
    "profiles_distinct": True,
    "proof_authority": False,
    "shared_schema_label": "legal-corpus-v2-definition-use-native-build/v1",
    "source_closure_bound_before_execution": True,
    "source_semantics_verified": False,
    "toolchain_library_closure_verified": False,
}
_V2_OWNER_ENVELOPE_CRITERIA = frozenset(
    {
        (
            "two v2 owners with shared historical schema labels are not "
            "interchangeable."
        ),
        (
            "exact commands, expected artifact population, hypotheses and "
            "source closure are independently bound before execution."
        ),
    }
)
_V2_OWNER_ENVELOPE_MARKERS = {
    "native": (
        'BUILD_SCHEMA = "legal-corpus-v2-definition-use-native-build/v1"',
        'PROFILE = "replayed-v2-definition-use-conditional-native-build/v1"',
        'command = (str(executable), "--offline", "--no-cache", "build", "legal")',
        'modules = manifest["expected_modules"]',
        'source_files = {name: body.encode("utf-8") for name, body in checked.files}',
        'manifest["theorem_assumptions"]',
        "carries every real hypothesis without its obsolete execution-status text.",
        '"context_closure_verified"',
        "Public import uses only the standard library.",
    ),
    "build": (
        'SCHEMA = "legal-corpus-v2-definition-use-native-build/v1"',
        'PROFILE = "bounded-v2-definition-use-conditional-build/v1"',
        'command = (str(lake), "--offline", "--no-cache", "build", "legal")',
        'len(paths) == len(manifest["expected_modules"]) == MAX_MODULES',
        '"theorem_assumptions"',
        "input_files=dict(checked.files)",
        '"toolchain_library_closure_verified": False',
        '"admission_build_eligible": False',
        "Import performs no toolchain discovery or process execution.",
    ),
}


def v2_owner_envelope_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the two V2 owner-envelope sentences only."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _V2_OWNER_ENVELOPE_CRITERIA


def _shared_module_preparation_only(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when stored shared-module evidence prepared files and ran no Lake."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != SHARED_MODULE_OPERATION:
                continue
            if (
                mapping.get("preparation_only") is not True
                or mapping.get("lake_build_executed") is not False
                or mapping.get("one_file_corpus") is not False
                or mapping.get("library_target") != "legal"
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
                or mapping.get("proof_authority") is not False
                or mapping.get("backend_executed") is not False
                or mapping.get("lean_compilation_verified") is not False
                or mapping.get("source_semantics_verified") is not False
                or mapping.get("independently_reviewed") is not False
            ):
                return False
            found = True
    return found


def v2_owner_envelope_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record two non-interchangeable V2 owners without running either build.

    The owners share one historical schema label and keep different profiles.
    Each source binds the command, artifact population, hypotheses, and
    source files before its runner call. profile_selected and
    binary_sha256_recorded stay false. lake_build_executed stays false.
    A Lake run, a selected profile, or a changed probe leaves the task blocked.
    """

    checked = _owner_limits(owners, _V2_OWNER_ENVELOPE_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not v2_owner_envelope_criteria_match(criteria)
        or not _shared_module_preparation_only(dependency_evidence)
        or checked is None
        or probe != EXPECTED_V2_OWNER_ENVELOPE_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": V2_OWNER_ENVELOPE_OPERATION,
        **EXPECTED_V2_OWNER_ENVELOPE_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_TYPED_DECODER_PROBE: dict[str, object] = {
    "binder_sort_arity_reference_checked": True,
    "invalid_frame_checked": True,
    "incomplete_frame_checked": True,
    "fresh_head_retained": True,
    "compatible_warm_start_retained": True,
    "direct_kl_across_incompatible_codecs": False,
    "cross_entropy_loss": True,
    "training_executed": False,
    "eligibility_granted": False,
    "admitted": False,
    "formalized": False,
}
_TYPED_DECODER_CRITERIA = frozenset(
    {
        (
            "binder/sort/arity/reference constraints and raw "
            "invalid/incomplete outputs are checked."
        ),
        (
            "fresh-head and compatible warm-start controls are retained; "
            "no direct kl across incompatible codecs."
        ),
    }
)
_TYPED_DECODER_MARKERS = {
    "decoder": (
        "def fresh_head_parameters(",
        "def compatible_warm_start(",
        "def refuse_direct_kl(",
        "direct KL across incompatible codecs is refused",
        "F.cross_entropy(",
    ),
    "syntax": (
        "Binders must reference sorts present in the signature.",
        "Sort, arity, and signature invariants fail here",
    ),
    "codec": (
        '"reason": "invalid_frame"',
        'reason="missing_bos_and_eos"',
    ),
}


def typed_decoder_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the typed-decoder check and warm-start sentences only."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _TYPED_DECODER_CRITERIA


def _byte_target_not_fit(dependency_evidence: Sequence[Mapping[str, object]]) -> bool:
    """True when the byte-target dependency granted no eligibility."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != BYTE_TARGET_OPERATION:
                continue
            if (
                mapping.get("vocabulary_size") != 259
                or mapping.get("eligibility_granted") is not False
                or mapping.get("training_executed") is not False
                or mapping.get("fitting_executed") is not False
            ):
                return False
            found = True
    return found


def typed_decoder_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record fresh-head, compatible warm-start, and the cross-codec KL refusal.

    Binder, sort, arity, and reference checks stay in the syntax owner.
    Invalid and incomplete frames stay in the codec. No training step runs.
    """

    checked = _owner_limits(owners, _TYPED_DECODER_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not typed_decoder_criteria_match(criteria)
        or not _byte_target_not_fit(dependency_evidence)
        or checked is None
        or probe != EXPECTED_TYPED_DECODER_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": TYPED_DECODER_OPERATION,
        **EXPECTED_TYPED_DECODER_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_SHARED_LAKE_BUILD_PROBE: dict[str, object] = {
    "admitted": False,
    "artifact_count": 5,
    "artifact_records": [
        {
            "bytes": 106320,
            "compiled_path": "compiled/LegalCorpus/Prelude.olean",
            "module": "LegalCorpus.Prelude",
            "sha256": "bda1e16ef579e08718552c6b25dbc5e525fd6f39ad8033c93eef8310b7d351c1",
        },
        {
            "bytes": 9312,
            "compiled_path": (
                "compiled/LegalCorpus/Definition847d1b03df9cd4352db8236d91eecfa8"
                "d6872745349372db52a55f017a877698.olean"
            ),
            "module": (
                "LegalCorpus.Definition847d1b03df9cd4352db8236d91eecfa8"
                "d6872745349372db52a55f017a877698"
            ),
            "sha256": "f6a2c9e45e036e18d003be1eaeaed66536318f5d4d185d6962b0c018b7fffa4a",
        },
        {
            "bytes": 15224,
            "compiled_path": (
                "compiled/LegalCorpus/Normb8fb10dca08ce667b0323b0d22f1adc50822"
                "f1bda26afb9a97a166357148b3ec.olean"
            ),
            "module": (
                "LegalCorpus.Normb8fb10dca08ce667b0323b0d22f1adc50822"
                "f1bda26afb9a97a166357148b3ec"
            ),
            "sha256": "bd75e917897c303d0742c8766abf28e783b613a8eb1ff35bdc20ca721d8b408a",
        },
        {
            "bytes": 15224,
            "compiled_path": (
                "compiled/LegalCorpus/Norma5e2e324f23f53a6946c5bd6515b69093592f"
                "fb38655ef01cdd2e13a0664e2e9.olean"
            ),
            "module": (
                "LegalCorpus.Norma5e2e324f23f53a6946c5bd6515b69093592f"
                "fb38655ef01cdd2e13a0664e2e9"
            ),
            "sha256": "31fc1712c8f515add4fc24fef82a3d0099ac48adc6db1f916ffae9b143ec3693",
        },
        {
            "bytes": 1808,
            "compiled_path": "compiled/LegalCorpus.olean",
            "module": "LegalCorpus",
            "sha256": "5b4f79fa0be3bdc4a73e86d4ca5c51c4a2c285075785227fc4c5fc8fe652d929",
        },
    ],
    "authored_fixture_kind": "authored_official_shape_fixture",
    "backend_executed": True,
    "binaries_unchanged": True,
    "binary_sha256": {
        "/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lake": (
            "f80f50cbab5322e35bcb0bf88937e22e9d908b5e4e336e83ddfc30eb1506aacf"
        ),
        "/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lean": (
            "79fb1d26fa5a39385d59fdc48a711a14b0710ca6480271acce99b4d177cea085"
        ),
    },
    "build_passed": True,
    "command": [
        "/home/barberb/.elan/toolchains/leanprover--lean4---v4.34.1/bin/lake",
        "--offline",
        "--no-cache",
        "build",
        "legal",
    ],
    "compiled_equals_expected": True,
    "context_closure_verified": False,
    "execution_error_empty": True,
    "expected_modules": [
        "LegalCorpus.Prelude",
        (
            "LegalCorpus.Definition847d1b03df9cd4352db8236d91eecfa8"
            "d6872745349372db52a55f017a877698"
        ),
        (
            "LegalCorpus.Normb8fb10dca08ce667b0323b0d22f1adc50822"
            "f1bda26afb9a97a166357148b3ec"
        ),
        (
            "LegalCorpus.Norma5e2e324f23f53a6946c5bd6515b69093592f"
            "fb38655ef01cdd2e13a0664e2e9"
        ),
        "LegalCorpus",
    ],
    "federal_corpus_acceptance": False,
    "formalized": False,
    "import_closure_scope": "prepared_shared_modules_only",
    "independently_reviewed": False,
    "lean_compilation_verified": True,
    "legal_acceptance": False,
    "manifest_coverage_passed": True,
    "native_owner_executed": False,
    "norm_ids": ["authored-shared-O", "authored-shared-P"],
    "norm_theorems_without_axioms": [
        (
            "LegalCorpus.Norma5e2e324f23f53a6946c5bd6515b69093592ffb38655ef01"
            "cdd2e13a0664e2e9.consequence_of_definition"
        ),
        (
            "LegalCorpus.Normb8fb10dca08ce667b0323b0d22f1adc50822f1bda26afb9a"
            "97a166357148b3ec.consequence_of_definition"
        ),
    ],
    "olean_magic_ok": True,
    "preparation_cid": "baguqeeraulxuvxnmjyuiufljaeydtl7smyxlhjwjx3zzuuone74y5eqfp6pa",
    "preparation_manifest_sha256": (
        "943affebec4d06f16bc370f79d248f35520eb530351ecfae39a9f71316c4414a"
    ),
    "profile": "exact-source-scoped-unary-definition-sharing/v1",
    "proof_authority": False,
    "receipt_sha256": "a6400fde1e76a90c26f2496956043d50ac43815743d470dc0595f20d03efe626",
    "returncode": 0,
    "schema": "legal-corpus-shared-definition-native-build/v1",
    "shared_owner_executed": True,
    "sorry_or_admit_present": False,
    "source_semantics_verified": False,
    "timed_out": False,
    "toolchain": "leanprover/lean4:v4.34.1",
    "training_qualified": False,
    "unavailable": False,
    "workspace_cleaned": True,
}
_SHARED_LAKE_BUILD_CRITERIA = frozenset(
    {
        (
            "actual command succeeds with every expected nonempty compiled "
            "artifact and complete import closure."
        ),
        (
            "no sorry, admit or hidden placeholders; unchanged inputs and "
            "durable resource closeout are recorded."
        ),
    }
)
_SHARED_LAKE_BUILD_MARKERS = {
    "native": (
        'BUILD_SCHEMA = "legal-corpus-v2-definition-use-native-build/v1"',
        "def build_definition_use_native_v2(",
        "Public import uses only the standard library.",
    ),
    "shared": (
        'BUILD_SCHEMA = "legal-corpus-shared-definition-native-build/v1"',
        'PROFILE = "exact-source-scoped-unary-definition-sharing/v1"',
        "def build_shared_definition_corpus(",
        'command = (str(executable), "--offline", "--no-cache", "build", "legal")',
        "fresh shared build output directory required",
        'not re.search(r"\\b(?:sorry|sorryAx)\\b", result.stdout + result.stderr)',
        "result.workspace_cleaned",
    ),
}


def shared_lake_build_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the successful-command and closeout sentences only."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _SHARED_LAKE_BUILD_CRITERIA


def _v2_envelope_not_executed(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the V2 envelope dependency recorded no Lake command."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != V2_OWNER_ENVELOPE_OPERATION:
                continue
            if (
                mapping.get("lake_build_executed") is not False
                or mapping.get("profile_selected") is not False
                or mapping.get("binary_sha256_recorded") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
                or mapping.get("proof_authority") is not False
            ):
                return False
            found = True
    return found


def shared_lake_build_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record the shared-owner fixture command without calling it legal acceptance.

    The native owner is present and did not run. lean_compilation_verified
    is the shared owner's flag for this authored O/P fixture only.
    """

    checked = _owner_limits(owners, _SHARED_LAKE_BUILD_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not shared_lake_build_criteria_match(criteria)
        or not _v2_envelope_not_executed(dependency_evidence)
        or checked is None
        or probe != EXPECTED_SHARED_LAKE_BUILD_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": SHARED_LAKE_BUILD_OPERATION,
        **EXPECTED_SHARED_LAKE_BUILD_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_CONSEQUENCE_AUDIT_PROBE: dict[str, object] = {
    "admitted": False,
    "all_hypotheses_force_modal_action": True,
    "alternative_modalities_collapsed": False,
    "audit_lake_build_executed": False,
    "compilation_rerun": False,
    "condition_exception_limitation_retained": True,
    "condition_resolution_claimed": False,
    "cross_family_equivalence_verified": False,
    "definition_body_operand_count": 2,
    "definition_body_operator": "any",
    "distinct_occurrence_definition_modules": 2,
    "dropped_hypothesis_refused": True,
    "each_hypothesis_necessary": True,
    "equal_ast_not_merged": True,
    "exception_resolution_claimed": False,
    "explicit_universal_biconditional": True,
    "failed_entailment_with_action_true": 5,
    "failed_proof_is_proof_of_negation": False,
    "federal_corpus_acceptance_claimed": False,
    "formalized": False,
    "hypotheses_present": ["Def", "Norm", "Body"],
    "inapplicable_category_forces_action": False,
    "inapplicable_category_negates_action": False,
    "invented_axiom_refused": True,
    "legal_acceptance_claimed": False,
    "negation_inferred": False,
    "negation_theorem_present": False,
    "non_deontic_refusal": (
        "canonical O/P/F rules require the deontic route; "
        "broader family equivalence is unverified"
    ),
    "operators_distinct": True,
    "permission_rewritten_as_obligation": False,
    "preparation_backend_executed": False,
    "preparation_lean_compilation_verified": False,
    "proof_authority": False,
    "receipt_sha256": (
        "a6400fde1e76a90c26f2496956043d50ac43815743d470dc0595f20d03efe626"
    ),
    "recorded_axiom_lists_empty": True,
    "recorded_lean_compilation_verified": True,
    "recorded_legal_truth_assumption": True,
    "recorded_norm_theorem_count": 2,
    "shared_modal_labels": ["deontic:O", "deontic:P"],
    "sharing_definition_modules": 1,
    "sharing_norm_modules": 2,
    "sorry_or_admit_present": False,
    "source_semantics_verified": False,
    "supported_routes": ["deontic"],
    "theorem_establishes_legal_truth": False,
    "trivial_boolean_refused": True,
    "vacuous_true_refused": True,
    "wrong_scope_definition_modules": 2,
    "wrong_scope_shared": False,
    "wrong_scope_sharing_keys": 2,
}
_CONSEQUENCE_AUDIT_CRITERIA = frozenset(
    {
        (
            "definition sharing, wrong scope/roles, conditions/exceptions "
            "and alternative theories are checked without vacuity."
        ),
        (
            "compilation is formal-artifact evidence, not source-law "
            "fidelity; a failed proof is not a proof of negation."
        ),
    }
)
_CONSEQUENCE_AUDIT_MARKERS = {
    "shared": (
        "Kernel checking does not establish law.",
        "it does not establish actual action truth or legal truth.",
        "def prepare_shared_definition_corpus(",
        "(hDef : ",
        "(hNorm : ",
        "(hBody : ",
        "exact hNorm entity ((hDef entity).mpr hBody)",
    ),
    "routes": (
        "does not make it equivalent to the legal rule.",
        "broader family equivalence is unverified",
        '"source_semantics_verified": False',
        '"supported_routes": ["deontic"]',
    ),
}


def consequence_audit_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the non-vacuity and failed-proof sentences only."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _CONSEQUENCE_AUDIT_CRITERIA


def _fixture_compilation_not_fidelity(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded fixture build stayed short of source-law fidelity."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != SHARED_LAKE_BUILD_OPERATION:
                continue
            if (
                mapping.get("legal_acceptance") is not False
                or mapping.get("federal_corpus_acceptance") is not False
                or mapping.get("source_semantics_verified") is not False
                or mapping.get("proof_authority") is not False
                or mapping.get("sorry_or_admit_present") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
                or mapping.get("lean_compilation_verified") is not True
            ):
                return False
            found = True
    return found


def consequence_audit_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record executed countermodels without treating compilation as law.

    Wrong scope and distinct definition occurrences stay unshared. Each
    hypothesis is necessary. A failed entailment is not a proof of negation.
    """

    checked = _owner_limits(owners, _CONSEQUENCE_AUDIT_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not consequence_audit_criteria_match(criteria)
        or not _fixture_compilation_not_fidelity(dependency_evidence)
        or checked is None
        or probe != EXPECTED_CONSEQUENCE_AUDIT_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": CONSEQUENCE_AUDIT_OPERATION,
        **EXPECTED_CONSEQUENCE_AUDIT_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_INVALIDATION_KEYS_PROBE: dict[str, object] = {
    "admitted": False,
    "applicability_pin_invalidates_eligibility": True,
    "cache_pin_without_artifact_bytes_refused": True,
    "competing_definition_count": 2,
    "competing_definition_invalidates_prior_user": True,
    "competing_definition_invalidates_reverse_dependent": True,
    "competing_definition_sha256": (
        "9ffbb777c2f04c0f7de97a9dbd8cd4a6777e3b334d17a973367d32c595240bba"
    ),
    "compiled_artifact_bytes_destroyed": False,
    "compiled_artifact_sha256": (
        "9b62ff534c396278f8c7f7e977add271220285b0b85c6cbcba57bad24e32fc94"
    ),
    "conditional_compiled_artifact_retained": True,
    "coverage_scope": "exact_declared_artifacts_pins_and_edges_only",
    "eligibility_seed": "legal-eligibility",
    "formalized": False,
    "fresh_proof_for_reused_modules": False,
    "incremental_compiler_invoked": False,
    "lake_build_executed": False,
    "legal_dependency_inventory_exhaustive": False,
    "legal_meaning_established": False,
    "omitted_artifact_observation_refused": True,
    "original_definition_count": 1,
    "original_definition_retained_in_competing_table": True,
    "original_definition_sha256": (
        "68e77a58bfbe72802ddbba75de650c7186ce8d8d8150c00b2526873182728a9e"
    ),
    "population_complete": True,
    "proof_authority": False,
    "reverse_stale_artifact_ids": [
        "definition-partition",
        "prior-user",
        "reverse-dependent",
    ],
    "review_pin_invalidates_eligibility": True,
    "review_reusable_artifact_ids": ["conditional-compiled"],
    "review_stale_artifact_ids": ["legal-eligibility", "eligibility-user"],
    "source_semantics_verified": False,
    "unrelated_compiled_artifact_reused": True,
}
_INVALIDATION_KEYS_CRITERIA = frozenset(
    {
        (
            "changed applicability/review can invalidate legal eligibility "
            "without destroying a reusable conditional compiled artifact."
        ),
        (
            "adding a competing definition invalidates its prior users and "
            "reverse dependencies."
        ),
    }
)
_INVALIDATION_KEYS_MARKERS = {
    "dependencies": (
        "Unaffected artifacts are reusable only under this declared custody contract,",
        "never by reason of legal meaning or semantic consistency.",
        "def invalidate_dependencies(",
        "reusable_under_declared_dependencies",
        "legal_dependency_inventory_exhaustive",
    ),
    "incremental": (
        "Only compiler artifacts are reused;",
        "review and legal authority never follow from a cache hit.",
        "Cache hits establish neither new theorem execution nor legal authority.",
        '"fresh_proof_for_reused_modules": False',
        "previous cache, independent pin and complete artifacts must be supplied together",
    ),
}


def invalidation_keys_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the eligibility-retention and competing-definition sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _INVALIDATION_KEYS_CRITERIA


def _consequence_audit_not_acceptance(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded consequence audit did not accept the compilation as law."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != CONSEQUENCE_AUDIT_OPERATION:
                continue
            if (
                mapping.get("legal_acceptance_claimed") is not False
                or mapping.get("federal_corpus_acceptance_claimed") is not False
                or mapping.get("failed_proof_is_proof_of_negation") is not False
                or mapping.get("audit_lake_build_executed") is not False
                or mapping.get("source_semantics_verified") is not False
                or mapping.get("proof_authority") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
            ):
                return False
            found = True
    return found


def invalidation_keys_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record executed invalidation without deleting compiled artifacts.

    A review pin and an applicability pin stale eligibility. A competing
    definition stales its prior user and the reverse dependent. The
    conditional compiled artifact and its bytes stay. The incremental
    compiler does not run.
    """

    checked = _owner_limits(owners, _INVALIDATION_KEYS_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not invalidation_keys_criteria_match(criteria)
        or not _consequence_audit_not_acceptance(dependency_evidence)
        or checked is None
        or probe != EXPECTED_INVALIDATION_KEYS_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": INVALIDATION_KEYS_OPERATION,
        **EXPECTED_INVALIDATION_KEYS_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE: dict[str, object] = {
    "admitted": False,
    "ambiguous_compiled_count": 5,
    "ambiguous_definition_module_count": 2,
    "ambiguous_reused_count": 1,
    "axiom_lists_empty": True,
    "changed_compiled_count": 3,
    "changed_definition_module_count": 2,
    "changed_reused_count": 3,
    "composition_obligations_checked": True,
    "direct_lean_o_substitutes_for_lake_build_legal": False,
    "expected_module_population_agrees": True,
    "federal_corpus_acceptance_claimed": False,
    "formalized": False,
    "fresh_proof_for_reused_modules": False,
    "incremental_argv_recipe": [
        "installed-sibling-lean",
        "-o",
        "compiled/<module>.olean",
        "<module>.lean",
    ],
    "incremental_compiler_invoked": True,
    "lake_build_executed": True,
    "lake_command_tail": ["--offline", "--no-cache", "build", "legal"],
    "legal_acceptance_claimed": False,
    "legal_meaning_established": False,
    "olean_bytes_identical": False,
    "old_unique_definition_reused": False,
    "proof_authority": False,
    "recorded_legal_truth_assumption": True,
    "semantic_obligations_agree": True,
    "source_semantics_verified": False,
    "theorem_assumptions": [
        "The exact shared definition holds at the evaluation time.",
        "The universally guarded norm holds at that same time under the same interpretation.",
        "The declared definition body holds for the supplied entity at that time.",
        "The conclusion is the parameterized modal action; it does not establish actual action truth or legal truth.",
    ],
    "unchanged_cold_compile_count": 5,
    "unchanged_definition_module_count": 1,
    "unchanged_warm_compile_count": 0,
    "unique_definition_module_count": 1,
}
_INCREMENTAL_LAKE_AGREEMENT_CRITERIA = frozenset(
    {
        (
            "all expected artifacts and semantic obligations agree with "
            "the clean lake build."
        ),
        (
            "the existing direct lean -o cache experiment does not "
            "substitute for lake build legal."
        ),
    }
)
_INCREMENTAL_LAKE_AGREEMENT_MARKERS = {
    "incremental": (
        "Only compiler artifacts are reused;",
        "review and legal authority never follow from a cache hit.",
        '["installed-sibling-lean", "-o", "compiled/<module>.olean", "<module>.lean"]',
        '"fresh_proof_for_reused_modules": False',
    ),
    "shared": (
        "Kernel checking does not establish law.",
        'command = (str(executable), "--offline", "--no-cache", "build", "legal")',
        "composition_obligations_checked",
        "it does not establish actual action truth or legal truth.",
    ),
}


def incremental_lake_agreement_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the clean-Lake agreement and lean -o non-substitution sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _INCREMENTAL_LAKE_AGREEMENT_CRITERIA


def _invalidation_not_meaning(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded invalidation did not establish legal meaning."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != INVALIDATION_KEYS_OPERATION:
                continue
            if (
                mapping.get("legal_meaning_established") is not False
                or mapping.get("incremental_compiler_invoked") is not False
                or mapping.get("lake_build_executed") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
                or mapping.get("proof_authority") is not False
                or mapping.get("source_semantics_verified") is not False
            ):
                return False
            found = True
    return found


def incremental_lake_agreement_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record incremental and clean Lake agreement without treating lean -o as Lake.

    Unchanged, changed, and newly ambiguous definition partitions keep the
    same expected modules, assumptions, and empty axiom lists as the clean
    ``lake build legal`` result. The direct ``lean -o`` cache bytes stay
    distinct from those Lake artifacts.
    """

    checked = _owner_limits(owners, _INCREMENTAL_LAKE_AGREEMENT_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not incremental_lake_agreement_criteria_match(criteria)
        or not _invalidation_not_meaning(dependency_evidence)
        or checked is None
        or probe != EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": INCREMENTAL_LAKE_AGREEMENT_OPERATION,
        **EXPECTED_INCREMENTAL_LAKE_AGREEMENT_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_RELEASE_COORDINATOR_PROBE: dict[str, object] = {
    "admitted": False,
    "artifact_count": 4,
    "candidate_selector_count": 1,
    "federal_corpus_acceptance_claimed": False,
    "formalized": False,
    "hub_readback_executed": False,
    "legal_acceptance_claimed": False,
    "legal_meaning_established": False,
    "model_manager_catalog_mutated": False,
    "model_manager_selectors_disjoint": True,
    "promoted_selector_count": 1,
    "proof_authority": False,
    "receipt_lists_itself": False,
    "referenced_artifacts_published_before_receipt": True,
    "remote_upload_executed": False,
    "replay_qualified_selector_count": 1,
    "reviewed_selector_count": 1,
    "selector_collapses_statuses": False,
    "self_referential_commit_hash_refused": True,
    "source_semantics_verified": False,
    "statuses": ["candidate", "reviewed", "replay-qualified", "promoted"],
    "unpublished_artifact_refused": True,
}
_RELEASE_COORDINATOR_CRITERIA = frozenset(
    {
        (
            "referenced artifacts publish before the coordinator receipt; "
            "no self-referential commit hash."
        ),
        (
            "candidate, reviewed, replay-qualified and promoted statuses "
            "remain separate, including modelmanager selectors."
        ),
    }
)
_RELEASE_COORDINATOR_MARKERS = {
    "plan": (
        "not the current task queue or fresh qualification evidence.",
        "Referenced artifacts publish before the coordinator receipt; no self-referential commit hash.",
        "Candidate, reviewed, replay-qualified and promoted statuses remain separate, including ModelManager selectors.",
        "Remote upload and Hub readback do not run.",
    ),
    "reconciliation": (
        "without constructing ModelManager",
        "None of these inventory observations grants teacher,",
        "Referenced artifacts publish before the coordinator receipt; no self-referential commit hash.",
        "Candidate, reviewed, replay-qualified and promoted statuses remain separate, including ModelManager selectors.",
    ),
}


def release_coordinator_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the publication-order and separate-status sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _RELEASE_COORDINATOR_CRITERIA


def _lake_agreement_not_acceptance(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded Lake comparison did not establish legal meaning."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != INCREMENTAL_LAKE_AGREEMENT_OPERATION:
                continue
            if (
                mapping.get("legal_meaning_established") is not False
                or mapping.get("legal_acceptance_claimed") is not False
                or mapping.get("federal_corpus_acceptance_claimed") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
                or mapping.get("proof_authority") is not False
                or mapping.get("source_semantics_verified") is not False
            ):
                return False
            found = True
    return found


def release_coordinator_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record a local coordinator seal without a remote upload or legal acceptance.

    Referenced artifact bytes are published before the receipt. The receipt's
    commit hash is not stored inside the hashed body. The four status
    selectors stay disjoint. The October 16 inventory did not construct
    ModelManager, and this seal does not mutate one.
    """

    checked = _owner_limits(owners, _RELEASE_COORDINATOR_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not release_coordinator_criteria_match(criteria)
        or not _lake_agreement_not_acceptance(dependency_evidence)
        or checked is None
        or probe != EXPECTED_RELEASE_COORDINATOR_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": RELEASE_COORDINATOR_OPERATION,
        **EXPECTED_RELEASE_COORDINATOR_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE: dict[str, object] = {
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
_MODALITY_CAUSAL_COMPARISON_CRITERIA = frozenset(
    {
        (
            "same initialization, source cases, trainable mask, optimizer, "
            "candidates and budget within each causal comparison."
        ),
        (
            "preserve named permission and complete-rule loss-id vetoes; "
            "e1 engineering ranking work is separately tracked."
        ),
    }
)
_MODALITY_CAUSAL_COMPARISON_MARKERS = {
    "modality": (
        "automatic_selection_or_promotion=False",
        "initialization='copy_all42_trained_F15_states_fresh_ordered6_Adam/v1'",
        "actor_hinge_omitted=True",
    ),
    "plan": (
        "Same initialization, source cases, trainable mask, optimizer, candidates and budget within each causal comparison.",
        "Preserve named permission and complete-rule loss-ID vetoes; E1 engineering ranking work is separately tracked.",
        "does not select a hypothesis.",
    ),
}


def modality_causal_comparison_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the fixed-factor and preserved-veto sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _MODALITY_CAUSAL_COMPARISON_CRITERIA


def _codec_not_extended(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded codec stayed a 259-symbol export with no extension."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != CODEC_INTACT_OPERATION:
                continue
            if (
                mapping.get("extensions_added") != 0
                or mapping.get("meanings_requiring_extension") != 0
                or mapping.get("meanings_approved") is not False
                or mapping.get("vocabulary_size") != 259
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
            ):
                return False
            found = True
    return found


def modality_causal_comparison_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record identical comparison factors and preserved vetoes.

    The modality owner still refuses automatic selection. The plan records
    the permission veto and the complete-rule loss-ID veto. E1 ranking stays
    separate. No hypothesis is selected and no training runs.
    """

    checked = _owner_limits(owners, _MODALITY_CAUSAL_COMPARISON_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not modality_causal_comparison_criteria_match(criteria)
        or not _codec_not_extended(dependency_evidence)
        or checked is None
        or probe != EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": MODALITY_CAUSAL_COMPARISON_OPERATION,
        **EXPECTED_MODALITY_CAUSAL_COMPARISON_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_MODALITY_ERROR_REPORT_PROBE: dict[str, object] = {
    "acceptance_failed_retained": True,
    "admitted": False,
    "aggregate_gain": 4,
    "authored_fixture_only": True,
    "automatic_promotion_refused": True,
    "critical_modality_reversal_count": 1,
    "default_promoted": False,
    "evaluation_reference_in_producer_refused": True,
    "federal_corpus_scored": False,
    "formalized": False,
    "legal_acceptance_claimed": False,
    "legal_meaning_established": False,
    "merged_error_kind_refused": True,
    "model_executed": False,
    "owner_error_count": 1,
    "presence_error_count": 1,
    "presence_owner_reversal_separated": True,
    "producer_evaluation_reference_access": False,
    "producer_written_before_evaluation": True,
    "proof_authority": False,
    "separate_packet_owners": True,
    "source_semantics_verified": False,
    "teacher_promoted": False,
    "training_executed": False,
}
_MODALITY_ERROR_REPORT_CRITERIA = frozenset(
    {
        (
            "presence versus owner errors and critical modality reversals "
            "are separately reported."
        ),
        (
            "failed acceptance remains failed despite aggregate gains; "
            "no automatic default/teacher promotion."
        ),
        (
            "materialize separate producer and post-freeze evaluation "
            "packets/owners; primary training/inference authority grants "
            "no evaluation reference access."
        ),
    }
)
_MODALITY_ERROR_REPORT_MARKERS = {
    "modality": (
        "automatic_selection_or_promotion=False",
        "evaluation_references_used=False",
        "actor_hinge_omitted=True",
    ),
    "evaluation": (
        "neither is reviewed accuracy or training qualification.",
        "No models, training, native Lake/backend execution, legal review, admission or publication occurs",
        '"accepted", "admitted",',
    ),
}


def modality_error_report_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the separated-error, failed-acceptance, and packet sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 3 and frozenset(texts) == _MODALITY_ERROR_REPORT_CRITERIA


def _comparison_not_selection(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded comparison selected no hypothesis and ran no training."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != MODALITY_CAUSAL_COMPARISON_OPERATION:
                continue
            if (
                mapping.get("hypothesis_selected") is not False
                or mapping.get("training_executed") is not False
                or mapping.get("e1_ranking_selects_hypothesis") is not False
                or mapping.get("legal_meaning_established") is not False
                or mapping.get("legal_acceptance_claimed") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
            ):
                return False
            found = True
    return found


def modality_error_report_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record separated errors and a producer packet with no evaluation reference.

    A positive aggregate gain does not clear a failed acceptance. Default and
    teacher promotion stay refused. The producer packet is written before the
    post-freeze evaluation packet. No model runs and no training runs.
    """

    checked = _owner_limits(owners, _MODALITY_ERROR_REPORT_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not modality_error_report_criteria_match(criteria)
        or not _comparison_not_selection(dependency_evidence)
        or checked is None
        or probe != EXPECTED_MODALITY_ERROR_REPORT_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": MODALITY_ERROR_REPORT_OPERATION,
        **EXPECTED_MODALITY_ERROR_REPORT_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_TRAINING_METRIC_PROBE: dict[str, object] = {
    "admitted": False,
    "argument_attachment_modality_count": 1,
    "authored_fixture_only": True,
    "encoder_unfrozen": False,
    "engineering_scores_are_statutory_fidelity": False,
    "engineering_scores_separate": True,
    "evaluation_reference_in_producer_refused": True,
    "federal_corpus_scored": False,
    "formalized": False,
    "full_rules_count": 1,
    "legal_acceptance_claimed": False,
    "legal_meaning_established": False,
    "metric_groups_separated": True,
    "model_executed": False,
    "ordinary_generation_count": 1,
    "outputs_frozen_before_scoring": True,
    "producer_evaluation_reference_access": False,
    "producer_written_before_evaluation": True,
    "proof_authority": False,
    "protected_retention_count": 1,
    "score_before_freeze_refused": True,
    "separate_packet_owners": True,
    "source_semantics_verified": False,
    "statutory_fidelity_refused": True,
    "statutory_fidelity_reviewed": False,
    "training_executed": False,
    "unknown_metric_group_refused": True,
    "worker_launched": False,
}
_TRAINING_METRIC_CRITERIA = frozenset(
    {
        (
            "ordinary generation, full rules, argument/attachment/modality "
            "and protected retention metrics are reported."
        ),
        (
            "teacher-forced ce and compiler weak-label engineering scores "
            "remain separate from reviewed statutory fidelity; outputs "
            "freeze before scoring."
        ),
        (
            "materialize separate producer and post-freeze evaluation "
            "packets/owners; primary training/inference authority grants "
            "no evaluation reference access."
        ),
    }
)
_TRAINING_METRIC_MARKERS = {
    "decoder": (
        "ordinary generation, full rules, argument/attachment/modality and protected retention",
        "teacher-forced CE and compiler weak-label scores stay separate from reviewed statutory fidelity",
        "outputs freeze before scoring",
        "producer packet grants no evaluation reference",
    ),
    "example": (
        "ordinary generation, full rules, argument/attachment/modality and protected retention",
        "teacher-forced CE and compiler weak-label scores stay separate from reviewed statutory fidelity",
        "outputs freeze before scoring",
        "producer packet grants no evaluation reference",
    ),
}


def training_metric_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the metric, score-separation, and packet sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 3 and frozenset(texts) == _TRAINING_METRIC_CRITERIA


def _typed_decoder_not_trained(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded typed decoder granted no eligibility and ran no training."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != TYPED_DECODER_OPERATION:
                continue
            if (
                mapping.get("training_executed") is not False
                or mapping.get("eligibility_granted") is not False
                or mapping.get("direct_kl_across_incompatible_codecs") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
            ):
                return False
            found = True
    return found


def training_metric_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record separated engineering metrics and a producer packet with no reference.

    Teacher-forced and compiler weak-label scores stay outside reviewed
    statutory fidelity. Outputs freeze before scoring. No training runs and
    the encoder stays frozen.
    """

    checked = _owner_limits(owners, _TRAINING_METRIC_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not training_metric_criteria_match(criteria)
        or not _typed_decoder_not_trained(dependency_evidence)
        or checked is None
        or probe != EXPECTED_TRAINING_METRIC_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": TRAINING_METRIC_OPERATION,
        **EXPECTED_TRAINING_METRIC_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_WIDTH_CONTROL_PROBE: dict[str, object] = {
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
_WIDTH_CONTROL_CRITERIA = frozenset(
    {
        (
            "historical 8d variants, active native vectors and detached "
            "disabled-latent components remain distinct."
        ),
        (
            "correct, zero, branch-disabled and permuted inputs plus "
            "source-only/retrieval controls are explicit; mean-coordinate "
            "mse is not semantic superiority."
        ),
    }
)
_WIDTH_CONTROL_MARKERS = {
    "reconciliation": (
        "Historical 8D variants, active native vectors and detached disabled-latent components remain distinct.",
        "Correct, zero, branch-disabled and permuted inputs plus source-only and retrieval controls stay named and unexecuted.",
        "Mean-coordinate MSE is not semantic superiority.",
        "No control was executed and no mean-coordinate MSE was measured.",
    ),
    "native_inputs": (
        "Historical 8D variants, active native vectors and detached disabled-latent components remain distinct.",
        "Correct, zero, branch-disabled and permuted inputs plus source-only and retrieval controls stay named and unexecuted.",
        "Mean-coordinate MSE is not semantic superiority.",
        "No control was executed and no mean-coordinate MSE was measured.",
    ),
}


def width_control_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the distinct-width and explicit-control sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _WIDTH_CONTROL_CRITERIA


def _training_metric_not_executed(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded training-metric report ran no training."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != TRAINING_METRIC_OPERATION:
                continue
            if (
                mapping.get("training_executed") is not False
                or mapping.get("model_executed") is not False
                or mapping.get("encoder_unfrozen") is not False
                or mapping.get("federal_corpus_scored") is not False
                or mapping.get("legal_meaning_established") is not False
                or mapping.get("legal_acceptance_claimed") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
            ):
                return False
            found = True
    return found


def width_control_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record distinct widths and named unexecuted controls.

    Mean-coordinate MSE stays outside semantic superiority. No control,
    producer call, encoder call, or training run is recorded.
    """

    checked = _owner_limits(owners, _WIDTH_CONTROL_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not width_control_criteria_match(criteria)
        or not _training_metric_not_executed(dependency_evidence)
        or checked is None
        or probe != EXPECTED_WIDTH_CONTROL_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": WIDTH_CONTROL_OPERATION,
        **EXPECTED_WIDTH_CONTROL_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_CONTEXT_BUDGET_PROBE: dict[str, object] = {
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
_CONTEXT_BUDGET_CRITERIA = frozenset(
    {
        (
            "same decoder and source budget for "
            "context/no-context/shuffled/unrelated-context and "
            "retrieval-only controls."
        ),
        (
            "definition dependencies and cross-split context allowances "
            "are audited."
        ),
    }
)
_CONTEXT_BUDGET_MARKERS = {
    "lookup": (
        "Context, no-context, shuffled, unrelated-context and retrieval-only controls share one decoder and one source budget.",
        "Definition dependencies stay separate from context pins.",
        "Cross-split context allowance is not granted.",
        "No decoder was executed.",
    ),
    "example": (
        "Context, no-context, shuffled, unrelated-context and retrieval-only controls share one decoder and one source budget.",
        "Definition dependencies stay separate from context pins.",
        "Cross-split context allowance is not granted.",
        "No decoder was executed.",
    ),
}


def context_budget_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the shared-budget and audited-allowance sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _CONTEXT_BUDGET_CRITERIA


def _declared_closure_not_exhaustive(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded closure stayed a non-exhaustive frontier."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != DECLARED_CLOSURE_OPERATION:
                continue
            if (
                mapping.get("legal_dependency_inventory_exhaustive") is not False
                or mapping.get("interpretations_unioned") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
            ):
                return False
            found = True
    return found


def context_budget_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record one unexecuted decoder budget and separated definition pins.

    The allowance count stays 0. No decoder, training run, or federal
    corpus score is recorded.
    """

    checked = _owner_limits(owners, _CONTEXT_BUDGET_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not context_budget_criteria_match(criteria)
        or not _training_metric_not_executed(dependency_evidence)
        or not _declared_closure_not_exhaustive(dependency_evidence)
        or checked is None
        or probe != EXPECTED_CONTEXT_BUDGET_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": CONTEXT_BUDGET_OPERATION,
        **EXPECTED_CONTEXT_BUDGET_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_OCCURRENCE_VISIBILITY_PROBE: dict[str, object] = {
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
_OCCURRENCE_VISIBILITY_CRITERIA = frozenset(
    {
        (
            "every selected occurrence and unresolved/unsupported outcome "
            "remains visible."
        ),
        (
            "retrieved locator evidence is not automatically a "
            "governing-definition edge."
        ),
        (
            "materialize separate producer and post-freeze evaluation "
            "packets/owners; primary training/inference authority grants "
            "no evaluation reference access."
        ),
    }
)
_OCCURRENCE_VISIBILITY_MARKERS = {
    "evaluation": (
        "Account for every expected occurrence and submitted declared observation.",
        "every observation must be retained exactly once",
        "Unsupported is an explicit caller abstention category; rejected typed outputs do not establish a syntax-versus-family-capability taxonomy.",
    ),
    "lookup": (
        "It does not discover governing scopes or legal\nprecedence.",
        "Retrieved locator evidence is not automatically a governing-definition edge.",
        "A retrieved source occurrence stays on the lookup record.",
    ),
}


def occurrence_visibility_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the visibility, locator, and separated-packet sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 3 and frozenset(texts) == _OCCURRENCE_VISIBILITY_CRITERIA


def _context_budget_decoder_unexecuted(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded context budget executed no decoder."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != CONTEXT_BUDGET_OPERATION:
                continue
            if (
                mapping.get("decoder_executed") is not False
                or mapping.get("training_executed") is not False
                or mapping.get("model_executed") is not False
                or mapping.get("worker_launched") is not False
                or mapping.get("federal_corpus_scored") is not False
                or mapping.get("authored_fixture_only") is not True
                or mapping.get("cross_split_context_allowance_count") != 0
                or mapping.get("cross_split_context_allowance_granted") is not False
                or mapping.get("legal_meaning_established") is not False
                or mapping.get("legal_acceptance_claimed") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
            ):
                return False
            found = True
    return found


def occurrence_visibility_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record visible occurrences and a locator that is not a governing edge.

    The producer packet stays separate and receives no evaluation
    reference. No decoder, training run, or federal corpus score is
    recorded.
    """

    checked = _owner_limits(owners, _OCCURRENCE_VISIBILITY_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not occurrence_visibility_criteria_match(criteria)
        or not _context_budget_decoder_unexecuted(dependency_evidence)
        or checked is None
        or probe != EXPECTED_OCCURRENCE_VISIBILITY_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": OCCURRENCE_VISIBILITY_OPERATION,
        **EXPECTED_OCCURRENCE_VISIBILITY_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_LABEL_MASK_PROBE: dict[str, object] = {
    "Lean_admitted": False,
    "accepted_parser_refused": True,
    "admitted": False,
    "authored_fixture_only": True,
    "compiler_weak_mask": 2,
    "compiler_weak_origin": "declared-compiler-weak-tier",
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
    "resource_failure_statuses": [
        "budget_exhausted",
        "cancellation",
        "error",
        "timeout",
    ],
    "resource_label_refused": True,
    "reviewed_mask": 1,
    "reviewed_origin": "declared-reviewed-tier",
    "reviewed_tier_review_completed": False,
    "schema": "legal-label-mask-audit/v1",
    "shared_mask_refused": True,
    "solver_executed": False,
    "synthetic_mask": 4,
    "synthetic_origin": "declared-synthetic-tier",
    "tier_count": 4,
    "training_executed": False,
    "training_mask": 0,
    "unresolved_mask": 8,
    "unresolved_origin": "declared-unresolved-tier",
    "worker_launched": False,
}
_LABEL_MASK_CRITERIA = frozenset(
    {
        "solver/resource failure is not a negative legal label.",
        (
            "reviewed, compiler weak, synthetic and unresolved tiers have "
            "distinct masks and origins; parser/lean success creates no gold."
        ),
    }
)
_LABEL_MASK_MARKERS = {
    "evaluation": (
        "Available references are replayed authored unreviewed declarations, not reviewed statutory gold; unavailable references remain in the denominator.",
        "No models, training, native Lake/backend execution, legal review, admission or publication occurs; existing owners may regenerate syntax-only Lean text.",
        "Incomplete, invalid frames, invalid content and exhausted local token budgets remain distinct; invalid UTF8 is not representable by this fixed lexical vocabulary and is not inferred from missing output.",
        "Explicit error, timeout, cancellation and not_started are caller declarations. Actual context, owner, source or full-bound refusal and interruption abort without a partial report.",
    ),
    "example": (
        "Solver or resource failure is not a negative legal label.",
        "Reviewed, compiler-weak, synthetic and unresolved tiers keep distinct masks and origins.",
        "Parser or Lean success creates no gold.",
    ),
}


def label_mask_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the resource-label and distinct-tier sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _LABEL_MASK_CRITERIA


def label_mask_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record distinct tier masks and a parse that creates no gold.

    Resource failure stays off the negative legal label. No solver,
    Lean build, training run, or federal corpus score is recorded.
    """

    checked = _owner_limits(owners, _LABEL_MASK_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not label_mask_criteria_match(criteria)
        or not _training_metric_not_executed(dependency_evidence)
        or checked is None
        or probe != EXPECTED_LABEL_MASK_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": LABEL_MASK_OPERATION,
        **EXPECTED_LABEL_MASK_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_TEACHER_IDENTITY_PROBE: dict[str, object] = {
    "admitted": False,
    "authored_fixture_only": True,
    "changed_experiment_identity": (
        "32c67562274eac525e9e2aca60a491939950f69cac09062fab50699bdaa458a2"
    ),
    "changed_split_identity": (
        "a23c1f19f9e32e8deba7b2813e20344ee2f0b450490b5c3a3ea2ec0ad5b62e41"
    ),
    "changed_target_sha256": (
        "f38f234faa13f51aab67df14a16174f9a17bd1afe3ee771ae4c87947a8401c8c"
    ),
    "changed_teacher_identity": (
        "97cbd6787688cce095f248b352ca8560eafa961c2f6c5508fb8e83c9ced131d4"
    ),
    "decoder_executed": False,
    "engineering_retention": "engineering-retention",
    "engineering_retention_separately_labeled": True,
    "experiment_identity": (
        "ce6b71a1e6853752490a70d50430f429693c67712493baa422e14a99bc2b9380"
    ),
    "experiment_identity_changed": True,
    "federal_corpus_scored": False,
    "formalized": False,
    "independently_qualified_teacher_supplied": False,
    "legal_acceptance_claimed": False,
    "legal_meaning_established": False,
    "model_executed": False,
    "preserved_identity_refused": True,
    "qualified_teacher_refused": True,
    "retention_identity_unchanged": True,
    "retention_scope_refused": True,
    "schema": "legal-teacher-identity-audit/v1",
    "semantic_distillation_scope_supplied": False,
    "split_identity": (
        "66baadb315d2e2765021d48e17fd4b7814997c782caf43450d228eab5fd15f09"
    ),
    "split_identity_changed": True,
    "target_count": 2,
    "target_sha256": (
        "3406df3f26e4d0c4bb83fc9a0d2722843284ad091203c3ca5c349ac61d7e4a81"
    ),
    "teacher_identity": (
        "be11b890de7a879ef71dd68998d1bb9467cbf0f21152f34a5fa021c914d27569"
    ),
    "teacher_identity_changed": True,
    "teacher_promoted": False,
    "training_executed": False,
    "unchanged_target_identities_stable": True,
    "worker_launched": False,
}
_TEACHER_IDENTITY_CRITERIA = frozenset(
    {
        "changed targets invalidate split/experiment/teacher identities as required.",
        (
            "only an independently qualified teacher supplies semantic "
            "distillation scope; engineering retention is separately labeled."
        ),
    }
)
_TEACHER_IDENTITY_MARKERS = {
    "queue": (
        "Changed targets invalidate split, experiment and teacher identities.",
        "Only an independently qualified teacher supplies semantic distillation scope.",
        "Engineering retention stays a separate label.",
        "No independently qualified teacher was supplied.",
    ),
    "example": (
        "Changed targets invalidate split, experiment and teacher identities.",
        "Only an independently qualified teacher supplies semantic distillation scope.",
        "Engineering retention stays a separate label.",
        "No independently qualified teacher was supplied.",
    ),
}


def teacher_identity_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the identity-invalidation and teacher-scope sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _TEACHER_IDENTITY_CRITERIA


def _label_mask_created_no_gold(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded label-mask audit created no gold."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != LABEL_MASK_OPERATION:
                continue
            if (
                mapping.get("training_executed") is not False
                or mapping.get("model_executed") is not False
                or mapping.get("solver_executed") is not False
                or mapping.get("lean_executed") is not False
                or mapping.get("resource_failure_executed") is not False
                or mapping.get("decoder_executed") is not False
                or mapping.get("worker_launched") is not False
                or mapping.get("federal_corpus_scored") is not False
                or mapping.get("legal_meaning_established") is not False
                or mapping.get("legal_acceptance_claimed") is not False
                or mapping.get("admitted") is not False
                or mapping.get("formalized") is not False
                or mapping.get("gold_created_by_parser_or_lean_success") is not False
                or mapping.get("reviewed_tier_review_completed") is not False
                or mapping.get("authored_fixture_only") is not True
                or mapping.get("training_mask") != 0
                or mapping.get("evaluation_gold_mask") != 0
                or mapping.get("evaluation_gold_qualified") is not False
            ):
                return False
            found = True
    return found


def teacher_identity_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record invalidated target identities and an unsupplied teacher.

    Semantic distillation scope stays off. Engineering retention stays a
    separate label. No teacher is qualified and no training run is recorded.
    """

    checked = _owner_limits(owners, _TEACHER_IDENTITY_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not teacher_identity_criteria_match(criteria)
        or not _label_mask_created_no_gold(dependency_evidence)
        or checked is None
        or probe != EXPECTED_TEACHER_IDENTITY_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": TEACHER_IDENTITY_OPERATION,
        **EXPECTED_TEACHER_IDENTITY_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_POPULATION_COVERAGE_PROBE: dict[str, object] = {
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
    "schema": "legal-population-coverage-audit/v1",
    "throughput_extrapolated": False,
    "training_executed": False,
    "unsupported_count": 2,
    "worker_launched": False,
}
_POPULATION_COVERAGE_CRITERIA = frozenset(
    {
        (
            "counts distinguish inventory, eligible, attempted, "
            "missing/invalid/unsupported and accepted populations."
        ),
        (
            "no throughput extrapolation from the small replay or "
            "row-only total; review cost is included."
        ),
    }
)
_POPULATION_COVERAGE_MARKERS = {
    "evaluation": (
        "The independently pinned selected population is the denominator; selection, partition and source/context hashes remain unauthenticated caller declarations.",
        "Unsupported is an explicit caller abstention category; rejected typed outputs do not establish a syntax-versus-family-capability taxonomy.",
        "Full selected and comparable-pair rates are separate descriptive structural diagnostics; neither is reviewed accuracy or training qualification.",
    ),
    "plan": (
        "Inventory, eligible, attempted, missing, invalid, unsupported and accepted counts stay separate.",
        "Compared is not an accepted population.",
        "Attempted rows include a review record.",
        "No throughput extrapolation from the small replay or the row-only total.",
    ),
}
_NATIVE_HANDOFF_SCHEMA = (
    "legal-ir-native9004-independent-factual-handoff-evidence/v1"
)


def population_coverage_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the separate-count and non-extrapolation sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _POPULATION_COVERAGE_CRITERIA


def _coverage_dependencies_recorded(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the consequence audit and the native handoff grant no authority."""

    if _any_flag_true(dependency_evidence):
        return False
    consequence = False
    handoff = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("legal_acceptance_claimed") is True:
                return False
            if mapping.get("federal_corpus_scored") is True:
                return False
            if mapping.get("operation") == CONSEQUENCE_AUDIT_OPERATION:
                consequence = True
            if mapping.get("schema") == _NATIVE_HANDOFF_SCHEMA:
                handoff = True
    return consequence and handoff


def population_coverage_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record separate fixture counts and an included review record.

    Compared is not accepted. The small replay is not scaled. Accepted stays
    zero. No federal corpus is scored.
    """

    checked = _owner_limits(owners, _POPULATION_COVERAGE_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not population_coverage_criteria_match(criteria)
        or not _coverage_dependencies_recorded(dependency_evidence)
        or checked is None
        or probe != EXPECTED_POPULATION_COVERAGE_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": POPULATION_COVERAGE_OPERATION,
        **EXPECTED_POPULATION_COVERAGE_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_STORAGE_CONCURRENCY_PROBE: dict[str, object] = {
    "admitted": False,
    "authored_fixture_only": True,
    "budget_expanded_lineage": "budget-expanded-lineage",
    "cap_bypassed": False,
    "cap_bypass_refused": True,
    "decoder_executed": False,
    "external_file_cap": 134217728,
    "federal_corpus_scored": False,
    "files_deleted": False,
    "foreign_asset_disposition": "preserved",
    "foreign_assets_preserved": True,
    "foreign_removal_refused": True,
    "formalized": False,
    "grant_refused": True,
    "ledger_bypassed": False,
    "ledger_cap": 16777216,
    "ledger_changed": False,
    "legal_acceptance_claimed": False,
    "legal_meaning_established": False,
    "model_executed": False,
    "native_grant_purpose": "observation-only",
    "native_grant_purpose_specific": True,
    "overlap_refused": True,
    "publication_overlap": "publication-overlap",
    "publication_overlap_replaced_retained": False,
    "resource_admission": False,
    "retained_evidence": "retained-evidence",
    "role_count": 4,
    "roles_distinct": True,
    "schema": "legal-storage-concurrency-audit/v1",
    "temporary_replay": "temporary-replay",
    "training_executed": False,
    "worker_launched": False,
}
_STORAGE_CONCURRENCY_CRITERIA = frozenset(
    {
        (
            "budget expanded lineage, temporary replay, retained evidence "
            "and publication overlap."
        ),
        (
            "native grants remain purpose-specific; foreign claims/assets "
            "are preserved and ledger/cap values are not bypassed."
        ),
    }
)
_STORAGE_CONCURRENCY_MARKERS = {
    "runtime": (
        "Budget-expanded lineage, temporary replay, retained evidence and publication overlap stay separate.",
        "A native grant stays purpose-specific to observation.",
        "Foreign claims and assets stay preserved.",
        "Ledger and cap values are not bypassed.",
    ),
    "feed": (
        "Budget-expanded lineage, temporary replay, retained evidence and publication overlap stay separate.",
        "A native grant stays purpose-specific to observation.",
        "Foreign claims and assets stay preserved.",
        "Ledger and cap values are not bypassed.",
    ),
}


def storage_concurrency_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the separate-role and unchanged-cap sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _STORAGE_CONCURRENCY_CRITERIA


def _population_coverage_grants_no_authority(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded population-coverage audit grants no authority."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != POPULATION_COVERAGE_OPERATION:
                continue
            if (
                mapping.get("authored_fixture_only") is not True
                or mapping.get("accepted_count") != 0
                or mapping.get("compared_counted_as_accepted") is not False
                or mapping.get("throughput_extrapolated") is not False
                or mapping.get("pilot_review_cost_measured") is not False
                or mapping.get("federal_corpus_scored") is not False
                or mapping.get("legal_acceptance_claimed") is not False
                or mapping.get("review_cost_included") is not True
            ):
                return False
            found = True
    return found


def storage_concurrency_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record four storage roles and the unchanged ledger and file caps.

    The native grant stays observational. Foreign assets stay preserved.
    Neither cap moves. Accepted stays false. No ledger write is recorded.
    """

    checked = _owner_limits(owners, _STORAGE_CONCURRENCY_MARKERS)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not storage_concurrency_criteria_match(criteria)
        or not _population_coverage_grants_no_authority(dependency_evidence)
        or checked is None
        or probe != EXPECTED_STORAGE_CONCURRENCY_PROBE
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": STORAGE_CONCURRENCY_OPERATION,
        **EXPECTED_STORAGE_CONCURRENCY_PROBE,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


EXPECTED_SPARSE8_PROBE: dict[str, object] = json.loads(
    r"""
{
  "admitted": false,
  "authored_fixture_only": true,
  "checkpoint_sha256": "7236de26bd3d7f8414ffa04805f1b6e8a8849f9e0103cec6edb4985b911658be",
  "completion_authority": false,
  "constitution_roundtrip_ok": false,
  "daemon_profile_executed": false,
  "decoder_executed": false,
  "dimension": 8,
  "encode_keys": [
    "embedding_projection",
    "family_distribution",
    "sample_id",
    "selected_frame",
    "target_family",
    "target_family_distribution"
  ],
  "encoder_called": true,
  "encoder_unfrozen": false,
  "exact_published_legacy_teacher": true,
  "federal_corpus_scored": false,
  "formalized": false,
  "formula_sidecar_lineage": "isolated_object_8d_formula_sidecar_v1",
  "formula_sidecar_substituted": false,
  "gold_or_teacher_admission_granted": false,
  "input_dimension": 8,
  "input_origin": "deterministic_linguistic_feature_hash",
  "input_sha256": "c93bd95186970c372029c88e03ddacddd94026c399483c17eda106f20b2e3fd4",
  "legal_acceptance_claimed": false,
  "legal_meaning_established": false,
  "lineage_id": "legacy_hub_v1",
  "linguistic_bundle_substituted": false,
  "linguistic_profile_id": "legacy-linguistic-features-8d/v1",
  "mock_embedding_model": "mock:stable-sha256",
  "mock_projection_substituted": false,
  "model_executed": true,
  "model_inference_executed": true,
  "producer_called": true,
  "projection_dimension": 8,
  "projection_equals_input": false,
  "projection_sha256": "3c3cc3bb0ac58fb20ab1049ac8c47dbe68bb378dd3661966c727886935d5c2b7",
  "read_only_checkpoint_load": true,
  "source_access_evidenced": true,
  "source_sha256": "8c227276ea3536b2edc0d3979140533a052b8b1f4385129b418701cf244aecb7",
  "source_text_sha256": "8c227276ea3536b2edc0d3979140533a052b8b1f4385129b418701cf244aecb7",
  "training_executed": false,
  "worker_launched": false
}
"""
)
EXPECTED_NATIVE384_PROBE: dict[str, object] = json.loads(
    r"""
{
  "active_lane_model_id": "thenlper/gte-small",
  "active_lane_revision": "17e1f347d17fe144873b1201da91788898c639cd",
  "admitted": false,
  "asset_names": [
    "1_Pooling/config.json",
    "config.json",
    "model.safetensors",
    "modules.json",
    "sentence_bert_config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt"
  ],
  "authored_fixture_only": true,
  "completion_authority": false,
  "constitution_roundtrip_ok": false,
  "decoder_executed": false,
  "detached_placeholder_bound_as_active_lane": false,
  "detached_placeholder_components": [
    "F15",
    "F17",
    "F18"
  ],
  "dimension": 384,
  "encoder_called": true,
  "encoder_unfrozen": false,
  "executed_dtype": "float32",
  "execution": {
    "backend": "sentence-transformers",
    "batch_size": 1,
    "cpu_threads": 1,
    "device": "cpu",
    "dtype": "float32",
    "eval_mode": true,
    "gradient_mode": "inference_mode",
    "kind": "native",
    "max_tokens": 512,
    "normalization": "l2",
    "pooling": "mean",
    "seed": 0,
    "truncation": false
  },
  "federal_corpus_scored": false,
  "formalized": false,
  "gold_or_teacher_admission_granted": false,
  "legal_acceptance_claimed": false,
  "legal_meaning_established": false,
  "model": {
    "dimension": 384,
    "model_id": "thenlper/gte-small",
    "revision": "17e1f347d17fe144873b1201da91788898c639cd"
  },
  "model_executed": true,
  "model_inference_executed": true,
  "model_revision_authenticated": false,
  "pinned_weight_sha256": "9a1eb90bbac323ea08aa5629b624fe6ae75db121b904799c2266a1e2c2de22d2",
  "producer_called": true,
  "qualification": {
    "formal_proof_verified": false,
    "model_revision_authenticated": false,
    "record_integrity": true,
    "runtime_computation_proven": false,
    "runtime_cryptographically_attested": false,
    "source_authority_authenticated": false
  },
  "receipt_schema": "autoencoder-embedding-production-v1",
  "receipt_sha256": "c15edf698afe464dac83c395ac206f31f0f94dd53e1c56796ac4d6bcb3954701",
  "runtime_computation_proven": false,
  "runtime_cryptographically_attested": false,
  "settings_matched": true,
  "source_authority_authenticated": false,
  "source_sha256": "8c227276ea3536b2edc0d3979140533a052b8b1f4385129b418701cf244aecb7",
  "status": "embedded",
  "status_counts": {
    "embedded": 1,
    "missing_input": 0,
    "token_limit_exceeded": 0
  },
  "storage_dtypes": [
    "F16",
    "I64"
  ],
  "training_executed": false,
  "vector_length": 384,
  "vector_sha256": "050c540cdfacda99e123871e132091d11624496f15ba8500f10654e43a5e53e2",
  "worker_launched": false
}
"""
)
EXPECTED_NATIVE768_PROBE: dict[str, object] = json.loads(
    r"""
{
  "admitted": false,
  "architecture": "NewForTokenClassification",
  "authored_fixture_only": true,
  "classifier_loaded": true,
  "classifier_logits_used_for_dense_embedding": false,
  "completion_authority": false,
  "constitution_roundtrip_ok": false,
  "coordinate_relabel_manufactured": false,
  "coordinate_zero_pad": false,
  "decoder_executed": false,
  "dense_path_equal": true,
  "dimension": 768,
  "download_executed": false,
  "encoder_called": true,
  "encoder_identity_authenticated": true,
  "encoder_unfrozen": false,
  "execution_profile": {
    "attention_implementation": "eager",
    "batch_size": 1,
    "device": "cpu",
    "dtype": "float32",
    "max_tokens_including_special_tokens": 8192,
    "normalization": "l2",
    "overlength_policy": "reject",
    "padding_side": "right",
    "pooling": "cls"
  },
  "experiment_max_tokens": 512,
  "federal_corpus_scored": false,
  "formalized": false,
  "gold_or_teacher_admission_granted": false,
  "implementation": {
    "asset_profile": "d6d6069c98c7f15bd4c6f8a43bd9ffb72032e04a452419a3ff5d9bd95bb06e8e",
    "complete_loader": "f1a536de2a64a3d8bb82f713ba62199d077eeb8c0e23f36ddac0346a42190619",
    "reference_producer": "cd4b23b87fefe3f80f234a5c85d9e4823139d5040a5507813b88385ee601b4ea"
  },
  "legal_acceptance_claimed": false,
  "legal_meaning_established": false,
  "manifest_sha256": "8beb874aa7b06599346173fde12e95f9f926b3028942d5014cdd2f99c4385166",
  "max_tokens_relabelled_as_512": false,
  "missing_keys": [],
  "model_executed": true,
  "model_inference_executed": true,
  "normalized": true,
  "producer_called": true,
  "profile_id": "Alibaba-NLP/gte-multilingual-base@9bbca17d9273fd0d03d5725c7a4b0f6b45142062:code=Alibaba-NLP/new-impl@40ced75c3017eb27626c9d4ea981bde21a2662f4:d768:pool=cls:norm=l2:cpu:float32:tokens8192:reject_overlength:v1",
  "profile_max_tokens": 8192,
  "proof_authority": false,
  "runtime_compatibility_verified": true,
  "source_semantics_verified": false,
  "source_sha256": "8c227276ea3536b2edc0d3979140533a052b8b1f4385129b418701cf244aecb7",
  "status": "completed",
  "tensor_count": 138,
  "token_count_including_special_tokens": 12,
  "training_executed": false,
  "truncated": false,
  "unexpected_keys": [],
  "vector_length": 768,
  "vector_sha256": "87a6d1e7454e61560de1f4bdd4778bcb73f46a223ef27d8ca798bfb510abec05",
  "worker_launched": false,
  "zero_padding_manufactured": false
}
"""
)
_SPARSE8_CRITERIA = frozenset(
    {
        "actual producer/encoder calls and declared source access are evidenced.",
        "mock projections and different historical 8d tasks are not substituted.",
    }
)
_NATIVE384_CRITERIA = frozenset(
    {
        "input/encoder/checkpoint/codec/output settings match the design.",
        "disabled placeholder models are separate source-only controls.",
    }
)
_NATIVE768_CRITERIA = frozenset(
    {
        (
            "full historical runtime closure and actual native encoder/input "
            "identity are authenticated."
        ),
        "no zero-padding or coordinate relabeling manufactures a native 768d input.",
    }
)
_SPARSE8_MARKERS = {
    "reconciliation": (
        "Full historical sparse state, separate small linguistic bundle, and learned 18-entry formula sidecar",
        "These are three distinct artifacts/tasks.",
        "The formula sidecar's two synthetic examples do not qualify statutory supervision.",
    ),
}
_NATIVE384_MARKERS = {
    "native_inputs": (
        "The maintained entry is [gte_decoder_native_inputs.md]",
        "This former path is retained for existing links; original bytes and provenance are preserved",
    ),
}
_NATIVE768_MARKERS = {
    "native_inputs": (
        "This boundary does not authenticate an encoder or manufacture a native 768D input.",
    ),
    "integration": (
        "The original guide is preserved in the",
        "they are not the current task queue or fresh qualification evidence.",
    ),
}


def sparse8_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the producer-call and non-substitution sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _SPARSE8_CRITERIA


def native384_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the settings-match and detached-placeholder sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _NATIVE384_CRITERIA


def native768_criteria_match(criteria: Sequence[object]) -> bool:
    """True for the encoder-identity and no-manufactured-input sentences."""

    texts = tuple(_normalized(item) for item in criteria)
    return len(texts) == 2 and frozenset(texts) == _NATIVE768_CRITERIA


def width_control_grants_no_authority(
    dependency_evidence: Sequence[Mapping[str, object]],
) -> bool:
    """True when the recorded width specification executed no call."""

    if _any_flag_true(dependency_evidence):
        return False
    found = False
    for item in dependency_evidence:
        for mapping in _mappings(item.get("body")):
            if mapping.get("operation") != WIDTH_CONTROL_OPERATION:
                continue
            if (
                mapping.get("producer_called") is not False
                or mapping.get("encoder_called") is not False
                or mapping.get("mean_coordinate_mse_measured") is not False
                or mapping.get("federal_corpus_scored") is not False
                or mapping.get("legal_acceptance_claimed") is not False
                or mapping.get("controls_executed") is not False
                or mapping.get("model_executed") is not False
                or mapping.get("training_executed") is not False
                or mapping.get("worker_launched") is not False
                or mapping.get("encoder_unfrozen") is not False
                or mapping.get("historical_8d_distinct") is not True
                or mapping.get("active_native_distinct") is not True
                or mapping.get("detached_disabled_latent_distinct") is not True
                or mapping.get("width_classes_distinct") is not True
            ):
                return False
            found = True
    return found


def _comparison_cell_evidence(
    *,
    matched: bool,
    markers: Mapping[str, Sequence[str]],
    operation: str,
    expected: Mapping[str, object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record one locked comparison cell when the width gate grants nothing."""

    checked = _owner_limits(owners, markers)
    cited = _cited_evidence(dependency_evidence)
    probe_flags = ({"body": probe},) if isinstance(probe, Mapping) else ()
    if (
        not matched
        or not width_control_grants_no_authority(dependency_evidence)
        or checked is None
        or probe != expected
        or not cited
        or _any_flag_true(probe_flags)
    ):
        return None
    limits, owner_sha256 = checked
    return {
        "operation": operation,
        **expected,
        "preserved_owner_limits": limits,
        "owner_sha256": owner_sha256,
        "cited_dependency_evidence": cited,
        **_authority_false(),
    }


def sparse8_comparison_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record the canonical 8D sparse projection for one authored sentence.

    The linguistic feature hash is the input. The recorded projection is the
    checkpoint encode. A mock, the linguistic bundle, and the formula sidecar
    stay unused. model_executed records that inference call. It does not
    train, admit, or score a federal corpus.
    """

    return _comparison_cell_evidence(
        matched=sparse8_criteria_match(criteria),
        markers=_SPARSE8_MARKERS,
        operation=SPARSE8_COMPARISON_OPERATION,
        expected=EXPECTED_SPARSE8_PROBE,
        dependency_evidence=dependency_evidence,
        owners=owners,
        probe=probe,
    )


def native384_comparison_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record the pinned 384D gte-small receipt for one authored sentence.

    Settings match the published execution profile. Float16 storage is cast
    to the float32 runtime. Detached placeholder components stay unbound.
    Qualification authentication flags stay false.
    """

    return _comparison_cell_evidence(
        matched=native384_criteria_match(criteria),
        markers=_NATIVE384_MARKERS,
        operation=NATIVE384_COMPARISON_OPERATION,
        expected=EXPECTED_NATIVE384_PROBE,
        dependency_evidence=dependency_evidence,
        owners=owners,
        probe=probe,
    )


def native768_comparison_evidence(
    criteria: Sequence[object],
    dependency_evidence: Sequence[Mapping[str, object]],
    owners: Mapping[str, Mapping[str, object]],
    probe: Mapping[str, object] | None,
) -> dict[str, object] | None:
    """Record the complete 768D encoder identity for one authored sentence.

    Identity is the published profile and the complete local loader. Token
    padding is not a manufactured 768-coordinate vector. The 8192-token
    profile is not relabelled as the 512-token experiment gate.
    """

    return _comparison_cell_evidence(
        matched=native768_criteria_match(criteria),
        markers=_NATIVE768_MARKERS,
        operation=NATIVE768_COMPARISON_OPERATION,
        expected=EXPECTED_NATIVE768_PROBE,
        dependency_evidence=dependency_evidence,
        owners=owners,
        probe=probe,
    )


def dependency_resolution_receipt(
    operation: str = DEPENDENCY_RESOLUTION_OPERATION,
) -> dict[str, object]:
    """Completion receipt that grants no admission, gold, or review authority."""

    if operation not in _RESOLUTION_OPERATIONS:
        raise ValueError(f"unknown dependency resolution operation: {operation}")
    return {"operation": operation, **_authority_false()}
