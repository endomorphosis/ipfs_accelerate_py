"""Audit finite recorded relation coverage without authorizing source omission.

The immutable native bundle supplies membership and relation provenance. This
does not verify current repository bytes, complete extraction, test relevance,
or an owner admission. Unknown target names remain exact recorded names. A
state-binding read is never repaired by a similarly named call target.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType

from .semantic_selection_dependency_audit import _encoded, _literal_json

SCHEMA = "supervisor-semantic-selection-relation-audit@1"
MAX_SYMBOLS = 1024
MAX_LINKS = 8192
MAX_BUNDLE_BYTES = 32_000_000
MAX_REPORT_BYTES = 2_000_000
MAX_BLOCKS = 65_536
MAX_BLOCK_BYTES = 2_000_000

_INDEX_FIELDS = (
    "symbol_fact", "artifact_fact", "symbol_node", "capsule",
    "semantic_link", "analysis_limitation",
)
_DIRECT_PROJECTION_FIELDS = (
    "dependency_stable_ids", "dependency_version_cids", "dependency_fact_cids",
    "dependency_link_ids", "test_refs", "fixture_refs", "proof_obligation_refs",
    "schema_relations", "serialization_relations",
)
_COVERAGE_FIELDS = (
    "test_refs", "fixture_refs", "proof_obligation_refs", "schema_relations",
    "serialization_relations", "effects",
)


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _bounded_bundle(bundle, root_cid, limits):
    from multiformats import CID

    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        SemanticStateBundle,
        SemanticStateRoot,
    )

    for name, value, ceiling in limits:
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be an integer from 1 to {ceiling}")
    if (type(bundle) is not SemanticStateBundle
            or type(bundle.root) is not SemanticStateRoot
            or type(bundle.blocks) is not MappingProxyType):
        raise ValueError("relation audit requires the exact immutable native bundle")
    if type(root_cid) is not str or not root_cid or len(root_cid) > 16_384:
        raise ValueError("literal bounded native root CID required")
    if root_cid != bundle.root.root_cid:
        raise ValueError("relation audit bundle/root mismatch")
    if len(bundle.blocks) > MAX_BLOCKS:
        raise ValueError("native bundle exceeds block count bound")
    bundle_root, bundle_blocks = bundle.root, bundle.blocks
    captured_blocks = {}
    total_bytes = 0
    byte_limit = dict((name, value) for name, value, _ in limits)["max_bundle_bytes"]
    # Preflight before the native canonical verifier or any indexed traversal.
    for cid, raw in bundle_blocks.items():
        if type(cid) is not str or len(cid) > 16_384 or type(raw) is not bytes:
            raise ValueError("native bundle requires literal CID/bytes blocks")
        total_bytes += len(raw)
        if len(raw) > MAX_BLOCK_BYTES or total_bytes > byte_limit:
            raise ValueError("native bundle exceeds byte bounds")
        try:
            if CID.decode(cid).codec.name == "dag-json":
                _literal_json(json.loads(raw))
        except (RecursionError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError("native structured block exceeds JSON bounds") from error
        captured_blocks[cid] = raw
    if (bundle.root is not bundle_root or bundle.blocks is not bundle_blocks
            or not _same_literal_blocks(bundle_blocks, captured_blocks)
            or bundle_root.root_cid != root_cid):
        raise ValueError("native relation inventory changed during preflight")
    return total_bytes, bundle_root, bundle_blocks, MappingProxyType(captured_blocks)


def _index(view, name):
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        SortedPairIndex,
    )

    cid = getattr(view.root, name + "_index_cid")
    payload = json.loads(view.get_block(cid))
    return SortedPairIndex.from_dict({**payload, "index_cid": cid})


def _checked_provenance(metadata):
    """Admit only typed scalar provenance, never arbitrary metadata bodies."""
    from ipfs_datasets_py.logic.software_contracts.semantic_index.models import (
        _text,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_index.symbol_graph import (
        STATUS_DEFINITE,
        STATUS_FINITE_MAY,
        STATUS_UNRESOLVED,
    )

    resolution = metadata.get("resolution")
    if resolution is not None and (
            type(resolution) is not str
            or resolution not in {STATUS_DEFINITE, STATUS_FINITE_MAY, STATUS_UNRESOLVED}):
        raise ValueError("native link resolution provenance requires a supported scalar status")
    target = metadata.get("unresolved_target")
    if target is not None:
        if type(target) is not str:
            raise ValueError("native link unresolved target provenance requires literal text")
        _text(target, "unresolved_target")
        try:
            target.encode("utf-8")
        except UnicodeError as error:
            raise ValueError("native link unresolved target provenance is not UTF-8 text") from error


def _edge(link):
    """Recover an exact producer edge; missing unresolved text refuses."""
    from ipfs_datasets_py.logic.software_contracts.semantic_index.models import (
        DependencyEdge,
    )

    payload = link.to_dict()
    metadata = payload["metadata"]
    _checked_provenance(metadata)
    target = link.target_stable_id
    if link.target_kind == "unresolved":
        target = metadata.get("unresolved_target")
        if type(target) is not str or not target:
            raise ValueError("unresolved native link lacks recorded target provenance")
    edge = DependencyEdge(
        source_id=link.source_stable_id, target_id=target,
        relation=link.relation, extraction_method=link.extraction_method,
        confidence=link.confidence, extractor_version=link.extractor_version,
        span=link.source_span, metadata=metadata,
    )
    if edge.edge_id != link.edge_id:
        raise ValueError("native link target provenance does not reproduce edge_id")
    return edge


def _row(link, *, root_cid):
    payload = link.to_dict()
    metadata = payload["metadata"]
    return {
        "root_cid": root_cid,
        **{field: payload[field] for field in (
            "edge_id", "link_cid", "source_stable_id", "source_version_cid",
            "source_fact_cid", "target_kind", "target_stable_id",
            "target_version_cid", "target_fact_cid", "relation", "source_span",
            "extraction_method", "confidence", "extractor_version",
        )},
        "resolution": metadata.get("resolution"),
        "unresolved_target": metadata.get("unresolved_target"),
        "metadata_sha256": _digest(metadata),
    }


def _counts(rows):
    return dict(sorted(Counter(row["relation"] for row in rows).items()))


@dataclass(frozen=True, slots=True)
class _VerifiedRelationInventory:
    """Private per-operation native records; never accepted from public input."""

    bundle: object
    bundle_root: object
    bundle_blocks: MappingProxyType
    captured_blocks: MappingProxyType
    root: object
    root_cid: str
    bundle_bytes: int
    limits: tuple
    indexes: MappingProxyType
    symbol_ids: frozenset
    facts: MappingProxyType
    artifact_facts: MappingProxyType
    limitations: tuple
    links: tuple
    outgoing: MappingProxyType
    capsules: MappingProxyType
    coverage_counts: MappingProxyType
    reason_counts: MappingProxyType
    ledger_row_bytes: tuple
    full_capsule_identities_sha256: str
    full_relation_ledger_sha256: str


def _same_literal_blocks(blocks, captured_blocks):
    """Compare exact literal entries before any equality can admit subclasses."""
    if (type(blocks) is not MappingProxyType
            or len(blocks) != len(captured_blocks)):
        return False
    try:
        for cid, raw in blocks.items():
            # bytearray, bytes subclasses and equal string subclasses are not
            # native immutable entries, even when their equality says otherwise.
            if type(cid) is not str or type(raw) is not bytes:
                return False
            if cid not in captured_blocks or raw != captured_blocks[cid]:
                return False
    except RuntimeError:
        # An illicit mutable-backed proxy can change size during traversal.
        return False
    return True


def _check_relation_inventory(inventory):
    """Reject replacement or mutation of the original finite byte inventory."""
    if type(inventory) is not _VerifiedRelationInventory:
        raise ValueError("relation inventory requires the internal native inventory")
    bundle = inventory.bundle
    if (bundle.root is not inventory.bundle_root
            or bundle.blocks is not inventory.bundle_blocks
            or bundle.root.root_cid != inventory.root_cid
            or not _same_literal_blocks(bundle.blocks, inventory.captured_blocks)):
        raise ValueError("native relation inventory changed during operation")


def _verified_relation_inventory(
    *, bundle, root_cid: str, selected_symbol_ids: list[str],
    max_symbols: int = MAX_SYMBOLS, max_links: int = MAX_LINKS,
    max_bundle_bytes: int = MAX_BUNDLE_BYTES,
    max_report_bytes: int = MAX_REPORT_BYTES,
) -> _VerifiedRelationInventory:
    """Build one finite inventory through the complete native verification path.

    Reverify full root index membership, source/target fact versions, node
    adjacency and capsule relation projections using native verification and
    the native dependency projection. Unresolved links need exact recorded
    target provenance to reconstruct their producer edge identity; bundles
    lacking it refuse rather than pairing names by order or spelling.

    A finite recorded-coverage Boolean concerns only this root. Eligibility is
    always experimental or refused; current source, owner admission, extraction
    completeness and effective task coverage remain outside this operation.
    """
    from ipfs_datasets_py.logic.software_contracts.semantic_state.api import (
        open_semantic_state,
        verify_semantic_state_bundle,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_state.capsules import (
        _DEPENDENCY_RELATIONS,
        _as_text_sequence,
        _dependency_payload,
        _thaw_mapping,
        capsule_source_key,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        CAPSULE_COMPILER_VERSION,
        SEMANTIC_CAPSULE_SCHEMA,
    )

    limits = (
        ("max_symbols", max_symbols, MAX_SYMBOLS),
        ("max_links", max_links, MAX_LINKS),
        ("max_bundle_bytes", max_bundle_bytes, MAX_BUNDLE_BYTES),
        ("max_report_bytes", max_report_bytes, MAX_REPORT_BYTES),
    )
    # Preflight owns the captured bytes and bounds before native traversal.
    # A caller cannot switch generations between seed, graph and nominee stages.
    bundle_bytes, bundle_root, bundle_blocks, captured_blocks = _bounded_bundle(
        bundle, root_cid, limits,
    )
    if (type(selected_symbol_ids) is not list or len(selected_symbol_ids) > max_symbols
            or any(type(sid) is not str or not sid or len(sid) > 16_384
                   for sid in selected_symbol_ids)):
        raise ValueError("selected symbol IDs require a bounded literal list")
    selected = set(selected_symbol_ids)
    if len(selected) != len(selected_symbol_ids):
        raise ValueError("selected symbol IDs must be distinct")

    verify_semantic_state_bundle(bundle)
    view = open_semantic_state(root_cid, captured_blocks.__getitem__)
    if (view.root.capsule_schema != SEMANTIC_CAPSULE_SCHEMA
            or view.root.capsule_compiler_version != CAPSULE_COMPILER_VERSION):
        raise ValueError("relation audit requires the supported native capsule compiler")
    indexes = {name: _index(view, name) for name in _INDEX_FIELDS}
    for name, index in indexes.items():
        ceiling = max_symbols if name in {"symbol_fact", "symbol_node", "capsule"} else max_links
        if len(index.pairs) > ceiling:
            raise ValueError(f"native {name} index exceeds its bound")
    members = {name: set(dict(index.pairs)) for name, index in indexes.items()}
    symbol_ids = members["symbol_fact"]
    if symbol_ids != members["symbol_node"] or symbol_ids != members["capsule"]:
        raise ValueError("native fact/node/capsule index membership disagrees")
    if not selected <= symbol_ids:
        raise ValueError("selected symbols are foreign to the native inventory")

    facts = {sid: view.symbol_fact(sid) for sid in sorted(symbol_ids)}
    for fact in facts.values():
        if fact.repository_id != view.root.repository_id:
            raise ValueError("native symbol fact repository differs from root")
    artifact_facts = {aid: view.artifact_fact(aid)
                      for aid in sorted(members["artifact_fact"])}
    limitations = [view.analysis_limitation(cid) for cid in sorted(members["analysis_limitation"])]

    links = [view.semantic_link(eid) for eid in sorted(members["semantic_link"])]
    edges = {link.edge_id: _edge(link) for link in links}
    for link in links:
        target = edges[link.edge_id].target_id
        # Native compilation gives symbol inventory membership precedence over
        # artifact membership, including a collision between their logical IDs.
        expected_kind = ("symbol" if target in members["symbol_fact"] else
                         "artifact" if target in members["artifact_fact"] else "unresolved")
        if link.target_kind != expected_kind:
            raise ValueError("native link target kind disagrees with root inventory membership")
    outgoing = {sid: [] for sid in symbol_ids}
    incoming = {sid: [] for sid in symbol_ids}
    for link in links:
        outgoing[link.source_stable_id].append(link)
        if link.target_kind == "symbol":
            incoming[link.target_stable_id].append(link)

    capsules = {}
    coverage_counts = {}
    reason_counts = {}
    for sid in sorted(symbol_ids):
        node = view.symbol_node(sid)
        capsule = view.capsule(sid)
        fact = facts[sid]
        if (node.symbol_fact_cid != fact.fact_cid or node.version_cid != fact.version_cid
                or node.capsule_cid != capsule.capsule_cid
                or capsule.symbol_fact_cid != fact.fact_cid
                or capsule.version_cid != fact.version_cid
                or capsule.source_cid != fact.source_cid
                or capsule.source_slice_path != fact.symbol.module_path
                or capsule.producer_key() != capsule_source_key(fact.symbol)
                or capsule.capsule_schema != view.root.capsule_schema
                or capsule.capsule_compiler_version != view.root.capsule_compiler_version):
            raise ValueError("native capsule/node fact or version bindings disagree")
        if set(node.outgoing_link_cids) != {link.link_cid for link in outgoing[sid]}:
            raise ValueError("native node outgoing adjacency disagrees with root links")
        if set(node.incoming_link_cids) != {link.link_cid for link in incoming[sid]}:
            raise ValueError("native node incoming adjacency disagrees with root links")

        projection = _dependency_payload(
            [edges[link.edge_id] for link in outgoing[sid]], fact_by_stable_id=facts,
        )
        payload = capsule.to_dict()
        for field in _DIRECT_PROJECTION_FIELDS:
            if payload[field] != projection[field]:
                raise ValueError(f"native capsule {field} relation projection disagrees")
        # Reproduce the current native compiler's merge with sealed symbol
        # metadata. Exact field agreement still cannot prove complete extraction.
        symbol_metadata = _thaw_mapping(fact.symbol.metadata)
        expected_effects = sorted(set(_as_text_sequence(symbol_metadata.get("effects")))
                                  | set(projection["effects"]))
        if payload["effects"] != expected_effects:
            raise ValueError("native capsule effects relation projection disagrees")
        expected_exceptions = _thaw_mapping(symbol_metadata.get("exception_behavior"))
        for kind, values in projection["exception_behavior"].items():
            retained = expected_exceptions.get(kind)
            expected_exceptions[kind] = (sorted(set(retained) | set(values))
                                         if isinstance(retained, list) else list(values))
        if payload["exception_behavior"] != expected_exceptions:
            raise ValueError("native capsule exceptions relation projection disagrees")
        capsules[sid] = capsule
        coverage_counts[sid] = {field: len(payload[field]) for field in _COVERAGE_FIELDS}
        reason_counts[sid] = len(node.raw_source_required_reasons)

    all_rows = [_row(link, root_cid=root_cid) for link in links]
    inventory = _VerifiedRelationInventory(
        bundle=bundle, bundle_root=bundle_root, bundle_blocks=bundle_blocks,
        captured_blocks=captured_blocks, root=view.root, root_cid=root_cid,
        bundle_bytes=bundle_bytes, limits=limits,
        indexes=MappingProxyType(indexes), symbol_ids=frozenset(symbol_ids),
        facts=MappingProxyType(facts), artifact_facts=MappingProxyType(artifact_facts),
        limitations=tuple(limitations), links=tuple(links),
        outgoing=MappingProxyType({sid: tuple(value) for sid, value in outgoing.items()}),
        capsules=MappingProxyType(capsules),
        coverage_counts=MappingProxyType({sid: MappingProxyType(value)
                                        for sid, value in coverage_counts.items()}),
        reason_counts=MappingProxyType(reason_counts),
        # Every renderer owns fresh literal rows; report mutation cannot become
        # an authority input to the following render or the graph builder.
        ledger_row_bytes=tuple(_encoded(row) for row in all_rows),
        full_capsule_identities_sha256=_digest(
            [[sid, capsules[sid].capsule_cid] for sid in sorted(capsules)]),
        full_relation_ledger_sha256=_digest(all_rows),
    )
    _check_relation_inventory(inventory)
    return inventory


def _relation_audit_from_inventory(inventory, selected_symbol_ids: list[str]) -> dict:
    """Render selection-specific diagnostics from one private verified inventory."""
    from ipfs_datasets_py.logic.software_contracts.semantic_state.capsules import (
        _DEPENDENCY_RELATIONS,
    )
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        CAPSULE_COMPILER_VERSION,
    )

    _check_relation_inventory(inventory)
    limits = inventory.limits
    bounds = {name: value for name, value, _ in limits}
    max_symbols, max_report_bytes = bounds["max_symbols"], bounds["max_report_bytes"]
    if (type(selected_symbol_ids) is not list or len(selected_symbol_ids) > max_symbols
            or any(type(sid) is not str or not sid or len(sid) > 16_384
                   for sid in selected_symbol_ids)):
        raise ValueError("selected symbol IDs require a bounded literal list")
    selected = set(selected_symbol_ids)
    if len(selected) != len(selected_symbol_ids):
        raise ValueError("selected symbol IDs must be distinct")
    if not selected <= inventory.symbol_ids:
        raise ValueError("selected symbols are foreign to the native inventory")
    root_cid = inventory.root_cid
    capsules, links = inventory.capsules, inventory.links
    coverage_counts, reason_counts = inventory.coverage_counts, inventory.reason_counts
    limitations, bundle_bytes = inventory.limitations, inventory.bundle_bytes
    all_rows = [json.loads(raw) for raw in inventory.ledger_row_bytes]
    outgoing_rows = [row for row in all_rows if row["source_stable_id"] in selected]
    omitted_rows = [row for row in outgoing_rows
                    if row["target_kind"] == "symbol" and row["target_stable_id"] not in selected]
    incoming_omitted = [row for row in all_rows
                        if row["target_kind"] == "symbol" and row["target_stable_id"] in selected
                        and row["source_stable_id"] not in selected]
    artifacts = [row for row in outgoing_rows if row["target_kind"] == "artifact"]
    unresolved = [row for row in outgoing_rows if row["target_kind"] == "unresolved"]
    state_unknown = [row for row in unresolved if row["relation"] in {"reads_state", "writes_state"}]
    other_unknown = [row for row in unresolved if row["relation"] not in {"reads_state", "writes_state"}]
    missing_calls = sorted({row["target_stable_id"] for row in omitted_rows if row["relation"] == "calls"})
    missing_other = sorted({row["target_stable_id"] for row in omitted_rows
                            if row["relation"] in _DEPENDENCY_RELATIONS and row["relation"] != "calls"})
    missing_tests = sorted({row["target_stable_id"] for row in omitted_rows if row["relation"] == "tested_by"})
    reasons = ["current_source_and_owner_admission_unverified", "complete_extraction_and_task_coverage_unverified"]
    for present, reason in (
        (not selected, "empty_selection"),
        (missing_calls, "known_call_dependency_omitted"),
        (missing_other, "known_other_dependency_omitted"),
        (missing_tests, "known_test_relation_omitted"),
        (omitted_rows, "known_outgoing_relation_target_omitted"),
        (incoming_omitted, "incoming_caller_or_relation_omitted"),
        (state_unknown, "unresolved_state_binding"),
        (other_unknown, "unresolved_other_reference"),
        (artifacts, "artifact_relation_effective_coverage_unverified"),
        (limitations, "native_analysis_limitations_present"),
        (any(reason_counts[sid] for sid in selected), "native_raw_source_requirement_present"),
    ):
        if present:
            reasons.append(reason)
    report = {
        "schema": SCHEMA,
        "input_basis": "exact_immutable_native_bundle",
        "coverage_scope": "recorded_root_relations_only_not_producer_completeness",
        "root_cid": root_cid,
        "index_cids": {name: getattr(inventory.root, name + "_index_cid") for name in _INDEX_FIELDS},
        "selected_symbol_ids": sorted(selected),
        "full_capsule_count": len(capsules), "selected_capsule_count": len(selected),
        "full_link_count": len(links), "bundle_block_count": len(inventory.captured_blocks),
        "bundle_bytes": bundle_bytes,
        "full_capsule_identities_sha256": inventory.full_capsule_identities_sha256,
        "full_relation_ledger_sha256": inventory.full_relation_ledger_sha256,
        "selection_sha256": _digest(sorted(selected)),
        "outgoing_relation_rows": outgoing_rows,
        "omitted_outgoing_relation_rows": omitted_rows,
        "incoming_omitted_relation_rows": incoming_omitted,
        "resolved_artifact_relation_rows": artifacts,
        "missing_known_call_dependency_symbol_ids": missing_calls,
        "missing_known_other_dependency_symbol_ids": missing_other,
        "omitted_outgoing_test_symbol_ids": missing_tests,
        "unresolved_state_binding_rows": state_unknown,
        "unresolved_other_reference_rows": other_unknown,
        "unresolved_target_association": "exact_recorded_metadata_and_edge_id_reconstruction",
        "provenance_scalar_fields_verified": True,
        "link_target_kinds_match_root_inventory": True,
        "capsule_coverage_field_counts": {sid: dict(coverage_counts[sid]) for sid in sorted(selected)},
        "raw_source_required_reason_counts": {sid: reason_counts[sid] for sid in sorted(selected)},
        "native_analysis_limitation_count": len(limitations),
        "native_analysis_limitation_ids": [item.limitation_cid for item in limitations],
        "relation_counts": {"full": _counts(all_rows), "selected_outgoing": _counts(outgoing_rows),
                            "incoming_omitted": _counts(incoming_omitted)},
        "root_index_and_fact_bindings_verified": True,
        "root_index_records_verified": True,
        "node_adjacency_verified": True,
        "native_capsule_relation_projection_verified": True,
        "capsule_relation_projections_verified": True,
        "metadata_effect_exception_field_bindings_verified": True,
        "metadata_effect_exception_completeness_verified": False,
        "supported_capsule_compiler_version": CAPSULE_COMPILER_VERSION,
        "declared_known_call_closure_selected": bool(selected) and not missing_calls
            and not any(row["relation"] == "calls" for row in unresolved),
        "selected_declared_relation_coverage_complete": bool(selected)
            and not any((omitted_rows, incoming_omitted, artifacts, unresolved, limitations))
            and not any(reason_counts[sid] for sid in selected),
        "selection_eligibility": "experimental_only" if selected else "refused_empty_selection",
        "selection_refusal_reasons": reasons,
        "source_freshness_verified": False, "dependency_semantics_complete": False,
        "full_program_completeness_verified": False, "test_task_impact_relevance_complete": False,
        "effective_task_coverage_verified": False, "source_substitution_authority": False,
        "execution_authority": False, "proof_authority": False, "completion_authority": False,
        "acceptance_verified": False, "deployment_ready": False, "activation_authority": False,
        "task_acceptance_verified": False, "program_equivalence_verified": False,
        "runtime_activation": False,
        "source_bodies_in_report": False, "report_is_provider_input": False,
        "source_body_fields_in_report": False,
        "bounds": {name: value for name, value, _ in limits},
    }
    _literal_json(report)
    if len(_encoded(report)) > max_report_bytes:
        raise ValueError("relation audit report exceeds byte bound")
    _check_relation_inventory(inventory)
    return report


def audit_selected_capsule_relations(
    *, bundle, root_cid: str, selected_symbol_ids: list[str],
    max_symbols: int = MAX_SYMBOLS, max_links: int = MAX_LINKS,
    max_bundle_bytes: int = MAX_BUNDLE_BYTES,
    max_report_bytes: int = MAX_REPORT_BYTES,
) -> dict:
    """Independently verify one finite native root and audit recorded relations.

    Current source, owner admission, extraction completeness, and effective task
    coverage remain outside this operation. Internal inventory reuse is scoped
    to one call and never accepts report verification bits as authority.
    """
    inventory = _verified_relation_inventory(
        bundle=bundle, root_cid=root_cid, selected_symbol_ids=selected_symbol_ids,
        max_symbols=max_symbols, max_links=max_links,
        max_bundle_bytes=max_bundle_bytes, max_report_bytes=max_report_bytes,
    )
    return _relation_audit_from_inventory(inventory, selected_symbol_ids)
