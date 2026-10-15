"""Bounded advisory nomination over a supplied native capsule graph.

This helper copies whole records; it never constructs a worker artifact or
changes native selection, source checks, admissions, storage, or routing. A
closed declared graph is not a complete description of program semantics.
"""

from __future__ import annotations

import hashlib
import json

from .semantic_selection_dependency_audit import (
    MAX_CAPSULE_BYTES,
    MAX_EDGES,
    MAX_NODES,
    MAX_TOTAL_CAPSULE_BYTES,
    _encoded,
    _literal_json,
    audit_selected_capsule_dependencies,
)

SCHEMA = "supervisor-semantic-capsule-closure-nomination@1"
MAX_CAPSULES = 32
MAX_PACKET_BYTES = 65_536
MAX_REPORT_BYTES = 2_000_000


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def _bounded_snapshot(full_capsules, seed_symbol_ids, *, max_nodes):
    """Snapshot literal JSON before native parsing, within existing audit caps."""
    if type(full_capsules) is not list or type(seed_symbol_ids) is not list:
        raise ValueError("capsule inventory and seed IDs must be literal lists")
    if len(full_capsules) > max_nodes or len(seed_symbol_ids) > max_nodes:
        raise ValueError("capsule inventory or seeds exceed node bound")
    if any(type(seed) is not str for seed in seed_symbol_ids):
        raise ValueError("seed IDs must be literal strings")
    if any(len(seed) > 16_384 for seed in seed_symbol_ids):
        raise ValueError("seed IDs exceed the native text bound")

    snapshot = []
    total_bytes = 0
    for payload in full_capsules:
        if type(payload) is not dict:
            raise ValueError("each native capsule must be a literal dictionary")
        _literal_json(payload)
        raw = _encoded(payload)
        total_bytes += len(raw)
        if len(raw) > MAX_CAPSULE_BYTES or total_bytes > MAX_TOTAL_CAPSULE_BYTES:
            raise ValueError("capsule inventory exceeds audit byte bounds")
        snapshot.append(json.loads(raw))
    return snapshot, list(seed_symbol_ids)


def nominate_declared_capsule_closure(
    *, full_capsules: list[dict], seed_symbol_ids: list[str],
    max_capsules: int = MAX_CAPSULES, max_packet_bytes: int = MAX_PACKET_BYTES,
    max_nodes: int = MAX_NODES, max_edges: int = MAX_EDGES,
) -> dict:
    """Nominate every reachable known native capsule without partial closure.

    Capsule count and bytes bound the complete canonical ``nominee_packet``,
    including its wrappers and bindings. Measurements and diagnostics are a
    separate receipt that is never provider input. An insufficient candidate
    budget returns a refusal with no nominee bodies or nominee IDs. That
    diagnostic refusal can exceed the caller's tiny candidate budget. The
    entire report has an independent two-megabyte hard bound. Invalid input,
    native identities, graph limits, or excessive diagnostics raise ``ValueError``.

    Symbolic native dependencies remain unresolved text. Neither this function
    nor a declared-graph closure result establishes current source, complete
    dependency extraction, acceptance, proof, or source substitution authority.
    """
    for name, value, ceiling in (
        ("max_capsules", max_capsules, MAX_CAPSULES),
        ("max_packet_bytes", max_packet_bytes, MAX_PACKET_BYTES),
        ("max_nodes", max_nodes, MAX_NODES),
        ("max_edges", max_edges, MAX_EDGES),
    ):
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be an integer from 1 to {ceiling}")

    records, seeds = _bounded_snapshot(full_capsules, seed_symbol_ids, max_nodes=max_nodes)
    audit = audit_selected_capsule_dependencies(
        full_capsules=records, selected_symbol_ids=seeds,
        max_nodes=max_nodes, max_edges=max_edges,
    )
    known_ids = audit["known_transitive_closure_symbol_ids"]
    nomination_audit = audit_selected_capsule_dependencies(
        full_capsules=records, selected_symbol_ids=list(known_ids),
        max_nodes=max_nodes, max_edges=max_edges,
    )
    for key in ("full_capsule_identities_sha256", "declared_graph_sha256"):
        if audit[key] != nomination_audit[key]:
            raise ValueError("seed and closure audits differ on the snapshotted inventory")
    by_symbol = {record["stable_symbol_id"]: record for record in records}
    bodies = [by_symbol[symbol_id] for symbol_id in known_ids]
    identities = [
        {"stable_symbol_id": record["stable_symbol_id"], "capsule_cid": record["capsule_cid"]}
        for record in bodies
    ]
    unresolved = list(audit["unresolved_dependency_symbol_ids"])
    no_authority = {
        "source_freshness_verified": False,
        "dependency_semantics_complete": False,
        "full_program_completeness_verified": False,
        "test_task_impact_relevance_complete": False,
        "source_substitution_authority": False,
        "execution_authority": False,
        "proof_authority": False,
        "completion_authority": False,
        "acceptance_verified": False,
        "deployment_ready": False,
        "activation_authority": False,
    }
    packet = {
        "schema": "supervisor-semantic-capsule-closure-nominee-packet@1",
        "input_basis": "caller_supplied_native_capsules_revalidated",
        "edge_basis": "outgoing_declared_dependency_stable_ids_only",
        "origin": {
            "full_capsule_count": audit["full_capsule_count"],
            "full_capsule_identities_sha256": audit["full_capsule_identities_sha256"],
            "declared_graph_sha256": audit["declared_graph_sha256"],
        },
        "initial_seed_symbol_ids": list(audit["selected_symbol_ids"]),
        "seed_selection_sha256": audit["selection_sha256"],
        "nominee_symbol_ids": list(known_ids),
        "nominee_selection_sha256": nomination_audit["selection_sha256"],
        "capsules": bodies,
        "capsule_identities": identities,
        "unresolved_dependency_symbol_ids": unresolved,
        "declared_graph_closed": bool(seeds) and nomination_audit["declared_graph_closure_selected"],
        "declared_graph_closed_scope": "reachable_supplied_native_records_and_declared_edges_only",
        "cycle_present": audit["cycle_present"],
        "cycle_analysis_scope": audit["cycle_analysis_scope"],
        **no_authority,
    }
    packet_raw = _encoded(packet)
    report = {
        "schema": SCHEMA,
        "input_basis": "caller_supplied_native_capsules_revalidated",
        "nomination_status": (
            "nomination_available_with_unresolved" if unresolved else
            "nomination_available_declared_graph_closed"
        ),
        "nomination_available": bool(seeds),
        "initial_seed_symbol_ids": list(audit["selected_symbol_ids"]),
        "added_known_symbol_ids": sorted(set(known_ids) - set(seeds)),
        "known_closure_symbol_ids": list(known_ids),
        "nominee_symbol_ids": list(known_ids),
        "nominee_packet": packet if seeds else None,
        "nominee_packet_bytes": len(packet_raw) if seeds else None,
        "nominee_packet_sha256": hashlib.sha256(packet_raw).hexdigest() if seeds else None,
        "known_closure_capsule_count": len(bodies),
        "known_closure_capsules_sha256": _digest(bodies),
        "known_closure_capsule_identities_sha256": _digest(identities),
        "unresolved_dependency_symbol_ids": unresolved,
        "declared_graph_closed": bool(seeds) and nomination_audit["declared_graph_closure_selected"],
        "declared_graph_closed_scope": "reachable_supplied_native_records_and_declared_edges_only",
        "graph_diagnostics": audit,
        "nomination_graph_diagnostics": nomination_audit,
        "cycle_present": audit["cycle_present"],
        "cycle_analysis_scope": audit["cycle_analysis_scope"],
        "nomination_refusal_reasons": [],
        "packet_budget_scope": "nominee_packet_only",
        "report_is_provider_input": False,
        "native_worker_packet_constructed": False,
        "edge_basis": "outgoing_declared_dependency_stable_ids_only",
        "bounds": {
            "max_capsules": max_capsules, "max_packet_bytes": max_packet_bytes,
            "max_nodes": max_nodes, "max_edges": max_edges,
            "max_report_bytes": MAX_REPORT_BYTES,
        },
        **no_authority,
    }
    if not seeds:
        report["nomination_status"] = "nomination_refused_empty_seeds"
        report["nomination_refusal_reasons"] = ["empty_seed_selection"]
    else:
        reasons = []
        if len(bodies) > max_capsules:
            reasons.append("known_closure_exceeds_capsule_budget")
        if len(packet_raw) > max_packet_bytes:
            reasons.append("full_candidate_packet_exceeds_byte_budget")
        if reasons:
            report.update({
                "nomination_status": "nomination_refused_budget",
                "nomination_available": False,
                "nominee_symbol_ids": [],
                "nominee_packet": None,
                "nominee_packet_bytes": None,
                "nominee_packet_sha256": None,
                "nomination_refusal_reasons": reasons,
                "attempted_candidate_packet_bytes": len(packet_raw),
                "attempted_candidate_packet_sha256": hashlib.sha256(packet_raw).hexdigest(),
            })
    if len(_encoded(report)) > MAX_REPORT_BYTES:
        raise ValueError("nomination diagnostics exceed the hard report byte bound")
    return report
