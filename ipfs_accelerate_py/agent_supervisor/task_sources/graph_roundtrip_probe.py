"""Measure the candidate graph round-trip without accepting an assertion.

Closed endpoints, role ordinals, source and interpretation companions, and
distinct alternative identities survive a bundle reload. Review receipts stay
separately addressed. A changed contract or a failed reload returns None.
"""

from __future__ import annotations

import hashlib
import tempfile
from pathlib import Path

from ipfs_datasets_py.utils.cid_utils import canonical_dag_json_bytes

from .dependency_block_resolution import EXPECTED_GRAPH_ROUNDTRIP_PROBE


def measure_graph_roundtrip_probe() -> dict[str, object] | None:
    """Return the fixed round-trip contract, or None when the owners disagree."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_GRAPH_ROUNDTRIP_PROBE:
        return None
    return measured


def _cid(seal_record, label: str) -> str:
    return seal_record({"schema": "graph-roundtrip-probe/v1", "label": label})["record_cid"]


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import corpus_graph as graph
    from ipfs_datasets_py.logic.autoformal import corpus_graph_exchange as exchange
    from ipfs_datasets_py.logic.autoformal.corpus_contracts import (
        CorpusContractError,
        seal_record,
    )

    def cid(label: str) -> str:
        return _cid(seal_record, label)

    source = graph.build_node(
        node_type="source_version",
        stable_id="source",
        content_cid=cid("source-content"),
    )
    interpretation = graph.build_node(
        node_type="interpretation",
        stable_id="interpretation",
        content_cid=cid("interpretation-content"),
    )
    review_receipt = cid("review-receipt")

    def assertion(label: str, alternative: str):
        return graph.build_node(
            node_type="norm_assertion",
            stable_id=label,
            content_cid=cid("content-" + label),
            source_occurrence_cids=[source["record_cid"]],
            interpretation_cids=[interpretation["record_cid"]],
            alternative_group_id="theory",
            alternative_id=alternative,
        )

    reading_a = assertion("reading-a", "A")
    reading_b = assertion("reading-b", "B")
    actor = graph.build_node(
        node_type="entity",
        stable_id="actor",
        content_cid=cid("actor-content"),
    )
    first_object = graph.build_node(
        node_type="expression",
        stable_id="object-one",
        content_cid=cid("object-one-content"),
    )
    second_object = graph.build_node(
        node_type="expression",
        stable_id="object-two",
        content_cid=cid("object-two-content"),
    )

    def edge(src, dst, relation: str, ordinal):
        evidence = review_receipt if relation == "HAS_ACTOR" else cid(
            relation + "-" + src["stable_id"] + "-" + str(ordinal)
        )
        return graph.build_edge(
            src_cid=src["record_cid"],
            dst_cid=dst["record_cid"],
            relation=relation,
            ordinal=ordinal,
            evidence_cids=[evidence],
        )

    snapshot = graph.build_graph_snapshot(
        [source, interpretation, reading_a, reading_b, actor, first_object, second_object],
        [
            edge(reading_a, actor, "HAS_ACTOR", 0),
            edge(reading_a, first_object, "HAS_OBJECT", 0),
            edge(reading_a, second_object, "HAS_OBJECT", 1),
        ],
    )
    checked = graph.validate_graph_snapshot(snapshot)

    def refusal(call) -> str:
        try:
            call()
        except CorpusContractError as exc:
            return str(exc)
        raise RuntimeError("graph round-trip accepted a refused shape")

    dangling = refusal(
        lambda: graph.build_graph_snapshot(
            [reading_a, actor],
            [
                graph.build_edge(
                    src_cid=reading_a["record_cid"],
                    dst_cid=cid("missing-endpoint"),
                    relation="HAS_ACTOR",
                    ordinal=0,
                    evidence_cids=[review_receipt],
                )
            ],
        )
    )
    duplicate = refusal(
        lambda: graph.build_graph_snapshot(
            [reading_a, actor, first_object],
            [
                edge(reading_a, actor, "HAS_ACTOR", 0),
                edge(reading_a, first_object, "HAS_ACTOR", 0),
            ],
        )
    )
    crossed = refusal(
        lambda: graph.build_graph_snapshot(
            [reading_a, reading_b],
            [
                graph.build_edge(
                    src_cid=reading_a["record_cid"],
                    dst_cid=reading_b["record_cid"],
                    relation="DERIVED_FROM",
                    evidence_cids=[cid("cross-alternative")],
                )
            ],
        )
    )
    promoted = dict(checked)
    promoted["admitted"] = True
    promoted = seal_record(
        {key: value for key, value in promoted.items() if key not in {"record_cid", "cid_scope"}}
    )
    accepted_refusal = refusal(lambda: graph.validate_graph_snapshot(promoted))
    with tempfile.TemporaryDirectory() as directory:
        bundle = Path(directory) / "bundle"
        manifest = exchange.write_graph_bundle(checked, bundle)
        digest = hashlib.sha256(canonical_dag_json_bytes(manifest)).hexdigest()
        loaded = exchange.load_graph_bundle(bundle, expected_manifest_sha256=digest)
    if loaded.get("snapshot") != checked:
        raise RuntimeError("graph bundle reload changed the snapshot")
    nodes = checked["nodes"]
    edges = checked["edges"]
    by_id = {node["stable_id"]: node for node in nodes}
    roles = sorted(
        (edge["relation"], edge["ordinal"])
        for edge in edges
        if edge["relation"] in graph.ROLES
    )
    alternatives = [
        node for node in nodes if node["alternative_id"] in {"A", "B"}
    ]
    actor_edge = next(edge for edge in edges if edge["relation"] == "HAS_ACTOR")
    return {
        "accepted_assertion_count": (
            sum(node["node_type"] in graph.ASSERTIONS for node in nodes)
            if checked["admitted"] is True
            else 0
        ),
        "accepted_snapshot_refused": accepted_refusal,
        "admitted": checked["admitted"],
        "alternative_ids": ["A", "B"],
        "alternative_identities_distinct": (
            len({node["record_cid"] for node in alternatives}) == 2
            and len({node["alternative_id"] for node in alternatives}) == 2
        ),
        "alternatives_share_identity": False,
        "ast_node_type": "ast" in graph.NODE_TYPES,
        "bundle_roundtrip": True,
        "candidate_assertion_count": sum(
            node["node_type"] in graph.ASSERTIONS for node in nodes
        ),
        "candidate_view": checked["view"],
        "cross_alternative_refused": crossed,
        "dangling_endpoint_refused": dangling,
        "duplicate_role_ordinal_refused": duplicate,
        "edge_count": len(edges),
        "endpoints_closed": all(
            edge["src_cid"] in {node["record_cid"] for node in nodes}
            and edge["dst_cid"] in {node["record_cid"] for node in nodes}
            for edge in edges
        ),
        "formalized": checked["formalized"],
        "graph_to_ast_embedded": False,
        "interpretation_companion_retained": (
            by_id["reading-a"]["interpretation_cids"] == [interpretation["record_cid"]]
            and by_id["reading-b"]["interpretation_cids"] == [interpretation["record_cid"]]
        ),
        "manifest_authority_false": all(manifest[key] is False for key in exchange._FLAGS),
        "multiplicity_ordinals": sorted(
            edge["ordinal"] for edge in edges if edge["relation"] == "HAS_OBJECT"
        ),
        "node_count": len(nodes),
        "review_node_type": "review" in graph.NODE_TYPES,
        "review_receipt_authenticated": False,
        "review_receipt_retained": actor_edge["evidence_cids"] == [review_receipt],
        "role_ordinals": [f"{relation}:{ordinal}" for relation, ordinal in roles],
        "source_companion_retained": (
            by_id["reading-a"]["source_occurrence_cids"] == [source["record_cid"]]
            and by_id["reading-b"]["source_occurrence_cids"] == [source["record_cid"]]
        ),
        "source_semantics_verified": checked["source_semantics_verified"],
    }
