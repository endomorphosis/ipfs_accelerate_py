"""Measure one candidate graph against the closed node and role contract.

The snapshot is not a statutory population. Equal spellings stay separate
nodes. Person and event types are refused. A mismatch returns None.
"""

from __future__ import annotations

from .dependency_block_resolution import EXPECTED_SCOPED_GRAPH_PROBE


def measure_scoped_graph_probe() -> dict[str, object] | None:
    """Return the fixed graph contract, or None when the live graph disagrees."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_SCOPED_GRAPH_PROBE:
        return None
    return measured


def _cid(seal_record, label: str) -> str:
    return seal_record({"schema": "scoped-graph-probe/v1", "label": label})["record_cid"]


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import corpus_graph as graph
    from ipfs_datasets_py.logic.autoformal.corpus_contracts import (
        CorpusContractError,
        seal_record,
    )

    def cid(label: str) -> str:
        return _cid(seal_record, label)

    def node(kind: str, label: str, **kwargs):
        if kind in graph.ASSERTIONS:
            kwargs.setdefault("source_occurrence_cids", [cid("source-" + label)])
            kwargs.setdefault("interpretation_cids", [cid("interpretation-" + label)])
        return graph.build_node(
            node_type=kind,
            stable_id=label,
            content_cid=cid("content-" + kind + "-" + label),
            **kwargs,
        )

    def edge(src, dst, relation: str, ordinal=None):
        return graph.build_edge(
            src_cid=src["record_cid"],
            dst_cid=dst["record_cid"],
            relation=relation,
            ordinal=ordinal,
            evidence_cids=[cid(relation + "-" + src["stable_id"] + "-" + dst["stable_id"] + "-" + str(ordinal))],
        )

    source = node("source_version", "edition")
    occurrence = node("occurrence", "mention")
    interpretation = node("interpretation", "reading")
    sense_a = node("term_sense", "Secretary", scope_cid=cid("title-one"))
    sense_b = node("term_sense", "Secretary", scope_cid=cid("title-two"))
    category = node("entity", "Secretary")
    other = node("expression", "object-two")
    definition = node("definition_assertion", "definition")
    norm = node(
        "norm_assertion",
        "rule",
        source_occurrence_cids=[occurrence["record_cid"]],
        interpretation_cids=[interpretation["record_cid"]],
    )
    record = graph.build_graph_snapshot(
        [
            source,
            occurrence,
            interpretation,
            sense_a,
            sense_b,
            category,
            other,
            definition,
            norm,
        ],
        [
            edge(occurrence, source, "HAS_SOURCE"),
            edge(occurrence, sense_a, "DENOTES"),
            edge(definition, sense_b, "DEFINES"),
            edge(norm, category, "HAS_ACTOR", 0),
            edge(norm, sense_a, "HAS_ACTION", 0),
            edge(norm, sense_b, "HAS_OBJECT", 0),
            edge(norm, other, "HAS_OBJECT", 1),
        ],
    )
    checked = graph.validate_graph_snapshot(record)
    nodes = checked["nodes"]
    edges = checked["edges"]
    secretary_senses = [
        item for item in nodes
        if item["node_type"] == "term_sense" and item["stable_id"] == "Secretary"
    ]
    secretary_entities = [
        item for item in nodes
        if item["node_type"] == "entity" and item["stable_id"] == "Secretary"
    ]
    norm_nodes = [item for item in nodes if item["node_type"] == "norm_assertion"]
    if len(norm_nodes) != 1:
        raise RuntimeError("norm assertion was not retained")
    norm_node = norm_nodes[0]
    roles = sorted(
        (item["relation"], item["ordinal"])
        for item in edges
        if item["relation"] in graph.ROLES
    )
    try:
        node("person", "ground-person")
    except CorpusContractError as exc:
        person_refusal = str(exc)
    else:
        raise RuntimeError("person node was accepted")
    try:
        node("event", "actual-event")
    except CorpusContractError as exc:
        event_refusal = str(exc)
    else:
        raise RuntimeError("event node was accepted")
    try:
        graph.build_graph_snapshot(
            [norm, category, sense_a],
            [
                edge(norm, category, "HAS_ACTOR", 0),
                edge(norm, sense_a, "HAS_ACTOR", 0),
            ],
        )
    except CorpusContractError as exc:
        duplicate_refusal = str(exc)
    else:
        raise RuntimeError("duplicate role ordinal was accepted")
    return {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "equal_spelling_term_senses": len(secretary_senses),
        "equal_spelling_entities": len(secretary_entities),
        "distinct_sense_ids": len({item["record_cid"] for item in secretary_senses}),
        "senses_merged": len({item["record_cid"] for item in secretary_senses}) != 2,
        "entity_remains_entity": (
            len(secretary_entities) == 1
            and secretary_entities[0]["node_type"] == "entity"
        ),
        "person_node_type": "person" in graph.NODE_TYPES,
        "event_node_type": "event" in graph.NODE_TYPES,
        "person_nodes": sum(item["node_type"] == "person" for item in nodes),
        "event_nodes": sum(item["node_type"] == "event" for item in nodes),
        "person_node_refused": person_refusal,
        "event_node_refused": event_refusal,
        "companions_retained": (
            norm_node["source_occurrence_cids"] == [occurrence["record_cid"]]
            and norm_node["interpretation_cids"] == [interpretation["record_cid"]]
        ),
        "source_version_retained": any(item["node_type"] == "source_version" for item in nodes),
        "interpretation_retained": any(item["node_type"] == "interpretation" for item in nodes),
        "ordered_role_ordinals": ",".join(f"{relation}:{ordinal}" for relation, ordinal in roles),
        "duplicate_role_refused": duplicate_refusal,
        "action_edge_count": sum(item["relation"] == "HAS_ACTION" for item in edges),
        "view": checked["view"],
        "graph_admitted": checked["admitted"],
        "graph_formalized": checked["formalized"],
        "graph_source_semantics_verified": checked["source_semantics_verified"],
    }
