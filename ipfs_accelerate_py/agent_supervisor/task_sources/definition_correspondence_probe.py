"""Measure definition correspondence without promoting the unary alias.

The definition-use alias stays an unreviewed assumption. Declared qualifier
slots stay on a non-admitted graph, including null slots. Extra qualifier
text is refused instead of discarded. Unresolved alternatives stay on
separate branches. A mismatch returns None.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from .dependency_block_resolution import EXPECTED_DEFINITION_CORRESPONDENCE_PROBE

_FACT_SCHEMA = "legal-definition-correspondence-fact/v1"
_QUALIFIER_FACETS = ("conditions", "exceptions", "temporal")
_ALIAS_SENTENCE = (
    "A whole local predicate is aliased by an unreviewed assumption; "
    "no identity or legal implication is proved."
)
_REFUSAL = (
    "complete norm paragraph coverage differs; extra text cannot be discarded"
)


def measure_definition_correspondence_probe() -> dict[str, object] | None:
    """Return the fixed correspondence contract, or None when owners disagree."""

    try:
        measured = _measure()
    except Exception:
        return None
    if measured != EXPECTED_DEFINITION_CORRESPONDENCE_PROBE:
        return None
    return measured


def _datasets_root() -> Path:
    import ipfs_datasets_py

    return Path(ipfs_datasets_py.__file__).resolve().parents[1]


def _load_path(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError("correspondence fixture is unavailable")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load(name: str, filename: str):
    return _load_path(
        name,
        _datasets_root() / "tests/unit/logic/autoformal" / filename,
    )


def _arguments(row: dict) -> dict[str, object]:
    argument = row["predicate_argument"]
    anchors = row["source_anchor_paths"]
    if type(anchors) is not list:
        raise RuntimeError("argument source anchors are not a list")
    if not anchors and row["disposition"] != "introduced_unresolved":
        raise RuntimeError("unanchored argument was given a source disposition")
    if anchors and row["disposition"] != "lexical_hypothesis":
        raise RuntimeError("anchored argument lost its lexical disposition")
    codes = row["unresolved_codes"]
    if type(codes) is not list or not codes:
        raise RuntimeError("argument dropped its unresolved issue")
    return {
        "declared_role": argument["declared_role"],
        "typed_path": row["typed_path"],
        "source_anchor_count": len(anchors),
        "disposition": row["disposition"],
        "unresolved_codes": list(codes),
    }


def _qualifier_slots(report: dict) -> list[dict[str, object]]:
    rules = report["source_scope_report"]["scope_declaration"]["rules"]
    if type(rules) is not list or not rules:
        raise RuntimeError("source qualifier rules are missing")
    slots = []
    for rule in rules:
        qualifiers = rule["qualifiers"]
        if type(qualifiers) is not dict:
            raise RuntimeError("qualifier slots are not a closed object")
        if not set(_QUALIFIER_FACETS) <= set(qualifiers):
            raise RuntimeError("a declared qualifier facet was dropped")
        for facet, value in qualifiers.items():
            if value is not None:
                raise RuntimeError("probe fixture gained a qualifier value")
            slots.append(
                {
                    "rule_id": rule["rule_id"],
                    "facet": facet,
                    "value": None,
                }
            )
    return slots


def _unresolved(report: dict) -> list[dict[str, str]]:
    rows = []
    for origin, key in (
        ("original", "original_unresolved"),
        ("introduced", "introduced_unresolved"),
    ):
        issues = report[key]
        if type(issues) is not list or not issues:
            raise RuntimeError("unresolved alternatives are missing")
        for issue in issues:
            rows.append({"origin": origin, "code": issue["code"]})
    codes = [row["code"] for row in rows]
    if len(codes) != len(set(codes)):
        raise RuntimeError("unresolved alternative codes collided")
    return rows


def _seal(seal_record, payload: dict) -> dict:
    return seal_record({"schema": _FACT_SCHEMA, **payload})


def _node(graph, seal_record, stable_id: str, node_type: str, payload: dict, *,
          group: str | None, alternative: str | None, occurrences=None):
    sealed = _seal(seal_record, payload)
    node = graph.build_node(
        node_type=node_type,
        stable_id=stable_id,
        content_cid=sealed["record_cid"],
        alternative_group_id=group,
        alternative_id=alternative,
        source_occurrence_cids=occurrences,
    )
    return node, sealed["record_cid"]


def _edge(graph, src: dict, dst: dict, evidence: str) -> dict:
    return graph.build_edge(
        src_cid=src["record_cid"],
        dst_cid=dst["record_cid"],
        relation="CONTAINS",
        evidence_cids=[evidence],
    )


def _project(join_reports: list[dict], definition_use: dict) -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import corpus_graph as graph
    from ipfs_datasets_py.logic.autoformal.corpus_contracts import seal_record

    alias = definition_use["alias_record"]
    binding = definition_use["binding_record"]
    if alias["assumed_not_proved"] is not True:
        raise RuntimeError("definition alias was treated as proved")
    if binding["alias_semantics"] != (
        "explicit_unreviewed_unary_biconditional_assumption"
    ):
        raise RuntimeError("definition alias changed its unreviewed semantics")
    if alias["lean_compilation_verified"] is not False:
        raise RuntimeError("definition alias gained Lean verification")
    roles = binding["body_roles"]
    occurrences = binding["body_occurrence_cids"]
    if (
        type(roles) is not list
        or roles != ["definition_body/0000", "definition_body/0001"]
        or type(occurrences) is not list
        or len(occurrences) != len(roles)
    ):
        raise RuntimeError("definition body roles were not retained in order")
    use_nodes = definition_use["graph"]["nodes"]
    if any(node["node_type"] == "norm_assertion" for node in use_nodes):
        raise RuntimeError("definition alias gained a norm assertion")
    if not any(
        node["node_type"] == "expression"
        and node["source_occurrence_cids"] == occurrences
        for node in use_nodes
    ):
        raise RuntimeError("body occurrences are missing from the alias graph")
    if _ALIAS_SENTENCE not in definition_use["limitations"]:
        raise RuntimeError("alias limitation was removed")
    false_flags = {
        key: definition_use[key]
        for key in (
            "admitted",
            "formalized",
            "source_semantics_verified",
            "independently_reviewed",
            "proof_authority",
            "backend_executed",
            "lean_compilation_verified",
            "training_qualified",
            "accepted_view_eligible",
        )
    }
    if any(value is not False for value in false_flags.values()):
        raise RuntimeError("definition-use authority flag was set")

    if len(join_reports) != 2:
        raise RuntimeError("two unresolved alternatives are required")
    component = join_reports[0]["component_id"]
    alternatives = []
    branches = []
    for report in join_reports:
        if report["component_id"] != component:
            raise RuntimeError("alternatives do not share one component")
        if report["alternative_id"] in alternatives:
            raise RuntimeError("alternative identities collided")
        alternatives.append(report["alternative_id"])
        if (
            report["graph"] is not None
            or report["formal_norm_ast"] is not None
            or report["governing_definition_use"] is not None
        ):
            raise RuntimeError("join promoted an operative norm or graph")
        if any(report["authority"].values()):
            raise RuntimeError("join authority flag was set")
        if (
            report["source_meaning_selected"] is not False
            or report["native_build_executed"] is not False
            or report["admissions"] != 0
        ):
            raise RuntimeError("join selected meaning or executed a build")
        arguments = [
            _arguments(row)
            for row in report["provenance_ledger"]
            if "predicate_argument" in row
        ]
        if len(arguments) != 6:
            raise RuntimeError("predicate arguments were dropped")
        branches.append(
            {
                "alternative_id": report["alternative_id"],
                "arguments": arguments,
                "qualifiers": _qualifier_slots(report),
                "unresolved": _unresolved(report),
            }
        )
    if [row["declared_role"] for row in branches[0]["arguments"]] != [
        row["declared_role"] for row in branches[1]["arguments"]
    ]:
        raise RuntimeError("alternatives changed argument roles")
    if branches[0]["qualifiers"] != branches[1]["qualifiers"]:
        raise RuntimeError("alternatives changed qualifier slots")

    nodes = []
    edges = []
    alias_root, _alias_evidence = _node(
        graph,
        seal_record,
        "alias-root",
        "expression",
        {
            "kind": "definition_use_alias",
            "operative_norm": False,
            "assumed_not_proved": True,
        },
        group=None,
        alternative=None,
    )
    nodes.append(alias_root)
    for index, role in enumerate(roles):
        body, evidence = _node(
            graph,
            seal_record,
            "alias-body/" + role,
            "expression",
            {
                "kind": "body_role",
                "role": role,
                "source_occurrence_cid": occurrences[index],
            },
            group=None,
            alternative=None,
            occurrences=[occurrences[index]],
        )
        nodes.append(body)
        edges.append(_edge(graph, alias_root, body, evidence))
    for branch in branches:
        alternative = branch["alternative_id"]
        root, _root_evidence = _node(
            graph,
            seal_record,
            alternative + "/branch",
            "expression",
            {
                "kind": "alternative",
                "alternative_id": alternative,
                "operative_norm": False,
            },
            group=component,
            alternative=alternative,
        )
        nodes.append(root)
        for index, argument in enumerate(branch["arguments"]):
            child, evidence = _node(
                graph,
                seal_record,
                alternative + "/argument/" + str(index),
                "expression",
                {"kind": "argument", "alternative_id": alternative, **argument},
                group=component,
                alternative=alternative,
            )
            nodes.append(child)
            edges.append(_edge(graph, root, child, evidence))
        for slot in branch["qualifiers"]:
            child, evidence = _node(
                graph,
                seal_record,
                alternative + "/qualifier/" + slot["rule_id"] + "/" + slot["facet"],
                "expression",
                {
                    "kind": "qualifier",
                    "alternative_id": alternative,
                    "retained": True,
                    **slot,
                },
                group=component,
                alternative=alternative,
            )
            nodes.append(child)
            edges.append(_edge(graph, root, child, evidence))
        for issue in branch["unresolved"]:
            child, evidence = _node(
                graph,
                seal_record,
                alternative + "/issue/" + issue["origin"] + "/" + issue["code"],
                "issue",
                {"kind": "unresolved", "alternative_id": alternative, **issue},
                group=component,
                alternative=alternative,
            )
            nodes.append(child)
            edges.append(_edge(graph, root, child, evidence))

    snapshot = graph.build_graph_snapshot(nodes, edges)
    if (
        snapshot["view"] != "candidate"
        or snapshot["admitted"] is not False
        or snapshot["formalized"] is not False
        or snapshot["source_semantics_verified"] is not False
    ):
        raise RuntimeError("correspondence graph was admitted")
    if any(node["node_type"] == "norm_assertion" for node in snapshot["nodes"]):
        raise RuntimeError("correspondence graph asserted a norm")
    by_cid = {node["record_cid"]: node for node in snapshot["nodes"]}
    for edge in snapshot["edges"]:
        src = by_cid[edge["src_cid"]]
        dst = by_cid[edge["dst_cid"]]
        if src["alternative_id"] != dst["alternative_id"]:
            raise RuntimeError("unresolved alternatives share an edge")
        if src["alternative_group_id"] != dst["alternative_group_id"]:
            raise RuntimeError("an edge left its alternative group")
    stable_ids = {node["stable_id"] for node in snapshot["nodes"]}
    expected_ids = {"alias-root"}
    expected_ids.update("alias-body/" + role for role in roles)
    for branch in branches:
        alternative = branch["alternative_id"]
        expected_ids.add(alternative + "/branch")
        expected_ids.update(
            alternative + "/argument/" + str(index)
            for index in range(len(branch["arguments"]))
        )
        expected_ids.update(
            alternative + "/qualifier/" + slot["rule_id"] + "/" + slot["facet"]
            for slot in branch["qualifiers"]
        )
        expected_ids.update(
            alternative + "/issue/" + issue["origin"] + "/" + issue["code"]
            for issue in branch["unresolved"]
        )
    if stable_ids != expected_ids:
        raise RuntimeError("graph correspondence dropped or invented a node")
    first = branches[0]
    return {
        "alias_counted_as_operative_norm": False,
        "alias_assumed_not_proved": True,
        "alias_semantics": (
            "explicit_unreviewed_unary_biconditional_assumption"
        ),
        "alias_norm_assertion_count": 0,
        "body_roles": list(roles),
        "body_roles_on_graph": True,
        "alternative_ids": sorted(alternatives),
        "alternatives_share_edge": False,
        "argument_count_per_branch": 6,
        "argument_roles": [row["declared_role"] for row in first["arguments"]],
        "source_anchor_counts": [
            row["source_anchor_count"] for row in first["arguments"]
        ],
        "arguments_missing_from_graph": 0,
        "qualifier_facets": list(_QUALIFIER_FACETS),
        "qualifier_slots_per_branch": len(first["qualifiers"]),
        "qualifier_values_null": True,
        "qualifier_slots_missing_from_graph": 0,
        "unresolved_codes": sorted(issue["code"] for issue in first["unresolved"]),
        "unresolved_on_graph": True,
        "graph_node_count": len(snapshot["nodes"]),
        "graph_edge_count": len(snapshot["edges"]),
        "graph_norm_assertion_count": 0,
        "graph_view": "candidate",
        "graph_admitted": False,
        "graph_formalized": False,
        "graph_source_semantics_verified": False,
        "native_build_executed": False,
        "source_identity_asserted": False,
    }


def _bridge_refusal() -> str:
    from ipfs_datasets_py.logic.autoformal import (
        legal_source_definition_norm_bridge as bridge,
    )

    saved_path = list(sys.path)
    try:
        fixture = _load_path(
            "lir_definition_norm_binding_fixture",
            _datasets_root()
            / "scripts/ops/legal_ir/check_legal_definition_norm_binding.py",
        )
        report = bridge.prepare_definition_norm(fixture.fixture_request())
        if report["profile"] != bridge.PROFILE:
            raise RuntimeError("definition-norm profile changed")
        if any(report[key] is not False for key in bridge.FALSE_FLAGS):
            raise RuntimeError("definition-norm authority flag was set")
        if "alias_semantics" in report or "binding_record" in report:
            raise RuntimeError("unary norm bridge absorbed the definition-use alias")
        try:
            bridge.prepare_definition_norm(
                fixture.fixture_request(norm_suffix=" only with approval.")
            )
        except ValueError as error:
            message = str(error)
        else:
            raise RuntimeError("qualifier text was discarded into the unary norm")
    finally:
        sys.path[:] = saved_path
    if message != _REFUSAL:
        raise RuntimeError("qualifier refusal changed")
    return message


def _measure() -> dict[str, object]:
    from ipfs_datasets_py.logic.autoformal import corpus_definition_use_v2 as use_owner
    from ipfs_datasets_py.logic.autoformal import (
        legal_source_typed_norm_join_v1 as join_owner,
    )

    use_test = _load("lir_definition_use_fixture", "test_corpus_definition_use_v2.py")
    definition_use = use_test.prepare(use_owner, use_test.fixture(use_owner))
    join_test = _load("lir_join_fixture", "test_legal_source_typed_norm_join_v1.py")
    first = join_test.case()
    second = join_test.case()
    second["join_request"]["alternative_id"] = "hypothesis-b"
    reports = [join_test.prepare(first), join_test.prepare(second)]
    if join_owner.prepare_source_typed_norm_join_v1 is None:
        raise RuntimeError("join owner is missing")
    measured = _project(reports, definition_use)
    measured["qualifier_text_discarded"] = False
    measured["qualifier_refusal"] = _bridge_refusal()
    return measured
