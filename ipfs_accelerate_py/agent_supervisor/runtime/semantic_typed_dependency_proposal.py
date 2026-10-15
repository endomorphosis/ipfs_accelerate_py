"""Body-free advisory graph and nomination from recorded native relations.

This pure operation verifies immutable producer records, not current source or
task coverage. Every graph member remains proposal-only and nonmandatory. A
caller must independently verify current artifact/source bindings before and
after using the result; no output here authorizes capsule/source omission.
"""
from __future__ import annotations

from collections import deque
import hashlib
import os

from .semantic_selection_dependency_audit import _encoded, _literal_json
from .semantic_selection_relation_audit import (
    MAX_BUNDLE_BYTES,
    MAX_LINKS,
    MAX_REPORT_BYTES,
    MAX_SYMBOLS,
    _check_relation_inventory,
    _relation_audit_from_inventory,
    _verified_relation_inventory,
    audit_selected_capsule_relations,
)


SCHEMA = "supervisor-semantic-typed-dependency-proposal@1"
ADAPTER_VERSION = "native-typed-dependency-proposal@1"
TASK_CLOSURE_SCHEMA = "supervisor-semantic-typed-dependency-proposal@2"
TASK_CLOSURE_ADAPTER_VERSION = "native-typed-dependency-proposal@2"
MAX_SELECTED_SYMBOLS = 32
MAX_GRAPH_NODES = 2048
MAX_GRAPH_EDGES = 8192
_INVENTORY_REUSE_ENV = "IPFS_ACCELERATE_TYPED_INVENTORY_REUSE"


def _identifier(kind, value):
    return "typed-native-" + kind + ":sha256:" + hashlib.sha256(_encoded(value)).hexdigest()


def _index(view, name):
    import json
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import SortedPairIndex

    cid = getattr(view.root, name + "_index_cid")
    return SortedPairIndex.from_dict({**json.loads(view.get_block(cid)), "index_cid": cid})


def _link_record(link):
    """Retain typed scalar provenance and spans; omit arbitrary metadata bodies."""
    payload = link.to_dict()
    return {
        **{name: payload[name] for name in (
            "edge_id", "link_cid", "relation", "source_stable_id",
            "source_version_cid", "source_fact_cid", "target_kind",
            "target_stable_id", "target_version_cid", "target_fact_cid",
            "source_span", "extraction_method", "confidence", "extractor_version",
        )},
        "resolution": payload["metadata"].get("resolution"),
        "unresolved_target": payload["metadata"].get("unresolved_target"),
        "metadata_sha256": hashlib.sha256(_encoded(payload["metadata"])).hexdigest(),
    }


def nominate_typed_semantic_dependencies(
    *, bundle, root_cid: str, graph_root_id: str,
    seed_symbol_ids: tuple[str, ...], max_selected_symbols: int = 32,
    max_graph_nodes: int = 2048, max_graph_edges: int = 8192,
    max_report_bytes: int = MAX_REPORT_BYTES, graph_scope: str = "full_root",
) -> dict:
    """Nominate known outgoing symbol targets without closing unknown bindings.

    The default ``full_root`` scope retains every root symbol and typed link
    and the original @1 report. Explicit ``task_closure`` scope emits an @2
    report and materializes only the outgoing resolved-symbol closure of the
    independently supplied seeds, retaining every outgoing recorded relation,
    unresolved obligation and artifact endpoint of that closure. Both scopes
    verify the complete native root, retain full membership commitments, and
    report incoming boundary relations without automatically selecting them.
    Nomination follows every outgoing *resolved symbol* relation, including
    recorded test, fixture and proof relations. Incoming callers are reported
    separately and are not automatically nominated. Unknown state bindings,
    artifact coverage, extraction limitations and task relevance stay explicit.

    Bounds or invalid input raise before any successful partial proposal is
    returned. The existing relation audit owns complete native validation;
    this adapter consumes its verified root and never repairs native records.
    """
    if type(graph_scope) is not str or graph_scope not in {"full_root", "task_closure"}:
        raise ValueError("graph_scope must be the literal full_root or task_closure")
    scoped_graph = graph_scope == "task_closure"
    adapter_version = TASK_CLOSURE_ADAPTER_VERSION if scoped_graph else ADAPTER_VERSION
    for name, value, ceiling in (
        ("max_selected_symbols", max_selected_symbols, MAX_SELECTED_SYMBOLS),
        ("max_graph_nodes", max_graph_nodes, MAX_GRAPH_NODES),
        ("max_graph_edges", max_graph_edges, MAX_GRAPH_EDGES),
        ("max_report_bytes", max_report_bytes, MAX_REPORT_BYTES),
    ):
        if type(value) is not int or not 1 <= value <= ceiling:
            raise ValueError(f"{name} must be an integer from 1 to {ceiling}")
    if (type(graph_root_id) is not str or not graph_root_id
            or graph_root_id != graph_root_id.strip() or "\x00" in graph_root_id
            or len(graph_root_id.encode()) > 4096):
        raise ValueError("bounded literal graph root identity required")
    if (type(seed_symbol_ids) is not tuple or not seed_symbol_ids
            or len(seed_symbol_ids) > max_selected_symbols
            or any(type(sid) is not str or not sid or len(sid) > 16_384
                   for sid in seed_symbol_ids)
            or len(set(seed_symbol_ids)) != len(seed_symbol_ids)):
        raise ValueError("nonempty bounded distinct literal seed tuple required")

    # Default-on reuse is confined to this call. The opt-out performs the
    # original seed audit, independent typed accessor traversal, and nominee
    # audit; both paths require all complete native validators and bounds.
    reuse_inventory = os.environ.get(_INVENTORY_REUSE_ENV) != "0"
    if reuse_inventory:
        inventory = _verified_relation_inventory(
            bundle=bundle, root_cid=root_cid, selected_symbol_ids=list(seed_symbol_ids),
            max_symbols=MAX_SYMBOLS, max_links=MAX_LINKS,
            max_bundle_bytes=MAX_BUNDLE_BYTES, max_report_bytes=max_report_bytes,
        )
        seed_audit = _relation_audit_from_inventory(inventory, list(seed_symbol_ids))
    else:
        seed_audit = audit_selected_capsule_relations(
            bundle=bundle, root_cid=root_cid, selected_symbol_ids=list(seed_symbol_ids),
            max_symbols=MAX_SYMBOLS, max_links=MAX_LINKS,
            max_bundle_bytes=MAX_BUNDLE_BYTES, max_report_bytes=max_report_bytes,
        )
    from ..analysis.semantic_dependency_graph import (
        SemanticAuthority, SemanticDependencyGraph, SemanticEdge, SemanticEdgeKind,
        SemanticNode, SemanticNodeKind, SemanticProvenance, SemanticTrust,
    )

    if reuse_inventory:
        _check_relation_inventory(inventory)
        native_root = inventory.root
        full_symbol_ids = tuple(sorted(inventory.symbol_ids))
        facts, capsules = inventory.facts, inventory.capsules
        # Complete native validation already bound every link, source/target
        # fact and incoming/outgoing node adjacency. Keep its typed records;
        # printed audit bits never replace these native validation results.
        outgoing, links = inventory.outgoing, inventory.links
    else:
        from ipfs_datasets_py.logic.software_contracts.semantic_state.api import (
            open_semantic_state,
        )
        view = open_semantic_state(root_cid, bundle.get_block)
        native_root = view.root
        full_symbol_ids = tuple(sid for sid, _ in _index(view, "symbol_fact").pairs)
        facts = {sid: view.symbol_fact(sid) for sid in full_symbol_ids}
        capsules = {sid: view.capsule(sid) for sid in full_symbol_ids}
        links_by_id, outgoing = {}, {}
        # The opt-out deliberately retains every native typed accessor read.
        for sid in full_symbol_ids:
            outgoing[sid] = view.semantic_links_for_symbol(sid, direction="outgoing")
            for link in outgoing[sid]:
                links_by_id[link.edge_id] = link
        links = tuple(links_by_id[eid] for eid in sorted(links_by_id))
    if len(links) != seed_audit["full_link_count"]:
        raise ValueError("typed root link population changed after relation audit")

    visited = set()
    pending = deque(sorted(seed_symbol_ids))
    while pending:
        sid = pending.popleft()
        if sid in visited:
            continue
        if len(visited) >= max_selected_symbols:
            raise ValueError("known outgoing relation nomination exceeds symbol bound")
        visited.add(sid)
        for link in outgoing[sid]:
            if link.target_kind == "symbol" and link.target_stable_id not in visited:
                pending.append(link.target_stable_id)
    nominees = sorted(visited)
    if reuse_inventory:
        # A fresh selected ledger over the same private verified records.
        nominee_audit = _relation_audit_from_inventory(inventory, nominees)
    else:
        nominee_audit = audit_selected_capsule_relations(
            bundle=bundle, root_cid=root_cid, selected_symbol_ids=nominees,
            max_symbols=MAX_SYMBOLS, max_links=MAX_LINKS,
            max_bundle_bytes=MAX_BUNDLE_BYTES, max_report_bytes=max_report_bytes,
        )
    for name in ("full_relation_ledger_sha256", "full_capsule_identities_sha256",
                 "index_cids", "full_capsule_count", "full_link_count"):
        if nominee_audit[name] != seed_audit[name]:
            raise ValueError("native relation inventory changed during nomination")

    # Only materialization changes with scope. The closure and both audits
    # above are computed over the same complete verified root inventory. No
    # omitted root member or incoming relation becomes a coverage permission.
    materialized_symbol_ids = tuple(nominees) if scoped_graph else full_symbol_ids
    materialized_links = (tuple(link for link in links if link.source_stable_id in visited)
                          if scoped_graph else links)

    nodes = {}
    edges = []
    common = dict(root_id=graph_root_id, source_root_id=root_cid,
        provenance=SemanticProvenance.SOURCE, trust=SemanticTrust.VERIFIED,
        authority=SemanticAuthority.PROPOSAL_ONLY, version=adapter_version)

    def add_node(node_id, kind, record, provenance_id):
        node = SemanticNode(node_id=node_id, kind=kind, record=record,
            provenance_id=provenance_id, **common)
        previous = nodes.get(node_id)
        if previous is not None:
            if previous.to_dict() != node.to_dict():
                raise ValueError("conflicting native proposal node binding")
            return node_id
        if len(nodes) >= max_graph_nodes:
            raise ValueError("typed proposal graph exceeds node bound")
        nodes[node_id] = node
        return node_id

    def add_edge(source, target, link, role):
        if len(edges) >= max_graph_edges:
            raise ValueError("typed proposal graph exceeds edge bound")
        edges.append(SemanticEdge(source=source, target=target,
            kind=SemanticEdgeKind.DEPENDS_ON, mandatory=False,
            provenance_id=link.link_cid + ":" + role,
            record={"edge_id": link.edge_id, "link_cid": link.link_cid,
                    "relation": str(link.relation), "endpoint_role": role}, **common))

    # Keep complete body-free symbol bindings independently of graph scope.
    # Worker/source joins need the whole root's identities even when only a
    # task closure becomes graph nodes. Build each scalar record once.
    root_symbol_records = {}
    for sid in full_symbol_ids:
        fact, capsule = facts[sid], capsules[sid]
        root_symbol_records[sid] = {
            "native_symbol_id": sid, "native_symbol_kind": str(fact.symbol.kind),
            "qualified_name": fact.symbol.qualified_name,
            "source_module_path": fact.symbol.module_path, "source_cid": fact.source_cid,
            "version_cid": fact.version_cid, "fact_cid": fact.fact_cid,
            "capsule_cid": capsule.capsule_cid,
            "source_span": None if fact.span is None else fact.span.to_dict(),
            "confidence": fact.confidence, "extractor_name": fact.symbol.extractor_name,
            "extractor_version": fact.symbol.extractor_version,
        }
    symbol_node_ids = {}
    for sid in materialized_symbol_ids:
        fact = facts[sid]
        node_id = _identifier("symbol", {"root_cid": root_cid, "stable_symbol_id": sid})
        symbol_node_ids[sid] = add_node(node_id, SemanticNodeKind.SYMBOL,
                                      root_symbol_records[sid], fact.fact_cid)

    unknown_node_ids = {}
    for link in materialized_links:
        relation = str(link.relation)
        observation_id = _identifier("observation", {"root_cid": root_cid, "link_cid": link.link_cid})
        observation_kind = (SemanticNodeKind.CALL if relation == "calls" else
            SemanticNodeKind.DATA_FLOW if relation in {"reads_state", "writes_state"} else
            SemanticNodeKind.ANNOTATION)
        add_node(observation_id, observation_kind, _link_record(link), link.link_cid)

        if link.target_kind == "symbol":
            target_id = symbol_node_ids[link.target_stable_id]
        elif link.target_kind == "artifact":
            artifact = (inventory.artifact_facts[link.target_stable_id]
                        if reuse_inventory else view.artifact_fact(link.target_stable_id))
            target_id = _identifier("artifact", {"root_cid": root_cid,
                "native_artifact_id": artifact.artifact_id})
            add_node(target_id, SemanticNodeKind.RESOURCE, {
                "native_artifact_id": artifact.artifact_id, "fact_cid": artifact.fact_cid,
                "source_cid": artifact.source_cid, "path": artifact.artifact.path,
                "native_artifact_kind": artifact.artifact.kind,
                "confidence": str(artifact.confidence),
                "effective_coverage_verified": False,
            }, artifact.fact_cid)
        else:
            source_fact = facts[link.source_stable_id]
            recorded_target = link.metadata.get("unresolved_target")
            is_state = relation in {"reads_state", "writes_state"}
            is_global = is_state and recorded_target.startswith("global:")
            identity = ({"root_cid": root_cid, "source_module_path": source_fact.symbol.module_path,
                "binding_name": recorded_target[len("global:"):]} if is_global else
                {"root_cid": root_cid, "edge_id": link.edge_id})
            target_id = _identifier("global-binding" if is_global else "unknown-reference", identity)
            record = {
                "obligation_kind": "unresolved_state_binding" if is_state else "unresolved_reference",
                "binding_scope": "module_global" if is_global else "edge_specific_unknown",
                "recorded_target": recorded_target,
                "source_module_path": source_fact.symbol.module_path,
                "source_cid": source_fact.source_cid,
                "resolution": "unresolved", "binding_value_verified": False,
            }
            if is_global:
                record["binding_name"] = recorded_target[len("global:"):]
            else:
                record["edge_id"] = link.edge_id
            # A shared module binding can be observed from several symbols.
            # Its provenance is the source blob, not one arbitrarily first edge.
            add_node(target_id, SemanticNodeKind.OBLIGATION, record,
                (source_fact.source_cid or root_cid) if is_global else link.link_cid)
            unknown_node_ids[link.edge_id] = target_id
        # Observation nodes preserve recursive calls without a direct self-edge.
        add_edge(symbol_node_ids[link.source_stable_id], observation_id, link, "source_to_observation")
        add_edge(observation_id, target_id, link, "observation_to_target")

    graph = SemanticDependencyGraph(root_id=graph_root_id,
        nodes=tuple(nodes.values()), edges=tuple(edges))
    if any(node.authoritative for node in graph.nodes) or any(
            edge.authoritative or edge.mandatory for edge in graph.edges):
        raise ValueError("typed proposal cannot carry mandatory authority")

    def unresolved_rows(name):
        return [{**row, "obligation_node_id": unknown_node_ids[row["edge_id"]]}
                for row in nominee_audit[name]]

    result = {
        "schema": TASK_CLOSURE_SCHEMA if scoped_graph else SCHEMA,
        "adapter_version": adapter_version,
        "root_cid": root_cid, "semantic_root_cid": root_cid, "graph_root_id": graph_root_id,
        "repository_id": native_root.repository_id, "producer": native_root.producer.to_dict(),
        "input_basis": "exact_immutable_native_bundle",
        "coverage_scope": "recorded_root_relations_only_not_producer_or_task_completeness",
        "graph": graph.to_dict(), "full_symbol_ids": list(full_symbol_ids),
        "seed_symbol_ids": sorted(seed_symbol_ids), "nominee_symbol_ids": nominees,
        "nomination_available": True, "nomination_status": "experimental_only",
        "nomination_basis": "all_outgoing_recorded_resolved_symbol_relations",
        "incoming_relations_automatically_selected": False,
        "seed_relation_audit": seed_audit, "nominee_relation_audit": nominee_audit,
        "unresolved_state_binding_rows": unresolved_rows("unresolved_state_binding_rows"),
        "unresolved_other_reference_rows": unresolved_rows("unresolved_other_reference_rows"),
        "resolved_artifact_relation_rows": nominee_audit["resolved_artifact_relation_rows"],
        "selection_refusal_reasons": nominee_audit["selection_refusal_reasons"],
        "source_freshness_verified": False, "current_source_wrapper_required": True,
        "dependency_semantics_complete": False, "full_program_completeness_verified": False,
        "effective_task_coverage_verified": False, "test_task_impact_relevance_complete": False,
        "source_substitution_authority": False, "proof_authority": False,
        "execution_authority": False, "completion_authority": False,
        "acceptance_verified": False, "task_acceptance_verified": False,
        "runtime_activation": False, "activation_authority": False, "deployment_ready": False,
        "required_optional_policy_created": False, "source_bodies_in_report": False,
        "report_is_provider_input": False,
        "bounds": {"max_selected_symbols": max_selected_symbols,
            "max_graph_nodes": max_graph_nodes, "max_graph_edges": max_graph_edges,
            "max_report_bytes": max_report_bytes},
    }
    if scoped_graph:
        def population(values):
            return {"count": len(values),
                    "sha256": hashlib.sha256(_encoded(values)).hexdigest()}

        def link_bindings(selected_links):
            return sorted([[link.edge_id, link.link_cid] for link in selected_links])

        def artifact_bindings(selected_links):
            # Complete native verification already bound these identities to
            # the root artifact index and exact target fact versions. The
            # population concerns referenced endpoints, not unrelated indexed
            # artifacts that neither graph scope materializes.
            return [list(binding) for binding in sorted({
                (link.target_stable_id, link.target_fact_cid)
                for link in selected_links if link.target_kind == "artifact"
            })]

        full_symbol_records = [root_symbol_records[sid] for sid in full_symbol_ids]
        root_symbol_records_sha256 = hashlib.sha256(_encoded(full_symbol_records)).hexdigest()
        omitted_symbol_ids = sorted(set(full_symbol_ids) - visited)
        omitted_links = tuple(link for link in links if link.source_stable_id not in visited)
        full_artifact_bindings = artifact_bindings(links)
        materialized_artifact_bindings = artifact_bindings(materialized_links)
        materialized_artifact_set = {tuple(value) for value in materialized_artifact_bindings}
        omitted_artifact_bindings = [value for value in full_artifact_bindings
                                    if tuple(value) not in materialized_artifact_set]
        populations = {
            "symbols": {
                "full_root": population(list(full_symbol_ids)),
                "materialized": population(list(materialized_symbol_ids)),
                "omitted_root": population(omitted_symbol_ids),
            },
            "relations": {
                "full_root": population(link_bindings(links)),
                "materialized": population(link_bindings(materialized_links)),
                "omitted_root": population(link_bindings(omitted_links)),
            },
            "referenced_artifact_endpoints": {
                "full_root": population(full_artifact_bindings),
                "materialized": population(materialized_artifact_bindings),
                "omitted_root": population(omitted_artifact_bindings),
            },
        }
        boundary_rows = nominee_audit["incoming_omitted_relation_rows"]
        boundary_sha256 = hashlib.sha256(_encoded(boundary_rows)).hexdigest()
        graph_sha256 = hashlib.sha256(_encoded(result["graph"])).hexdigest()
        scope_commitment = {
            "schema": "supervisor-semantic-typed-graph-scope@1",
            "root_cid": root_cid, "graph_root_id": graph_root_id,
            "graph_scope": graph_scope,
            "materialization_basis": "all_outgoing_recorded_relations_of_explicit_seed_resolved_symbol_closure",
            "seed_symbol_ids": sorted(seed_symbol_ids),
            "materialized_symbol_ids": list(materialized_symbol_ids),
            "omitted_root_symbol_ids": omitted_symbol_ids,
            "index_cids": seed_audit["index_cids"],
            "root_symbol_records_count": len(full_symbol_records),
            "root_symbol_records_sha256": root_symbol_records_sha256,
            "full_capsule_identities_sha256": seed_audit["full_capsule_identities_sha256"],
            "full_relation_ledger_sha256": seed_audit["full_relation_ledger_sha256"],
            "graph_population": populations,
            "incoming_boundary_relation_count": len(boundary_rows),
            "incoming_boundary_relations_sha256": boundary_sha256,
            "graph_sha256": graph_sha256,
            "full_root_graph_materialized": not omitted_symbol_ids and not omitted_links,
        }
        result.update({
            "graph_scope": graph_scope,
            "root_symbol_records": full_symbol_records,
            "root_symbol_records_count": len(full_symbol_records),
            "root_symbol_records_sha256": root_symbol_records_sha256,
            "materialized_symbol_ids": list(materialized_symbol_ids),
            "omitted_root_symbol_ids": omitted_symbol_ids,
            "graph_population": populations,
            "incoming_boundary_relation_rows": boundary_rows,
            "incoming_boundary_relation_count": len(boundary_rows),
            "incoming_boundary_relations_sha256": boundary_sha256,
            "graph_sha256": graph_sha256,
            "full_root_graph_materialized": scope_commitment["full_root_graph_materialized"],
            "graph_scope_commitment": scope_commitment,
            "graph_scope_commitment_sha256": hashlib.sha256(_encoded(scope_commitment)).hexdigest(),
        })
    _literal_json(result)
    if len(_encoded(result)) > max_report_bytes:
        raise ValueError("typed dependency proposal exceeds report byte bound")
    if reuse_inventory:
        _check_relation_inventory(inventory)
    return result
