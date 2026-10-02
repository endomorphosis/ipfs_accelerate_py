"""Current-source typed discovery over existing native catalog/cache owners.

This local plane deliberately has no fabricated DQP/Quack release identity.
It reuses native AST, evidence-row and dependency types. Reconstructed rows are
inert evidence references; positive cache eligibility still requires its owner.
"""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import hashlib
import importlib
import json
import math
from pathlib import Path, PurePosixPath
import time

from ipfs_datasets_py.duckdb_control.intent_codebase_catalog import IntentCodebaseCatalog
from ipfs_datasets_py.knowledge_graphs.adapters.duckdb_code_evidence import EvidenceNodeRow, EvidenceEdgeRow
from ipfs_datasets_py.logic.software_contracts.codebase_integer_profile import IntegerOffsetContract
from ipfs_datasets_py.logic.software_contracts.codebase_scan_policy_live import verify_policy_current
from ipfs_datasets_py.logic.software_contracts.codebase_semantic_manifest import load_codebase_semantic_manifest
from ipfs_datasets_py.logic.software_contracts.content import canonical_dag_json_bytes, cid_for_structured
from ipfs_datasets_py.logic.software_contracts.duckdb_impact import ImpactGraph, ImpactBudget, closure
from ..proof.finite_checked_cache import FiniteCheckedCache
from ..proof.formal_verification_contracts import ProofReceipt

SCHEMA = "repository-local-code-evidence@1"
QUERY_SCHEMA = "repository-code-evidence-query@1"
MAX_BYTES = 4 * 1024 * 1024
PRODUCERS = (__name__,
    "ipfs_datasets_py.duckdb_control.intent_codebase_catalog",
    "ipfs_datasets_py.logic.software_contracts.codebase_semantic_manifest",
    "ipfs_datasets_py.logic.software_contracts.codebase_scan_policy_live",
    "ipfs_datasets_py.logic.software_contracts.codebase_ir",
    "ipfs_datasets_py.logic.software_contracts.duckdb_ast_store",
    "ipfs_datasets_py.logic.software_contracts.duckdb_impact",
    "ipfs_datasets_py.knowledge_graphs.adapters.duckdb_code_evidence",
    "ipfs_accelerate_py.agent_supervisor.proof.finite_checked_cache",
    "ipfs_accelerate_py.agent_supervisor.planning.behavioral_codebase_match")
FALSE = dict(proof_authority=False, source_equivalence_proved=False,
             execution_authority=False, completion_authority=False,
             omission_authority=False, dqp_release_verified=False)


class RepositoryCodeEvidenceError(ValueError):
    """A current owner, bounded selection or consumed identity differs."""


def _require(condition, message):
    if not condition:
        raise RepositoryCodeEvidenceError(message)


def _copy(value):
    raw = canonical_dag_json_bytes(value)
    _require(len(raw) <= MAX_BYTES, "evidence query exceeds aggregate byte bound")
    return json.loads(raw)


def _pins():
    return {name: hashlib.sha256(Path(importlib.import_module(name).__file__).read_bytes()).hexdigest()
            for name in PRODUCERS}


def _path(value):
    _require(type(value) is str and 0 < len(value.encode()) <= 1024
             and str(PurePosixPath(value)) == value and not value.startswith("/")
             and ".." not in PurePosixPath(value).parts and "\\" not in value
             and not any(ord(c) < 32 for c in value), "exact bounded relative source path required")


@dataclass(frozen=True)
class RelevantUnitsQuery:
    paths: tuple[str, ...] = ()
    page_size: int = 16

    def __post_init__(self):
        _require(type(self.paths) is tuple and len(self.paths) <= 64
                 and len(set(self.paths)) == len(self.paths), "bounded unique exact path tuple required")
        for path in self.paths:
            _path(path)
        _require(type(self.page_size) is int and 1 <= self.page_size <= 64, "page size must be 1..64")

    def to_dict(self):
        return dict(kind="relevant_units", paths=list(self.paths), page_size=self.page_size)


@dataclass(frozen=True)
class UnitEvidenceQuery:
    path: str
    inputs: tuple[int, ...]
    page_size: int = 32

    def __post_init__(self):
        _path(self.path)
        _require(type(self.inputs) is tuple and 1 <= len(self.inputs) <= 32
                 and all(type(v) is int and abs(v) <= 2**31 for v in self.inputs)
                 and tuple(sorted(set(self.inputs))) == self.inputs, "explicit sorted nonempty bounded integer domain required")
        _require(type(self.page_size) is int and 1 <= self.page_size <= 64, "page size must be 1..64")

    def to_dict(self):
        return dict(kind="unit_evidence", path=self.path, inputs=list(self.inputs), page_size=self.page_size)


@dataclass(frozen=True)
class LocalCodeEvidencePlane:
    source_revision: str
    nodes: tuple[EvidenceNodeRow, ...]
    edges: tuple[EvidenceEdgeRow, ...]

    def to_dict(self):
        return dict(schema=SCHEMA, source_revision=self.source_revision,
            nodes=[row.to_dict() for row in self.nodes], edges=[row.to_dict() for row in self.edges],
            scope="reconstructed_native_references_not_live_capabilities", **FALSE)

    def reverse_dependencies(self, node_id):
        _require(node_id in {row.node_id for row in self.nodes}, "unknown typed evidence node")
        graph = ImpactGraph(source_revision=self.source_revision)
        for edge in self.edges:
            graph.add(edge.source, edge.target, edge.kind)
        result = closure(graph, [node_id], direction="reverse",
            budget=ImpactBudget(max_depth=4, max_rows=512, max_seconds=1))
        _require(not result.truncated, "dependency frontier is incomplete")
        return {**result.to_dict(), "historical_only": True, **FALSE}

    def project_dqp(self, *args, **kwargs):
        raise RepositoryCodeEvidenceError("no independently verified DQP release owner is available for this local profile")


class RepositoryCodeEvidence:
    """No new database, source head, proof head or warmed eligibility cache."""
    def __init__(self, catalog, checked_cache):
        _require(type(catalog) is IntentCodebaseCatalog and type(checked_cache) is FiniteCheckedCache
                 and checked_cache.artifacts is catalog.index.artifacts,
                 "exact native discovery and checked cache with the same artifact owner required")
        self.catalog, self.checked_cache, self.index = catalog, checked_cache, catalog.index

    @contextmanager
    def _current(self, repository, expected_head, semantic_manifest_cid, resources):
        values = dict(resources)
        timeout = values.pop("timeout_seconds", 120)
        _require(type(timeout) in (int, float) and math.isfinite(timeout) and 0 < timeout <= 300,
                 "bounded query deadline required")
        _require(set(values) <= {"scheduler", "parent_lease", "cancel_event"}, "unknown resource controls")
        cancel = values.get("cancel_event")
        _require(cancel is None or callable(getattr(cancel, "is_set", None)), "cancellation must provide is_set")
        deadline, pins = time.monotonic() + timeout, _pins()
        def remaining():
            from ipfs_datasets_py.optimizers.logic_theorem_optimizer.resource_scheduler import LeaseCancelledError, LeaseTimeoutError
            if cancel is not None and cancel.is_set():
                raise LeaseCancelledError("evidence query cancelled")
            left = deadline-time.monotonic()
            if left <= 0:
                raise LeaseTimeoutError("evidence query deadline expired")
            return left
        manifest = load_codebase_semantic_manifest(self.index, semantic_manifest_cid)
        _require(manifest["source_head"] == expected_head.to_dict(), "semantic manifest source head differs")
        def observe():
            verify_policy_current(self.index, repository, expected_head=expected_head,
                receipt_cid=manifest["policy_receipt_cid"], timeout_seconds=remaining(), **values)
            _require(_pins() == pins, "evidence query producer changed")
        observe()
        structural = self.index.load(expected_head.manifest_cid)
        _require(structural.ast_revision_id == expected_head.ast_revision_id,
                 "native AST and complete head revisions differ")
        _require(0 < len(manifest["units"]) <= 256, "bounded nonempty complete inventory required")
        # Each discovery record covers the entire manifest. Query one exact
        # inventory path, then retain every matching manifest nomination.
        anchor = manifest["units"][0]["path"]
        discovery = self.catalog.lookup(repository, expected_head=expected_head,
            policy_receipt_cid=manifest["policy_receipt_cid"], path=anchor,
            timeout_seconds=remaining(), **values)
        nominations = [row for row in discovery["records"]
                       if row["record"]["semantic_manifest_cid"] == semantic_manifest_cid]
        _require(nominations, "exact full-manifest discovery nomination is missing")
        commitments = dict(source_head=expected_head.to_dict(), semantic_manifest_cid=semantic_manifest_cid,
            policy_receipt_cid=manifest["policy_receipt_cid"], producers=pins,
            discovery_record_cids=sorted(row["record_cid"] for row in nominations))
        yield manifest, structural, commitments, remaining, values
        current = self.catalog.lookup(repository, expected_head=expected_head,
            policy_receipt_cid=manifest["policy_receipt_cid"], path=anchor,
            timeout_seconds=remaining(), **values)
        _require(current == discovery, "discovery population changed during query")
        observe()

    @staticmethod
    def _page(query, rows, commitments, cursor):
        rows = _copy(rows)
        identities = [cid_for_structured(row) for row in rows]
        _require(len(set(identities)) == len(identities), "duplicate query rows")
        root = cid_for_structured(dict(query=query.to_dict(), commitments=commitments, row_cids=identities))
        offset = 0
        if cursor is not None:
            _require(type(cursor) is dict and set(cursor) == {"schema", "root", "query", "offset", "prefix_cid"}
                     and cursor["schema"] == "repository-evidence-cursor@1"
                     and cursor["root"] == root and cursor["query"] == query.to_dict(), "cursor belongs to another complete query root")
            offset = cursor["offset"]
            _require(type(offset) is int and 0 < offset < len(rows) and offset % query.page_size == 0
                     and cursor["prefix_cid"] == cid_for_structured(identities[:offset]), "cursor frontier or consumed prefix differs")
        stop = min(len(rows), offset+query.page_size)
        continuation = None if stop == len(rows) else dict(schema="repository-evidence-cursor@1",
            root=root, query=query.to_dict(), offset=stop, prefix_cid=cid_for_structured(identities[:stop]))
        result = dict(schema=QUERY_SCHEMA, query=query.to_dict(), root=root, commitments=commitments,
            rows=rows[offset:stop], consumed_row_cids=identities[offset:stop], offset=offset,
            total_rows=len(rows), next_cursor=continuation, page_is_last=continuation is None,
            selection_complete=offset == 0 and continuation is None,
            cursor_prefix_is_not_proof_of_consumption=True, **FALSE)
        result["result_cid"] = cid_for_structured(result)
        return _copy(result)

    def relevant_units(self, *, query, repository, expected_head, semantic_manifest_cid, cursor=None, **resources):
        _require(type(query) is RelevantUnitsQuery, "exact relevant-unit query required")
        with self._current(repository, expected_head, semantic_manifest_cid, resources) as (manifest, structural, commitments, _, __):
            requested = set(query.paths)
            _require(requested <= {r["path"] for r in manifest["units"]}, "query contains paths outside complete source inventory")
            rows = []
            for row in manifest["units"]:
                if requested and row["path"] not in requested:
                    continue
                ast = self.index.lookup(structural, row["path"])
                rows.append(dict(path=row["path"], logical_unit_id=row["logical_unit_id"], version_cid=row["version_cid"],
                    source_cid=row["source_cid"], ast_cid=None if ast is None else ast.ast_cid,
                    ast_revision=structural.ast_revision_id,
                    ast_summary=None if ast is None else ast.to_supervisor_blob_summary(),
                    model_status=row["model_status"], declared_contract_cid=row["declared_contract_cid"],
                    relevance="explicit_path_selection" if requested else "complete_inventory",
                    semantic_relevance_inferred=False, **FALSE))
            result = self._page(query, sorted(rows, key=lambda r: r["path"]), commitments, cursor)
        return result

    def unit_evidence(self, *, query, repository, expected_head, semantic_manifest_cid,
            tool_policy, cursor=None, **resources):
        _require(type(query) is UnitEvidenceQuery, "exact unit-evidence query required")
        tool_policy = _copy(tool_policy)
        with self._current(repository, expected_head, semantic_manifest_cid, resources) as (manifest, structural, commitments, remaining, controls):
            selected = [r for r in manifest["units"] if r["path"] == query.path]
            _require(len(selected) == 1, "exact source unit is unavailable")
            unit = selected[0]
            ast = self.index.lookup(structural, query.path)
            looked = None
            if unit["model_status"] == "source_bound_model":
                contract = IntegerOffsetContract.from_dict(unit["declared_contract"])
                looked = self.checked_cache.lookup(owner_inputs=dict(index=self.index, repository=repository,
                    expected_head=expected_head, contract=contract, inputs=list(query.inputs),
                    tool_policy=tool_policy, **controls), timeout_seconds=remaining())
            refs = [] if unit["source_cid"] is None else self.checked_cache.dependents("source", unit["source_cid"], limit=64)
            commitments.update(unit_version_cid=unit["version_cid"], tool_policy= _copy(tool_policy),
                reverse_references=refs, cache=None if looked is None else {k: looked.get(k) for k in
                    ("status", "record_cid", "request_key", "positive_reuse_eligible", "scope")})
            nodes, edges = {}, {}
            revision = structural.ast_revision_id
            def node(kind, key):
                identity = cid_for_structured(dict(kind=kind, key=key))
                nodes[identity] = EvidenceNodeRow(identity, kind, key, "native_owner_reconstruction",
                    False, revision, tree_id=expected_head.snapshot_cid, symbol=query.path)
                return identity
            def edge(source, target, kind):
                identity = cid_for_structured(dict(source=source,target=target,kind=kind))
                edges[identity] = EvidenceEdgeRow(identity,kind,source,target,"native_owner_reconstruction",False,revision)
            uid = node("source_unit", unit["version_cid"])
            for kind, key in (("source", unit["source_cid"]), ("snapshot", expected_head.snapshot_cid),
                    ("semantic_manifest", semantic_manifest_cid), ("contract", unit["declared_contract_cid"]),
                    ("ast", None if ast is None else ast.ast_cid)):
                if key is not None:
                    edge(uid, node(kind,key), "depends_on")
            for ref in refs:
                edge(node("historical_cache_reference",ref["record_cid"]), node("source",unit["source_cid"]), "historically_depended_on")
            if looked is not None and looked.get("evidence") is not None:
                record = looked["evidence"]
                rid = node("checked_cache_" + looked["status"], looked["record_cid"])
                edge(rid,uid,"for_source_unit")
                for kind, key in record["dependencies"]:
                    edge(rid,node("dependency:"+kind,key),"depends_on")
                if record["receipt"] is not None:
                    receipt = ProofReceipt.from_dict(record["receipt"])
                    edge(rid,node("table_proof_receipt",receipt.receipt_id),"has_scoped_receipt")
                    commitments["receipt_id"] = receipt.receipt_id
                commitments["bundle_cid"] = record["bundle_cid"]
                commitments["canonical_key"] = record["correspondence"]["canonical_key"]
            plane = LocalCodeEvidencePlane(revision, tuple(nodes[k] for k in sorted(nodes)), tuple(edges[k] for k in sorted(edges)))
            rows = [dict(kind="node", value=n.to_dict()) for n in plane.nodes]
            rows += [dict(kind="edge",value=e.to_dict()) for e in plane.edges]
            _require(len(rows) <= 512, "typed evidence projection exceeds row bound")
            result = self._page(query, rows, commitments, cursor)
            result.update(unit_disposition=unit["model_status"], cache_status=None if looked is None else looked["status"],
                positive_reuse_eligible=False if looked is None else looked["positive_reuse_eligible"],
                positive_scope=None if looked is None else looked.get("scope"),
                fresh_native_observation=False if looked is None else looked.get("fresh_native_observation",False),
                reverse_source_dependencies=None if unit["source_cid"] is None else
                    plane.reverse_dependencies(node("source", unit["source_cid"])),
                scope="typed_native_references_with_separate_table_proof_eligibility")
            result["result_cid"] = cid_for_structured({k:v for k,v in result.items() if k != "result_cid"})
            _copy(result)
        return result

    def resolve_intent(self, *, repository, expected_head, semantic_manifest_cid, intent_document,
            source_text, output, tool_policy, **resources):
        """Complete two-clause supported profile; unsupported clauses stay visible.

        The freshly invoked matcher is the fact owner. This entry point takes
        no saved match or injected facts and does not accept a paging cursor.
        """
        from ..planning.behavioral_codebase_match import match_behavioral_intent
        matched = match_behavioral_intent(catalog=self.catalog, checked_cache=self.checked_cache,
            repository=repository, expected_head=expected_head, semantic_manifest_cid=semantic_manifest_cid,
            intent_document=intent_document, source_text=source_text, output=output, tool_policy=tool_policy, **resources)
        _require(len(matched["requirement_results"]) <= 16, "intent result exceeds closed query population")
        result = dict(schema="repository-intent-evidence-query@1", source_head=expected_head.to_dict(),
            semantic_manifest_cid=semantic_manifest_cid, match=matched,
            residual_obligations=[row for row in matched["requirement_results"]
                                  if row["statement_id"] in matched["residual_requirements"]],
            consumed_match_cid=matched["match_cid"], complete_requirement_ids=sorted(
                row["statement_id"] for row in matched["requirement_results"]),
            consumed_record_commitments=dict(checked_cache=matched["checked_cache"],
                discovery_record_cids=[] if matched["discovery"] is None else
                    sorted(row["record_cid"] for row in matched["discovery"]["records"])),
            selection_complete=True, reduced_task_population_authorized=False, **FALSE)
        result["result_cid"] = cid_for_structured(result)
        return _copy(result)


__all__ = ["RepositoryCodeEvidence", "RelevantUnitsQuery", "UnitEvidenceQuery",
           "LocalCodeEvidencePlane", "RepositoryCodeEvidenceError", "SCHEMA", "QUERY_SCHEMA"]
