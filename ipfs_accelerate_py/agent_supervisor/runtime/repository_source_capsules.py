"""Bounded, advisory capsule delivery from one already joined native state.

The datasets producer owns every CID and capsule. This module adds only bounded
lexical selection and source-read handles; it opens no catalog and does not scan,
train, prove, execute target code, or infer that omitted symbols are irrelevant.
"""
from __future__ import annotations

import json
import re

MAX_SYMBOLS = 512
MAX_FACTS = 4096
MAX_SOURCE_BYTES = 4 * 1024 * 1024
MAX_STATE_BYTES = 4 * 1024 * 1024
MAX_BUNDLE_BYTES = 16 * 1024 * 1024
MAX_RESULTS = 32
MAX_RESPONSE_BYTES = 65536
MAX_QUERY_BYTES = 4096


class RepositoryCapsuleError(ValueError):
    """An optional presentation failure, not a native index invalidation."""

    def __init__(self, code):
        self.code = code
        super().__init__(code)


def _require(value, code):
    if not value:
        raise RepositoryCapsuleError(code)


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode("utf-8")


def response_budget(max_bytes):
    _require(type(max_bytes) is int and 1 <= max_bytes <= MAX_RESPONSE_BYTES,
             "capsule_invalid_byte_limit")


def query_arguments(query, limit, max_bytes):
    response_budget(max_bytes)
    _require(type(limit) is int and 1 <= limit <= MAX_RESULTS, "capsule_invalid_limit")
    _require(type(query) is str and 0 < len(query) <= MAX_QUERY_BYTES and query.strip(),
             "capsule_invalid_query")
    try:
        _require(len(query.encode("utf-8")) <= MAX_QUERY_BYTES, "capsule_invalid_query")
    except UnicodeError as error:
        raise RepositoryCapsuleError("capsule_invalid_query") from error


def expansion_arguments(semantic_root_cid, stable_symbol_ids, max_bytes):
    from ipfs_datasets_py.logic.software_contracts.content import validate_cid

    response_budget(max_bytes)
    _require(type(stable_symbol_ids) is list and 1 <= len(stable_symbol_ids) <= MAX_RESULTS,
             "capsule_invalid_symbol_ids")
    _require(all(type(value) is str and 0 < len(value) <= 256 for value in stable_symbol_ids)
             and len(set(stable_symbol_ids)) == len(stable_symbol_ids), "capsule_invalid_symbol_ids")
    _require(type(semantic_root_cid) is str and 0 < len(semantic_root_cid) <= 256,
             "capsule_invalid_root")
    try:
        validate_cid(semantic_root_cid)
        for value in stable_symbol_ids:
            validate_cid(value)
    except ValueError as error:
        raise RepositoryCapsuleError("capsule_invalid_identity") from error


class RepositoryCapsules:
    """One bounded native bundle, reused only under its owner's source fences."""

    def __init__(self, native, manifest, *, checkpoint):
        from ..semantic_state.datasets_adapter import IpfsDatasetsSemanticStateProvider
        from .semantic_capsule_selection import immutable_bundle_capsule_reader

        state = native.semantic_state
        _require(len(state.symbols) <= MAX_SYMBOLS and len(state.edges) <= MAX_FACTS
                 and len(state.artifacts) <= MAX_FACTS, "capsule_population_limit")
        _require(sum(entry.size_bytes or 0 for entry in native.snapshot.entries) <= MAX_SOURCE_BYTES,
                 "capsule_source_limit")
        _require(len(_encoded(state.to_dict())) <= MAX_STATE_BYTES, "capsule_state_limit")
        checkpoint()
        self.provider = IpfsDatasetsSemanticStateProvider()
        self.bundle = self.provider.build_semantic_state(state)
        checkpoint()
        _require(sum(len(raw) for raw in self.bundle.blocks.values()) <= MAX_BUNDLE_BYTES,
                 "capsule_bundle_limit")
        self.view = self.provider.view_semantic_state_bundle(self.bundle)
        root = self.view.root
        _require(root.repository_id == state.repository_id
                 and root.producer.repository_state_cid == state.state_cid, "capsule_native_identity_differs")
        self.read_capsule = immutable_bundle_capsule_reader(self.bundle, root_cid=root.root_cid)
        self.symbols = {symbol.stable_id: symbol for symbol in state.symbols}
        self.manifest = manifest
        self.identity = dict(semantic_root_cid=root.root_cid, capsule_index_cid=root.capsule_index_cid,
            semantic_state_cid=state.state_cid, source_tree_sha256=manifest["source_tree_sha256"])
        checkpoint()

    def _base(self, schema):
        return dict(schema=schema, **self.identity, authority="structural_only",
            semantic_equivalence_claimed=False, proof_authority=False,
            execution_authority=False, completion_authority=False)

    def _admitted(self, symbol_id):
        from ..semantic_state.capsules import admit_capsule

        _require(symbol_id in self.symbols, "capsule_symbol_unavailable")
        symbol = self.symbols[symbol_id]
        capsule = self.read_capsule(symbol_id)
        entry = self.manifest["entries"].get(symbol.module_path)
        _require(entry is not None and entry["kind"] == "file"
                 and capsule.source_slice_path == symbol.module_path
                 and capsule.source_cid == symbol.source_cid
                 and capsule.version_cid == symbol.version_cid, "capsule_source_identity_differs")
        # The producer treats missing optional projections conservatively, but
        # corruption of a present block must never look like optional absence.
        projection = capsule.relevant_binding_projection_cid
        if projection is not None and projection in self.bundle.blocks:
            self.view.get_block(projection)
        assessment = self.provider.assess_capsule_freshness(capsule, current_state=self.view)
        admission = admit_capsule(capsule, semantic_state_root_cid=self.view.root.root_cid,
                                 assessment=assessment)
        _require(admission.freshness == "fresh", "capsule_freshness_unavailable")
        return symbol, capsule, admission, entry

    def query(self, query, *, limit, max_bytes, checkpoint):
        # Same lexical vocabulary and deterministic tie order as the existing
        # worker capsule selector. Unmatched symbols are explicitly omitted.
        terms = set(re.findall(r"[a-z][a-z0-9_]+", query.lower()))
        ranked = []
        for symbol in self.symbols.values():
            score = len(terms.intersection(re.findall(r"[a-z][a-z0-9_]+",
                (symbol.qualified_name + " " + symbol.module_path).lower())))
            if score:
                ranked.append((-score, symbol.module_path, symbol.qualified_name, symbol.stable_id))
        ranked.sort()
        response = self._base("repository-source-context@1")
        response.update(query=query, ranking="lexical_symbol_path", population_count=len(self.symbols),
            matching_count=len(ranked), selected_count=0, omitted_count=len(self.symbols),
            truncated=bool(ranked), candidates=[])
        _require(len(_encoded(response)) <= max_bytes, "capsule_response_limit")
        for _score, _path, _name, symbol_id in ranked:
            checkpoint()
            if len(response["candidates"]) >= limit:
                break
            symbol, capsule, admission, entry = self._admitted(symbol_id)
            candidate = dict(symbol_id=symbol_id, path=symbol.module_path, name=symbol.qualified_name,
                kind=str(symbol.kind), source_sha256=entry["original_sha256"], source_cid=symbol.source_cid,
                capsule_cid=capsule.capsule_cid, confidence=admission.ref.confidence,
                raw_source_required=admission.requires_raw_source)
            previous = (response["selected_count"], response["omitted_count"], response["truncated"])
            response["candidates"].append(candidate)
            response["selected_count"] += 1
            response["omitted_count"] -= 1
            response["truncated"] = response["selected_count"] < len(ranked)
            if len(_encoded(response)) > max_bytes:
                response["candidates"].pop()
                response["selected_count"], response["omitted_count"], response["truncated"] = previous
        checkpoint()
        return response

    def expand(self, *, semantic_root_cid, stable_symbol_ids, max_bytes, checkpoint):
        _require(semantic_root_cid == self.view.root.root_cid, "capsule_root_differs")
        response = self._base("repository-source-capsule-expansion@1")
        response.update(requested_stable_symbol_ids=list(stable_symbol_ids), capsules=[], admissions=[],
            raw_source_fetch_required={}, raw_source_policy="Read exact current source and verify SHA256 before editing.")
        _require(len(_encoded(response)) <= max_bytes, "capsule_response_limit")
        for symbol_id in stable_symbol_ids:
            checkpoint()
            symbol, capsule, admission, entry = self._admitted(symbol_id)
            response["capsules"].append(capsule.to_dict())
            response["admissions"].append(admission.to_dict())
            response["raw_source_fetch_required"][symbol.module_path] = dict(
                source_sha256=entry["original_sha256"], source_cid=symbol.source_cid)
            _require(len(_encoded(response)) <= max_bytes, "capsule_response_limit")
        checkpoint()
        return response
