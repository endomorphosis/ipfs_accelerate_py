"""Scoped selection from one fully verified immutable native producer bundle.

Only the capsule index lookup is shared within this call. Each capsule keeps
native byte/CID/schema/identity checks, and every potentially fitting candidate
retains the existing native freshness and admission checks.
"""
from __future__ import annotations

import json
import re
from types import MappingProxyType


def _encoded(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False,
                      separators=(",", ":"), allow_nan=False).encode()


def immutable_bundle_capsule_reader(bundle, *, root_cid):
    """Return a call-local reader; never memoize an injected mutable reader."""
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        SemanticStateBundle, SemanticCapsule, SortedPairIndex, verify_block_bytes,
    )

    if (type(bundle) is not SemanticStateBundle or type(bundle.blocks) is not MappingProxyType
            or bundle.root.root_cid != root_cid):
        raise ValueError("capsule selection requires the exact immutable native bundle/root")
    # Capture the immutable values, not a mutable or subclassed block callback.
    blocks, index_cid = bundle.blocks, bundle.root.capsule_index_cid

    def identity(cid):
        raw = blocks[cid]
        verify_block_bytes(cid, raw)
        value = json.loads(raw)
        if type(value) is not dict:
            raise ValueError("native capsule selection requires a structured identity")
        return value

    index = SortedPairIndex.from_dict({**identity(index_cid), "index_cid": index_cid})
    by_symbol = dict(index.pairs)

    def capsule(stable_symbol_id):
        if type(stable_symbol_id) is not str or not stable_symbol_id:
            raise ValueError("stable_symbol_id must be a nonempty string")
        cid = by_symbol[stable_symbol_id]
        value = SemanticCapsule.from_dict({**identity(cid), "capsule_cid": cid})
        if value.stable_symbol_id != stable_symbol_id:
            raise ValueError("native capsule stable_symbol_id differs from root index")
        return value

    return capsule


def select_worker_capsules(*, bundle, view, provider, symbols, sources, required,
                          worker_query, worker_capsule_limit, worker_max_bytes,
                          worker_ranked_symbol_ids=None):
    from ..semantic_state.capsules import admit_capsule

    read_capsule = immutable_bundle_capsule_reader(bundle, root_cid=view.root.root_cid)
    candidates = list(symbols)
    if worker_query:
        terms = set(re.findall(r"[a-z][a-z0-9_]+", worker_query.lower()))
        candidates.sort(key=lambda symbol: (
            -len(terms.intersection(re.findall(r"[a-z][a-z0-9_]+",
                (symbol.qualified_name + " " + symbol.module_path).lower()))),
            symbol.module_path, symbol.qualified_name, symbol.stable_id,
        ))
    if worker_ranked_symbol_ids is not None:
        known = {symbol.stable_id for symbol in candidates}
        if (not worker_query or type(worker_ranked_symbol_ids) not in (list, tuple)
                or len(worker_ranked_symbol_ids) > 20
                or any(type(value) is not str or value not in known for value in worker_ranked_symbol_ids)
                or len(set(worker_ranked_symbol_ids)) != len(worker_ranked_symbol_ids)):
            raise ValueError("retrieval ranking requires unique current native symbol identities")
        ranks = {value: index for index, value in enumerate(worker_ranked_symbol_ids)}
        # Stable sort retains the existing lexical order after the retrieved
        # symbols. Ranking changes priority, never the source or admission scope.
        candidates.sort(key=lambda symbol: ranks.get(symbol.stable_id, len(ranks)))
    capsules, admissions, selected_symbols = [], [], []
    used = 8192 + sum(len(sources[name]) for name in required)
    for symbol in candidates:
        if worker_query and len(capsules) >= worker_capsule_limit:
            break
        capsule = read_capsule(symbol.stable_id)
        payload = capsule.to_dict()
        capsule_bytes = len(_encoded(payload))
        # An admission has nonnegative size. This strict lower bound can reject
        # only a candidate which the previous full-size calculation rejected.
        if worker_query and used + capsule_bytes + 1600 > worker_max_bytes:
            continue
        freshness = provider.assess_capsule_freshness(capsule, current_state=view)
        admission = admit_capsule(capsule, semantic_state_root_cid=view.root.root_cid,
            assessment=freshness, force_raw_source=symbol.module_path in required)
        if worker_query:
            size = capsule_bytes + len(_encoded(admission.to_dict())) + 1600
            if used + size > worker_max_bytes:
                continue
            used += size
            selected_symbols.append({"path": symbol.module_path, "qualified_name": symbol.qualified_name,
                                     "stable_symbol_id": symbol.stable_id})
        capsules.append(payload)
        admissions.append(admission)
    return capsules, admissions, selected_symbols


def resolve_worker_retrieval(*, repository, selection, task_id, worker_query,
                             manifest, program_paths, symbols, retained_sources=None):
    """Resolve a verified, current retrieval result to native capsule identities.

    The retrieval owner independently replays the query and its complete source
    AST. A path/name match alone cannot nominate a capsule from another source
    generation. This does not qualify the embedding model or retrieval quality.
    """
    from .code_retrieval_context import load_code_retrieval_context, replay_code_retrieval_context_for_audit

    if (type(selection) is not dict or set(selection) != {"artifact", "sha256"}
            or type(selection["artifact"]) is not str
            or type(selection["sha256"]) is not str
            or re.fullmatch(r"[0-9a-f]{64}", selection["sha256"]) is None):
        raise ValueError("exact retrieval artifact and digest required for capsule ranking")
    arguments = dict(repository=repository, artifact=selection["artifact"],
        expected_sha256=selection["sha256"], task_id=task_id)
    context = json.loads(load_code_retrieval_context(**arguments) if retained_sources is None else
        replay_code_retrieval_context_for_audit(**arguments,
            retained_sources={name: retained_sources[name] for name in program_paths}))
    expected_sources = {name: manifest[name]["sha256"] for name in program_paths}
    if (context["status"] != ("current" if retained_sources is None else "historical")
            or context["query_text"] != worker_query
            or context["source_sha256"] != expected_sources):
        raise ValueError("capsule retrieval ranking differs from current source/query scope")
    by_name = {}
    for symbol in symbols:
        by_name.setdefault((symbol.module_path, symbol.qualified_name), []).append(symbol)
    ranked = []
    for hit in context["hits"]:
        matches = by_name.get((hit["path"], hit["symbol"]), [])
        if len(matches) != 1:
            raise ValueError("retrieval hit does not identify one current native symbol")
        symbol = matches[0]
        if (symbol.source_cid != manifest[hit["path"]]["source_cid"]
                or symbol.span is None or symbol.span.path != hit["path"]
                or symbol.span.start_line != hit["line_start"]
                or symbol.span.end_line != hit["line_end"]
                or symbol.stable_id in ranked):
            raise ValueError("retrieval hit differs from native symbol source/span identity")
        ranked.append(symbol.stable_id)
    return ranked
