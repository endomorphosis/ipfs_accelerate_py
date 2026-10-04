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
                          worker_query, worker_capsule_limit, worker_max_bytes):
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
