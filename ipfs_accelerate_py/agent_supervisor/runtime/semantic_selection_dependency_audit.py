"""Bounded diagnostics over dependencies declared by verified native capsules.

The caller supplies its already verified complete scoped capsule inventory.
Revalidating its native records does not verify current source or establish that
declared edges represent every dependency of the program. This helper has no
storage, provider, selector, source-substitution, or deployment capability.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections import deque

SCHEMA = "supervisor-semantic-selection-dependency-audit@1"
MAX_NODES = 1024
MAX_EDGES = 8192
MAX_CAPSULE_BYTES = 2_000_000
MAX_TOTAL_CAPSULE_BYTES = 16_000_000


def _literal_json(value):
    """Reject mutable/custom container interpretations and excessive nesting."""
    stack = [(value, 0)]
    visited = 0
    while stack:
        item, depth = stack.pop()
        visited += 1
        if depth > 40 or visited > 100_000:
            raise ValueError("capsule structure exceeds audit bounds")
        if type(item) is dict:
            if any(type(key) is not str for key in item):
                raise ValueError("capsule keys must be literal strings")
            stack.extend((child, depth + 1) for child in item.values())
        elif type(item) is list:
            stack.extend((child, depth + 1) for child in item)
        elif type(item) in (str, bool, int) or item is None:
            continue
        elif type(item) is float and math.isfinite(item):
            continue
        else:
            raise ValueError("capsules must contain literal finite JSON values")


def _encoded(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False).encode("utf-8")


def _digest(value):
    return hashlib.sha256(_encoded(value)).hexdigest()


def audit_selected_capsule_dependencies(
    *, full_capsules: list[dict], selected_symbol_ids: list[str],
    max_nodes: int = MAX_NODES, max_edges: int = MAX_EDGES,
) -> dict:
    """Report selection coverage of the reachable declared stable-ID graph.

    Known closure includes the selected seeds. A selected dependency is a seed
    reached as an edge target, including a seed reached through a cycle.
    Missing dependency targets remain unresolved. The closure Boolean describes
    only the supplied records and declared edges, never semantic completeness.
    """
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        SemanticCapsule,
    )

    if type(max_nodes) is not int or not 1 <= max_nodes <= MAX_NODES:
        raise ValueError("max_nodes must be an integer from 1 to 1024")
    if type(max_edges) is not int or not 1 <= max_edges <= MAX_EDGES:
        raise ValueError("max_edges must be an integer from 1 to 8192")
    if type(full_capsules) is not list or type(selected_symbol_ids) is not list:
        raise ValueError("capsule inventory and selected IDs must be literal lists")
    if len(full_capsules) > max_nodes or len(selected_symbol_ids) > max_nodes:
        raise ValueError("capsule inventory or selection exceeds node bound")

    graph, identities = {}, {}
    total_bytes = 0
    edge_count = 0
    for payload in full_capsules:
        if type(payload) is not dict:
            raise ValueError("each native capsule must be a literal dictionary")
        _literal_json(payload)
        raw = _encoded(payload)
        total_bytes += len(raw)
        if len(raw) > MAX_CAPSULE_BYTES or total_bytes > MAX_TOTAL_CAPSULE_BYTES:
            raise ValueError("capsule inventory exceeds audit byte bounds")
        dependencies = payload.get("dependency_stable_ids")
        if type(dependencies) is not list:
            raise ValueError("declared dependency IDs must be a literal list")
        if any(type(value) is not str for value in dependencies):
            raise ValueError("declared dependency IDs must be literal strings")
        if len(dependencies) != len(set(dependencies)):
            raise ValueError("declared dependency IDs must be unique")
        edge_count += len(dependencies)
        if edge_count > max_edges:
            raise ValueError("declared graph exceeds edge bound")
        # Native dependency stable IDs are bounded text, not necessarily CIDs.
        # Unresolved names such as global:helper_operation remain graph targets
        # verbatim. Native parsing enforces trimmed NFC printable text and CID
        # integrity of the capsule's own stable/version/content identities.
        capsule = SemanticCapsule.from_dict(payload)
        stable_id = capsule.stable_symbol_id
        if stable_id in graph:
            raise ValueError("native inventory contains duplicate stable symbol IDs")
        graph[stable_id] = tuple(sorted(dependencies))
        identities[stable_id] = capsule.capsule_cid

    for seed in selected_symbol_ids:
        if type(seed) is not str:
            raise ValueError("selected IDs must be literal strings")
        if seed not in graph:
            raise ValueError("selected seed is foreign to the supplied inventory")
    selected = set(selected_symbol_ids)
    if len(selected) != len(selected_symbol_ids):
        raise ValueError("selected seed IDs must be unique")

    known, targets, unresolved = set(), set(), set()
    pending = deque(sorted(selected))
    reachable_edges = 0
    while pending:
        stable_id = pending.popleft()
        if stable_id in known:
            continue
        known.add(stable_id)
        for target in graph[stable_id]:
            reachable_edges += 1
            targets.add(target)
            if target in graph:
                if target not in known:
                    pending.append(target)
            else:
                unresolved.add(target)
        if len(known | unresolved) > max_nodes:
            raise ValueError("reachable declared graph exceeds node bound")

    # Kahn's algorithm detects cycles without recursion or exposing source.
    incoming = dict.fromkeys(sorted(known), 0)
    for stable_id in sorted(known):
        for target in graph[stable_id]:
            if target in known:
                incoming[target] += 1
    ready = deque(stable_id for stable_id, count in incoming.items() if count == 0)
    removed = 0
    while ready:
        stable_id = ready.popleft()
        removed += 1
        for target in graph[stable_id]:
            if target in incoming:
                incoming[target] -= 1
                if incoming[target] == 0:
                    ready.append(target)

    missing = (targets & known) - selected
    return {
        "schema": SCHEMA,
        "input_basis": "caller_supplied_native_capsules_revalidated",
        "edge_basis": "declared_dependency_stable_ids",
        "selected_symbol_ids": sorted(selected),
        "known_transitive_closure_symbol_ids": sorted(known),
        "selected_dependency_symbol_ids": sorted(targets & selected),
        "missing_selected_dependency_symbol_ids": sorted(missing),
        "unresolved_dependency_symbol_ids": sorted(unresolved),
        "declared_graph_closure_selected": not missing and not unresolved,
        "cycle_present": removed != len(known),
        "cycle_analysis_scope": "reachable_known_declared_graph",
        "full_capsule_count": len(graph),
        "selected_capsule_count": len(selected),
        "declared_edge_count": edge_count,
        "reachable_known_count": len(known),
        "reachable_edge_count": reachable_edges,
        "unresolved_dependency_count": len(unresolved),
        "full_capsule_identities_sha256": _digest(sorted(identities.items())),
        "declared_graph_sha256": _digest([
            [stable_id, identities[stable_id], list(graph[stable_id])]
            for stable_id in sorted(graph)
        ]),
        "selection_sha256": _digest(sorted(selected)),
        "source_freshness_verified": False,
        "dependency_semantics_complete": False,
        "full_program_completeness_verified": False,
        "source_substitution_authority": False,
        "execution_authority": False,
        "proof_authority": False,
        "completion_authority": False,
        "deployment_ready": False,
    }
