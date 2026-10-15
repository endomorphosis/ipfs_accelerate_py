"""Bind advisory typed-link nominations to a current complete native artifact.

The existing native dispatch verifier owns source and producer admission. This
adapter observes that boundary before and after projection; it cannot turn
source observations into a task policy, proof, or permission to omit evidence.
"""

from __future__ import annotations

import hashlib
import hmac
import importlib
import json
import math
import os
import secrets
import stat
import sys
import re
from pathlib import Path, PurePosixPath
from types import MappingProxyType

from . import semantic_router_translation as codec
from .semantic_selection_relation_audit import MAX_BLOCKS, MAX_BUNDLE_BYTES
from .semantic_typed_dependency_proposal import nominate_typed_semantic_dependencies

SCHEMA = "supervisor-source-bound-typed-context-proposal@1"
WORKER_SCHEMA = "supervisor-worker-source-bound-typed-context-proposal@1"
SCOPED_SCHEMA = "supervisor-source-bound-typed-context-proposal@2"
SCOPED_WORKER_SCHEMA = "supervisor-worker-source-bound-typed-context-proposal@2"
_SHA = re.compile(r"[0-9a-f]{64}\Z")


class _TypedVerificationEligibilityError(ValueError):
    """Optional code/platform capture failed; retained evidence still must close."""


def _capture_blocks(repository, artifact, semantic_root_cid):
    """Read every retained block afresh, including omitted/unreferenced rows."""
    relative = PurePosixPath(artifact).parent / "blocks"
    descriptor = os.open(repository, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        for part in relative.parts:
            child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW,
                            dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        before = os.fstat(descriptor)
        names = []
        with os.scandir(descriptor) as entries:
            for entry in entries:
                if len(names) >= MAX_BLOCKS:
                    raise ValueError("typed source context block population exceeds bound")
                names.append(entry.name)
        if semantic_root_cid not in names:
            raise ValueError("typed source context root block is missing")
        blocks, total = {}, 0
        for name in sorted(names):
            raw = codec._read(repository, str(relative / name))
            total += len(raw)
            if total > MAX_BUNDLE_BYTES:
                raise ValueError("typed source context bundle exceeds byte bound")
            blocks[name] = raw
        after = os.fstat(descriptor)
        if any(getattr(before, key) != getattr(after, key) for key in (
                "st_dev", "st_ino", "st_mtime_ns", "st_ctime_ns")):
            raise ValueError("typed source context block directory changed during capture")
    finally:
        os.close(descriptor)
    return MappingProxyType(blocks)


def _load_bundle(repository, artifact, semantic_root_cid):
    """Construct the native bundle only on the full-validation path."""
    from ipfs_datasets_py.logic.software_contracts.semantic_state.models import (
        SemanticStateBundle, SemanticStateRoot,
    )

    blocks = _capture_blocks(repository, artifact, semantic_root_cid)
    root = SemanticStateRoot.from_dict({
        **codec._parse(blocks[semantic_root_cid].decode()), "root_cid": semantic_root_cid,
    })
    return SemanticStateBundle(root=root, blocks=blocks)


def project_source_bound_typed_links(
    *, repository: Path, artifact: str, expected_sha256: str, task_id: str,
    expected_semantic_root_cid: str, graph_root_id: str,
    seed_symbol_ids: tuple[str, ...], max_selected_symbols: int = 32,
    max_graph_nodes: int = 2048, max_graph_edges: int = 8192,
    max_report_bytes: int = 2_000_000, graph_scope: str = "full_root",
) -> dict:
    """Preserve the existing complete-parent source proposal boundary.

    Query-selected, role and decision parents still refuse. This default
    returns the existing schema and fields without a worker projection audit.
    """
    return _project_source_bound_typed_links(
        repository=repository, artifact=artifact, expected_sha256=expected_sha256,
        task_id=task_id, expected_semantic_root_cid=expected_semantic_root_cid,
        graph_root_id=graph_root_id, seed_symbol_ids=seed_symbol_ids,
        max_selected_symbols=max_selected_symbols, max_graph_nodes=max_graph_nodes,
        max_graph_edges=max_graph_edges, max_report_bytes=max_report_bytes,
        allow_worker_projection=False,
        graph_scope=graph_scope,
    )


def project_worker_source_bound_typed_links(
    *, repository: Path, artifact: str, expected_sha256: str, task_id: str,
    expected_semantic_root_cid: str, graph_root_id: str,
    seed_symbol_ids: tuple[str, ...], max_selected_symbols: int = 32,
    max_graph_nodes: int = 2048, max_graph_edges: int = 8192,
    max_report_bytes: int = 2_000_000, graph_scope: str = "full_root",
) -> dict:
    """Read a full native root alongside its authentic delivered worker subset.

    The complete source frontier and producer root are already retained by
    the original preparation. Reuse those immutable blocks without another
    scan, another worker parent, or any source/capsule substitution. The actual
    parent inventory is returned separately from the complete graph inventory.
    Query projection counts, order, native fact bindings, and raw-fetch records
    must match exactly. Role/decision views and malformed projections refuse.
    """
    return _project_source_bound_typed_links(
        repository=repository, artifact=artifact, expected_sha256=expected_sha256,
        task_id=task_id, expected_semantic_root_cid=expected_semantic_root_cid,
        graph_root_id=graph_root_id, seed_symbol_ids=seed_symbol_ids,
        max_selected_symbols=max_selected_symbols, max_graph_nodes=max_graph_nodes,
        max_graph_edges=max_graph_edges, max_report_bytes=max_report_bytes,
        allow_worker_projection=True,
        graph_scope=graph_scope,
    )


def _proposal_symbol_records(proposal, *, graph_scope):
    """Separate complete authenticated identity metadata from graph membership."""
    if graph_scope == "full_root":
        return [node["record"] for node in proposal["graph"]["nodes"]
                if "native_symbol_id" in node["record"]]
    if (proposal.get("schema") not in {
            "supervisor-semantic-typed-dependency-proposal@2", SCOPED_SCHEMA, SCOPED_WORKER_SCHEMA}
            or proposal.get("graph_scope") != "task_closure"
            or proposal.get("adapter_version") != "native-typed-dependency-proposal@2"):
        raise ValueError("typed source context requires the explicit scoped proposal version")
    graph = proposal.get("graph")
    if (type(graph) is not dict or type(graph.get("nodes")) is not list
            or len(graph["nodes"]) > 2048
            or any(type(node) is not dict or type(node.get("record")) is not dict
                   for node in graph["nodes"])):
        raise ValueError("typed source context requires bounded typed graph records")
    graph_records = [node["record"] for node in graph["nodes"]
                     if "native_symbol_id" in node["record"]]
    records = proposal.get("root_symbol_records")
    if (type(records) is not list or not 1 <= len(records) <= 1024
            or any(type(record) is not dict for record in records)
            or type(proposal.get("root_symbol_records_count")) is not int
            or proposal["root_symbol_records_count"] != len(records)
            or hashlib.sha256(codec._json(records).encode()).hexdigest()
                != proposal.get("root_symbol_records_sha256")):
        raise ValueError("typed source context complete native symbol metadata differs")
    full_ids = proposal.get("full_symbol_ids")
    materialized = proposal.get("materialized_symbol_ids")
    omitted = proposal.get("omitted_root_symbol_ids")
    record_ids = [record.get("native_symbol_id") for record in records]
    if (type(full_ids) is not list or any(type(sid) is not str for sid in record_ids)
            or record_ids != full_ids or record_ids != sorted(set(record_ids))
            or type(materialized) is not list or materialized != proposal.get("nominee_symbol_ids")
            or any(type(sid) is not str for sid in materialized)
            or materialized != sorted(set(materialized)) or not set(materialized) <= set(full_ids)
            or type(omitted) is not list or omitted != sorted(set(full_ids) - set(materialized))):
        raise ValueError("typed source context scoped and complete symbol populations differ")
    by_symbol = {record["native_symbol_id"]: record for record in records}
    if (any(type(record.get("native_symbol_id")) is not str for record in graph_records)
            or sorted(record["native_symbol_id"] for record in graph_records) != materialized
            or any(record != by_symbol[record["native_symbol_id"]] for record in graph_records)):
        raise ValueError("typed source context scoped graph differs from native root metadata")
    return records


def _worker_projection_binding(*, semantic, capsule_index_cid, symbol_records, parent_bytes):
    """Join the delivered inventory to exact immutable native graph facts."""
    rows = semantic.get("capsules")
    if type(rows) is not list or len(rows) > 1024:
        raise ValueError("typed worker context requires a bounded literal capsule inventory")
    by_symbol, native_capsules = {}, set()
    for record in symbol_records:
        sid, cid = record.get("native_symbol_id"), record.get("capsule_cid")
        if (type(sid) is not str or not sid or type(cid) is not str or not cid
                or sid in by_symbol or cid in native_capsules):
            raise ValueError("typed worker context native symbol/capsule mapping differs")
        by_symbol[sid] = record
        native_capsules.add(cid)
    inventory, seen_symbols, seen_capsules = [], set(), set()
    expected_selected = []
    for row in rows:
        if type(row) is not dict:
            raise ValueError("typed worker context requires literal native capsule records")
        sid, cid = row.get("stable_symbol_id"), row.get("capsule_cid")
        if (type(sid) is not str or type(cid) is not str or sid not in by_symbol
                or sid in seen_symbols or cid in seen_capsules
                or by_symbol[sid]["capsule_cid"] != cid):
            raise ValueError("typed worker context delivered capsule identity differs from native root")
        seen_symbols.add(sid)
        seen_capsules.add(cid)
        inventory.append({"stable_symbol_id": sid, "capsule_cid": cid})
        fact = by_symbol[sid]
        path, qualified_name = fact.get("source_module_path"), fact.get("qualified_name")
        if type(path) is not str or type(qualified_name) is not str:
            raise ValueError("typed worker context selected symbol metadata differs")
        expected_selected.append({"path": path, "qualified_name": qualified_name,
                                  "stable_symbol_id": sid})

    projected = "worker_projection" in semantic
    full_count, selected_count = len(by_symbol), len(inventory)
    projection = semantic.get("worker_projection")
    if projected:
        fields = {
            "schema", "selector", "query", "max_capsules", "max_bytes",
            "full_capsule_count", "selected_capsule_count", "omitted_capsule_count",
            "selected_symbols", "capsule_index_cid", "raw_source_fetch_required",
            "omission_reason", "raw_required_caveat", "semantic_equivalence_claimed",
            "completion_authority",
        }
        if (type(projection) is not dict
                or set(projection) not in (fields, fields | {"exact_byte_budget_repacked"})
                or projection.get("schema") != "supervisor-semantic-worker-projection@1"
                or projection.get("selector") != "lexical-query-symbol-name@1"
                or type(projection.get("query")) is not str or not projection["query"]
                or len(projection["query"].encode()) > 8192
                or projection.get("semantic_equivalence_claimed") is not False
                or projection.get("completion_authority") is not False):
            raise ValueError("typed worker context requires the exact native query projection")
        if ("exact_byte_budget_repacked" in projection
                and projection["exact_byte_budget_repacked"] is not True):
            raise ValueError("typed worker context native byte-repack declaration differs")
        maximum, byte_limit = projection.get("max_capsules"), projection.get("max_bytes")
        if (type(maximum) is not int or not 0 <= maximum <= 32 or selected_count > maximum
                or type(byte_limit) is not int or not 8192 <= byte_limit <= 65536
                or parent_bytes > byte_limit):
            raise ValueError("typed worker context projection exceeds native count or byte bounds")
        counts = {"full_capsule_count": full_count,
                  "selected_capsule_count": selected_count,
                  "omitted_capsule_count": full_count - selected_count}
        if any(type(projection.get(name)) is not int or projection[name] != expected
               for name, expected in counts.items()):
            raise ValueError("typed worker context projection denominator differs from native root")
        if projection.get("capsule_index_cid") != capsule_index_cid:
            raise ValueError("typed worker context projection capsule index differs from native root")
        selected = projection.get("selected_symbols")
        if (type(selected) is not list or any(type(row) is not dict for row in selected)
                or selected != expected_selected):
            raise ValueError("typed worker context projection selected symbol order or facts differ")
        required = set(semantic["required_raw_paths"])
        expected_fetch = {path: binding for path, binding in semantic["manifest"].items()
                          if path not in required}
        fetch = projection.get("raw_source_fetch_required")
        if type(fetch) is not dict or fetch != expected_fetch:
            raise ValueError("typed worker context raw source fetch frontier differs")
        for field in ("omission_reason", "raw_required_caveat"):
            value = projection[field]
            if type(value) is not str or not value or len(value.encode()) > 8192:
                raise ValueError("typed worker context projection requires bounded literal caveats")
    elif selected_count != full_count:
        raise ValueError("typed worker context unprojected parent omits native capsule population")
    binding = {
        "capsule_index_cid": capsule_index_cid,
        "full_symbol_count": full_count, "worker_symbol_count": selected_count,
        "omitted_symbol_count": full_count - selected_count,
        "parent_is_query_selected": projected, "producer_inventory_complete": True,
        "worker_inventory_complete": not projected,
    }
    if projected:
        binding.update(
            projection_schema=projection["schema"], selector=projection["selector"],
            query_sha256=codec._sha(projection["query"]),
            max_capsules=projection["max_capsules"], max_bytes=projection["max_bytes"],
            exact_byte_budget_repacked=projection.get("exact_byte_budget_repacked", False),
        )
    return inventory, binding


def _project_source_bound_typed_links(
    *, repository: Path, artifact: str, expected_sha256: str, task_id: str,
    expected_semantic_root_cid: str, graph_root_id: str,
    seed_symbol_ids: tuple[str, ...], max_selected_symbols: int = 32,
    max_graph_nodes: int = 2048, max_graph_edges: int = 8192,
    max_report_bytes: int = 2_000_000, allow_worker_projection: bool = False,
    require_current_source: bool = True,
    graph_scope: str = "full_root",
) -> dict:
    """Nominate recorded typed dependencies while preserving current-source checks.

    Requires a complete native worker artifact, independently expected hash,
    task and source root. A selected/role/decision view cannot stand in for its
    full parent. The returned graph is proposal-only, even though producer and
    current source checks passed at this invocation's opening/closing fences.
    An owner must still supply independent task-wide required/optional policy.
    """
    if type(allow_worker_projection) is not bool or type(require_current_source) is not bool:
        raise ValueError("literal native projection and current-source verification choices required")
    if type(graph_scope) is not str or graph_scope not in {"full_root", "task_closure"}:
        raise ValueError("literal supported typed graph scope required")
    if not isinstance(repository, Path) or not repository.is_absolute():
        raise ValueError("typed source context requires an absolute repository Path")
    root = repository.absolute()
    if root.resolve(strict=True) != root:
        raise ValueError("typed source context repository must be canonical")
    if type(artifact) is not str or len(artifact.encode()) > 4096:
        raise ValueError("typed source context requires a bounded artifact path")
    relative = PurePosixPath(artifact)
    if (relative.is_absolute() or str(relative) != artifact or ".." in relative.parts
            or artifact in {"", "."} or "\x00" in artifact):
        raise ValueError("typed source context artifact path must be canonical and relative")
    if type(expected_sha256) is not str or not _SHA.fullmatch(expected_sha256):
        raise ValueError("typed source context requires an independent SHA256")
    for name, value in (("task_id", task_id), ("semantic_root_cid", expected_semantic_root_cid),
                        ("graph_root_id", graph_root_id)):
        if (type(value) is not str or not value or value != value.strip()
                or "\x00" in value or len(value.encode()) > 8192):
            raise ValueError("typed source context requires bounded " + name)
    for name, value, maximum in (
        ("max_selected_symbols", max_selected_symbols, 32),
        ("max_graph_nodes", max_graph_nodes, 2048),
        ("max_graph_edges", max_graph_edges, 8192),
        ("max_report_bytes", max_report_bytes, 2_000_000),
    ):
        if type(value) is not int or not 1 <= value <= maximum:
            raise ValueError("typed source context has invalid " + name)
    if (type(seed_symbol_ids) is not tuple or not seed_symbol_ids
            or len(seed_symbol_ids) > max_selected_symbols
            or any(type(value) is not str or not value or len(value.encode()) > 8192
                   for value in seed_symbol_ids)
            or len(set(seed_symbol_ids)) != len(seed_symbol_ids)):
        raise ValueError("typed source context requires distinct bounded literal seed IDs")

    raw = codec._read(root, artifact)
    if codec._sha(raw) != expected_sha256:
        raise ValueError("typed source context artifact SHA256 differs")
    text = raw.decode("utf-8")
    initial = codec._parse(text)
    if (type(initial) is not dict
            or initial.get("schema") != "supervisor-semantic-worker-context@1"
            or ("worker_projection" in initial and not allow_worker_projection)):
        raise ValueError("typed source context requires the complete native worker artifact")
    if initial.get("semantic_root_cid") != expected_semantic_root_cid:
        raise ValueError("typed source context semantic root differs")
    semantic, _ = codec._verified_semantic(
        root, artifact=artifact, text=text, task_id=task_id, current=require_current_source,
    )
    if codec._json(semantic) != codec._json(initial):
        raise ValueError("typed source context verified parent differs")
    required = semantic.get("required_raw_paths")
    if (type(required) is not list or any(type(path) is not str for path in required)
            or len(required) != len(set(required))
            or not set(required) <= set(semantic["manifest"])
            or not set(required) <= set(semantic["raw_sources"])):
        raise ValueError("typed source context required raw paths lack literal delivery")
    # The private owner cache replaces only deterministic native graph replay.
    # Parent/source verification, worker joins and lineage reconstruction remain
    # fresh. Neither public input nor a printed verification flag supplies it.
    cache_binding = None
    proposal = None
    if os.environ.get("IPFS_ACCELERATE_TYPED_VERIFICATION_CACHE", "1") != "0":
        try:
            captured = _capture_blocks(root, artifact, expected_semantic_root_cid)
            cache_binding = _typed_verification_binding(
                root, artifact, expected_sha256, task_id, expected_semantic_root_cid,
                graph_root_id, seed_symbol_ids, graph_scope, allow_worker_projection,
                max_selected_symbols, max_graph_nodes, max_graph_edges, max_report_bytes,
                captured,
            )
            proposal = _load_cached_typed_proposal(root, cache_binding, max_report_bytes)
        except (OSError, ValueError, ImportError):
            # Cache eligibility is optional; the original native path owns
            # source/bundle refusal and must actually run on every miss.
            cache_binding = None
    cache_hit = proposal is not None
    if cache_hit:
        capsule_index_cid = codec._parse(captured[expected_semantic_root_cid].decode())["capsule_index_cid"]
    else:
        bundle = _load_bundle(root, artifact, expected_semantic_root_cid)
        capsule_index_cid = bundle.root.capsule_index_cid if allow_worker_projection else None
        proposal = nominate_typed_semantic_dependencies(
            bundle=bundle, root_cid=expected_semantic_root_cid, graph_root_id=graph_root_id,
            seed_symbol_ids=seed_symbol_ids, max_selected_symbols=max_selected_symbols,
            max_graph_nodes=max_graph_nodes, max_graph_edges=max_graph_edges,
            max_report_bytes=max_report_bytes,
            graph_scope=graph_scope,
        )
    population = [row["stable_symbol_id"] for row in semantic["capsules"]]
    if (len(population) != len(set(population))
            or (not allow_worker_projection and sorted(population) != proposal["full_symbol_ids"])):
        raise ValueError("typed source context parent omits native capsule population")
    symbol_records = _proposal_symbol_records(proposal, graph_scope=graph_scope)
    projected_ids = [record["native_symbol_id"] for record in symbol_records]
    expected_graph_ids = proposal["full_symbol_ids"] if allow_worker_projection else population
    if sorted(projected_ids) != sorted(expected_graph_ids):
        raise ValueError("typed source context graph symbol population differs")
    for record in symbol_records:
        binding = semantic["manifest"].get(record["source_module_path"])
        if binding is None or binding["source_cid"] != record["source_cid"]:
            raise ValueError("typed source context symbol escapes current source frontier")
    result = {**proposal, "schema": SCHEMA if graph_scope == "full_root" else SCOPED_SCHEMA, "source_binding": {
        "task_id": task_id, "artifact": artifact, "artifact_sha256": expected_sha256,
        "semantic_root_cid": expected_semantic_root_cid, "scope_cid": semantic["scope_cid"],
        "source_manifest": semantic["manifest"],
        "required_raw_paths": semantic["required_raw_paths"],
        "raw_source_paths": sorted(semantic["raw_sources"]),
    }, "source_freshness_verified": require_current_source,
        "source_check_scope": ("opening_and_closing_current_native_verification" if require_current_source
                               else "opening_and_closing_historical_native_verification"),
        "owner_task_policy_verified": False}
    if allow_worker_projection:
        inventory, projection_binding = _worker_projection_binding(
            semantic=semantic, capsule_index_cid=capsule_index_cid, symbol_records=symbol_records,
            parent_bytes=len(raw),
        )
        result.update(worker_inventory=inventory, worker_projection_binding=projection_binding)
        if projection_binding["parent_is_query_selected"]:
            result["schema"] = WORKER_SCHEMA if graph_scope == "full_root" else SCOPED_WORKER_SCHEMA
    if len(codec._json(result).encode()) > max_report_bytes:
        raise ValueError("typed source context full report exceeds byte bound")
    final, _ = codec._verified_semantic(
        root, artifact=artifact, text=text, task_id=task_id, current=require_current_source,
    )
    if codec._json(final) != codec._json(semantic):
        raise ValueError("typed source context parent changed during projection")
    if cache_binding is not None:
        closing_blocks = _capture_blocks(root, artifact, expected_semantic_root_cid)
        try:
            closing = _typed_verification_binding(
                root, artifact, expected_sha256, task_id, expected_semantic_root_cid,
                graph_root_id, seed_symbol_ids, graph_scope, allow_worker_projection,
                max_selected_symbols, max_graph_nodes, max_graph_edges, max_report_bytes,
                closing_blocks,
            )
        except _TypedVerificationEligibilityError as error:
            if cache_hit:
                raise ValueError("authenticated typed verification cache closing capture unavailable") from error
            # A completed full native replay is valid without optional cache
            # publication. Never mint an entry without a complete closing key.
            return result
        if closing != cache_binding:
            raise ValueError("typed source context full block population or validators changed")
        if cache_hit:
            if _load_cached_typed_proposal(root, closing, max_report_bytes) != proposal:
                raise ValueError("authenticated typed verification cache changed during replay")
        elif require_current_source:
            # Historical readers never mint reusable current-owner evidence.
            # Invalid/unavailable cache storage leaves successful native work
            # usable; it cannot make a cache entry eligible for later reuse.
            _publish_cached_typed_proposal(root, closing, proposal, max_report_bytes)
    return result


def replay_source_bound_typed_links_for_audit(
    *, repository: Path, artifact: str, expected_sha256: str, task_id: str,
    expected_semantic_root_cid: str, graph_root_id: str,
    seed_symbol_ids: tuple[str, ...], max_selected_symbols: int = 32,
    max_graph_nodes: int = 2048, max_graph_edges: int = 8192,
    max_report_bytes: int = 2_000_000, graph_scope: str = "full_root",
) -> dict:
    """Replay retained producer evidence; never claim current source freshness."""
    # First use the common full-parent path. A projected parent is handled by
    # the same implementation with the complete retained native root. Detect
    # projection only after canonical/bounded native path validation.
    if not isinstance(repository, Path) or not repository.is_absolute():
        raise ValueError("historical typed replay requires an absolute repository Path")
    root = repository.absolute()
    if root.resolve(strict=True) != root:
        raise ValueError("historical typed replay repository must be canonical")
    if type(artifact) is not str or len(artifact.encode()) > 4096:
        raise ValueError("bounded historical typed artifact required")
    relative = PurePosixPath(artifact)
    if (relative.is_absolute() or str(relative) != artifact or ".." in relative.parts
            or artifact in {"", "."} or "\x00" in artifact):
        raise ValueError("canonical relative historical typed artifact required")
    if type(expected_sha256) is not str or not _SHA.fullmatch(expected_sha256):
        raise ValueError("independent historical typed artifact hash required")
    raw = codec._read(root, artifact)
    if codec._sha(raw) != expected_sha256:
        raise ValueError("historical typed artifact SHA256 differs")
    parent = codec._parse(raw.decode())
    if type(parent) is not dict:
        raise ValueError("historical native worker object required")
    return _project_source_bound_typed_links(
        repository=root, artifact=artifact, expected_sha256=expected_sha256,
        task_id=task_id, expected_semantic_root_cid=expected_semantic_root_cid,
        graph_root_id=graph_root_id, seed_symbol_ids=seed_symbol_ids,
        max_selected_symbols=max_selected_symbols, max_graph_nodes=max_graph_nodes,
        max_graph_edges=max_graph_edges, max_report_bytes=max_report_bytes,
        allow_worker_projection="worker_projection" in parent, require_current_source=False,
        graph_scope=graph_scope,
    )


_TYPED_CACHE_SCHEMA = "supervisor-semantic-typed-verification-cache@1"
_TYPED_CACHE_DIRECTORY = "semantic-typed-verification-cache"
_TYPED_CACHE_KEY = ".owner-key"
_TYPED_CACHE_MAXIMUM = 2_000_000
_TYPED_CACHE_FAILURES = (
    OSError, ValueError, TypeError, OverflowError, RecursionError, RuntimeError,
)
_TYPED_CACHE_STAT_FIELDS = (
    "st_dev", "st_ino", "st_size", "st_mtime_ns", "st_ctime_ns", "st_uid", "st_mode", "st_nlink",
)


def _typed_cache_limit(maximum):
    if type(maximum) is not int or not 1 <= maximum <= _TYPED_CACHE_MAXIMUM:
        raise ValueError("bounded literal typed cache byte limit required")
    return maximum


def _typed_cache_json_bytes(value, maximum):
    """Encode only bounded literal JSON, before touching owner cache state."""
    if type(value) is not dict:
        raise ValueError("literal typed cache JSON object required")
    stack, nodes, estimated = [(value, 0)], 0, 0
    while stack:
        item, depth = stack.pop()
        nodes += 1
        if nodes > 250_000 or depth > 64:
            raise ValueError("typed cache JSON exceeds its structure bound")
        kind = type(item)
        if kind is dict:
            estimated += 2 + max(0, len(item) - 1)
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("literal typed cache JSON keys required")
                estimated += len(key.encode("utf-8")) + 3
                stack.append((child, depth + 1))
        elif kind is list:
            estimated += 2 + max(0, len(item) - 1)
            stack.extend((child, depth + 1) for child in item)
        elif kind is str:
            estimated += len(item.encode("utf-8")) + 2
        elif kind is bool or item is None:
            estimated += 5
        elif kind is int:
            if item.bit_length() > 128:
                raise ValueError("bounded typed cache JSON integer required")
            estimated += 40
        elif kind is float:
            if not math.isfinite(item):
                raise ValueError("finite typed cache JSON number required")
            estimated += 32
        else:
            raise ValueError("literal typed cache JSON values required")
        if estimated > maximum:
            raise ValueError("typed cache JSON exceeds its byte bound")
    raw = codec._json(value).encode("utf-8")
    if not 1 <= len(raw) <= maximum:
        raise ValueError("typed cache JSON exceeds its encoded byte bound")
    return raw


def _typed_cache_json_object(raw, maximum):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("duplicate typed cache JSON key")
            result[key] = value
        return result

    def reject_constant(value):
        raise ValueError("non-finite typed cache JSON constant")

    value = json.loads(raw.decode("utf-8"), object_pairs_hook=unique,
                       parse_constant=reject_constant)
    if _typed_cache_json_bytes(value, maximum) != raw:
        raise ValueError("exact canonical typed cache JSON required")
    return value


def _typed_cache_stat(info):
    return tuple(getattr(info, field) for field in _TYPED_CACHE_STAT_FIELDS)


def _typed_cache_directory_info(descriptor, *, mode=None, owned=True):
    info = os.fstat(descriptor)
    if (not stat.S_ISDIR(info.st_mode)
            or (owned and info.st_uid != os.geteuid())
            or (owned and stat.S_IMODE(info.st_mode) & 0o022)
            or (mode is not None and stat.S_IMODE(info.st_mode) != mode)):
        raise ValueError("protected typed cache directory required")
    return (info.st_dev, info.st_ino, info.st_uid, info.st_mode)


def _typed_cache_repository_fd(repository):
    """Reject symlinks at every absolute-path component using directory fds."""
    if not isinstance(repository, Path) or not repository.is_absolute():
        raise ValueError("absolute typed cache repository Path required")
    text = str(repository)
    if (len(text.encode()) > 4096 or len(repository.parts) > 128
            or any(char in text for char in ("\x00", "\n", "\r"))
            or str(Path(text)) != text or ".." in repository.parts):
        raise ValueError("bounded canonical typed cache repository required")
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
    descriptor = os.open(repository.anchor, flags)
    try:
        for component in repository.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        _typed_cache_directory_info(descriptor)
        result, descriptor = descriptor, None
        return result
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _typed_cache_child_fd(parent, name, *, create, mode=None):
    created = False
    if create:
        try:
            os.mkdir(name, 0o700 if mode == 0o700 else 0o755, dir_fd=parent)
            created = True
        except FileExistsError:
            pass
    descriptor = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=parent)
    try:
        # Existing permissions are never changed or repaired.
        _typed_cache_directory_info(descriptor, mode=mode)
        if created:
            os.fsync(descriptor)
            os.fsync(parent)
        return descriptor
    except BaseException:
        os.close(descriptor)
        raise


def _typed_cache_open(repository, *, create):
    root = runtime = cache = None
    try:
        root = _typed_cache_repository_fd(repository)
        runtime = _typed_cache_child_fd(root, ".runtime", create=create)
        cache = _typed_cache_child_fd(runtime, _TYPED_CACHE_DIRECTORY,
                                     create=create, mode=0o700)
        result = root, runtime, cache
        root = runtime = cache = None
        return result
    finally:
        for descriptor in (cache, runtime, root):
            if descriptor is not None:
                os.close(descriptor)


def _typed_cache_check_paths(repository, descriptors):
    """The final public path must still designate the exact held directories."""
    current = _typed_cache_open(repository, create=False)
    try:
        for position, (held, fresh) in enumerate(zip(descriptors, current)):
            mode = 0o700 if position == 2 else None
            if (_typed_cache_directory_info(held, mode=mode)
                    != _typed_cache_directory_info(fresh, mode=mode)):
                raise ValueError("typed cache directory path changed")
    finally:
        for descriptor in reversed(current):
            os.close(descriptor)


def _typed_cache_read_file(parent, name, maximum, *, exact_size=None):
    descriptor = os.open(name, os.O_RDONLY | os.O_NONBLOCK | os.O_NOFOLLOW | os.O_CLOEXEC,
                         dir_fd=parent)
    try:
        before = os.fstat(descriptor)
        if (not stat.S_ISREG(before.st_mode) or before.st_uid != os.geteuid()
                or before.st_nlink != 1 or stat.S_IMODE(before.st_mode) != 0o400
                or not 1 <= before.st_size <= maximum
                or (exact_size is not None and before.st_size != exact_size)):
            raise ValueError("single-link immutable owner-only typed cache file required")
        with os.fdopen(descriptor, "rb", closefd=False) as stream:
            raw = stream.read(maximum + 1)
        after = os.fstat(descriptor)
        path_info = os.stat(name, dir_fd=parent, follow_symlinks=False)
        signature = _typed_cache_stat(before)
        if (len(raw) != before.st_size or len(raw) > maximum
                or signature != _typed_cache_stat(after)
                or signature != _typed_cache_stat(path_info)):
            raise ValueError("typed cache file changed or its path was replaced")
        return raw, signature
    finally:
        os.close(descriptor)


def _typed_cache_atomic_file(parent, name, raw):
    """Publish a complete unnamed inode atomically, without replacing a name.

    Linking an O_TMPFILE through our held proc fd exposes exactly one name.
    Process death leaves either no file or a complete single-link file; readers
    never need to repair named temporaries or accept a two-link crash residue.
    Unsupported filesystems retain the caller's ordinary native fallback.
    """
    anonymous = getattr(os, "O_TMPFILE", None)
    if type(anonymous) is not int or not anonymous:
        raise OSError("anonymous typed cache publication is unavailable")
    descriptor = None
    try:
        descriptor = os.open(".", os.O_WRONLY | anonymous | os.O_CLOEXEC,
                             0o600, dir_fd=parent)
        info = os.fstat(descriptor)
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.geteuid() or info.st_nlink != 0:
            raise ValueError("new unnamed owner-only typed cache inode required")
        offset = 0
        while offset < len(raw):
            count = os.write(descriptor, raw[offset:])
            if count <= 0:
                raise OSError("short typed cache publication write")
            offset += count
        os.fsync(descriptor)
        os.fchmod(descriptor, 0o400)
        os.fsync(descriptor)
        try:
            # The sole followed symlink is our own live descriptor, never a
            # caller-controlled source path. dst_dir_fd forces linkat semantics.
            os.link("/proc/self/fd/" + str(descriptor), name, dst_dir_fd=parent,
                    follow_symlinks=True)
        except FileExistsError:
            # The caller freshly reads/authenticates the existing immutable file.
            return
        published = os.stat(name, dir_fd=parent, follow_symlinks=False)
        held = os.fstat(descriptor)
        if ((published.st_dev, published.st_ino) != (held.st_dev, held.st_ino)
                or not stat.S_ISREG(published.st_mode)
                or published.st_uid != os.geteuid() or published.st_nlink != 1
                or held.st_nlink != 1 or stat.S_IMODE(published.st_mode) != 0o400
                or published.st_size != len(raw) or held.st_size != len(raw)):
            raise ValueError("published typed cache inode custody differs")
        os.fsync(parent)
    finally:
        if descriptor is not None:
            os.close(descriptor)


def _typed_cache_key(parent, *, create):
    try:
        return _typed_cache_read_file(parent, _TYPED_CACHE_KEY, 32, exact_size=32)
    except FileNotFoundError:
        if not create:
            raise
    _typed_cache_atomic_file(parent, _TYPED_CACHE_KEY, secrets.token_bytes(32))
    # A concurrent protected publisher may have won; only the actual key counts.
    return _typed_cache_read_file(parent, _TYPED_CACHE_KEY, 32, exact_size=32)


def _typed_cache_entry_name(binding_raw, key):
    generation = hashlib.sha256(key).digest()
    return hashlib.sha256(b"typed-native-cache-entry@1\0" + generation + binding_raw).hexdigest() + ".json"


def _typed_cache_authenticated_entry(raw, *, binding_raw, key, maximum):
    envelope = _typed_cache_json_object(raw, maximum)
    if (set(envelope) != {"schema", "payload", "hmac_sha256"}
            or envelope["schema"] != _TYPED_CACHE_SCHEMA
            or type(envelope["payload"]) is not dict):
        raise ValueError("closed typed cache envelope required")
    payload = envelope["payload"]
    if (set(payload) != {"binding", "proposal", "owner_uid", "key_generation_sha256"}
            or type(payload["owner_uid"]) is not int or payload["owner_uid"] != os.geteuid()
            or payload["key_generation_sha256"] != hashlib.sha256(key).hexdigest()
            or type(payload["binding"]) is not dict or type(payload["proposal"]) is not dict
            or _typed_cache_json_bytes(payload["binding"], maximum) != binding_raw):
        raise ValueError("typed cache input or owner-key generation differs")
    signature = envelope["hmac_sha256"]
    if (type(signature) is not str or not _SHA.fullmatch(signature)
            or not hmac.compare_digest(signature,
                hmac.new(key, _typed_cache_json_bytes(payload, maximum), hashlib.sha256).hexdigest())):
        raise ValueError("typed cache owner authentication failed")
    # json.loads already detached every nested object from caller and stored bytes.
    return payload["proposal"]


def _load_cached_typed_proposal(repository, binding, maximum):
    """Read a locally authenticated proposal; never create or repair cache state.

    This authenticates only the same protected owner UID/key trust domain. The
    binding must come from fresh complete native input and validator captures;
    cache contents or public verification flags cannot supply that binding.
    The caller still owns all parent/source, policy and closing-input fences.
    """
    descriptors = None
    try:
        maximum = _typed_cache_limit(maximum)
        binding_raw = _typed_cache_json_bytes(binding, maximum)
        descriptors = _typed_cache_open(repository, create=False)
        cache = descriptors[2]
        key, key_info = _typed_cache_key(cache, create=False)
        name = _typed_cache_entry_name(binding_raw, key)
        raw, entry_info = _typed_cache_read_file(cache, name, maximum)
        proposal = _typed_cache_authenticated_entry(raw, binding_raw=binding_raw,
                                                   key=key, maximum=maximum)
        # Close both key and record custody before exposing the authenticated copy.
        if (_typed_cache_key(cache, create=False) != (key, key_info)
                or _typed_cache_read_file(cache, name, maximum) != (raw, entry_info)):
            raise ValueError("typed cache owner key or record changed during lookup")
        _typed_cache_check_paths(repository, descriptors)
        return proposal
    except _TYPED_CACHE_FAILURES:
        return None
    finally:
        if descriptors is not None:
            for descriptor in reversed(descriptors):
                os.close(descriptor)


def _publish_cached_typed_proposal(repository, binding, proposal, maximum):
    """Best-effort publication AFTER genuine native success and closing fences.

    No public API should accept a caller's claimed verified proposal and forward
    it here. Failure remains a cache miss, never native validation success.
    An existing invalid state is neither overwritten nor permission-repaired.
    """
    descriptors = None
    try:
        maximum = _typed_cache_limit(maximum)
        binding_raw = _typed_cache_json_bytes(binding, maximum)
        proposal_raw = _typed_cache_json_bytes(proposal, maximum)
        # Capture detached literal values once; later caller mutation cannot alter MAC.
        binding_copy = _typed_cache_json_object(binding_raw, maximum)
        proposal_copy = _typed_cache_json_object(proposal_raw, maximum)
        descriptors = _typed_cache_open(repository, create=True)
        cache = descriptors[2]
        key, key_info = _typed_cache_key(cache, create=True)
        name = _typed_cache_entry_name(binding_raw, key)
        payload = {"binding": binding_copy, "proposal": proposal_copy,
                   "owner_uid": os.geteuid(),
                   "key_generation_sha256": hashlib.sha256(key).hexdigest()}
        signature = hmac.new(key, _typed_cache_json_bytes(payload, maximum),
                             hashlib.sha256).hexdigest()
        raw = _typed_cache_json_bytes({"schema": _TYPED_CACHE_SCHEMA, "payload": payload,
                                      "hmac_sha256": signature}, maximum)
        try:
            existing, _ = _typed_cache_read_file(cache, name, maximum)
        except FileNotFoundError:
            _typed_cache_atomic_file(cache, name, raw)
        else:
            if existing != raw:
                raise ValueError("existing typed cache entry differs; replacement forbidden")
        published, published_info = _typed_cache_read_file(cache, name, maximum)
        if published != raw:
            raise ValueError("published typed cache entry differs")
        _typed_cache_authenticated_entry(published, binding_raw=binding_raw,
                                         key=key, maximum=maximum)
        if (_typed_cache_key(cache, create=False) != (key, key_info)
                or _typed_cache_read_file(cache, name, maximum) != (published, published_info)):
            raise ValueError("typed cache owner key or record changed during publication")
        _typed_cache_check_paths(repository, descriptors)
    except _TYPED_CACHE_FAILURES:
        return None
    finally:
        if descriptors is not None:
            for descriptor in reversed(descriptors):
                os.close(descriptor)
    return None


_typed_code_native_packages = frozenset({"ipfs_accelerate_py", "ipfs_datasets_py", "ipfs_kit_py"})

_typed_code_minimal_cid_dependency_trees = frozenset({
    "multiformats", "multiformats_config", "typing_validation", "bases",
})

_typed_code_max_files = 32_768

_typed_code_max_directories = 16_384

_typed_code_max_total_bytes = 512 * 1024 * 1024

_typed_code_max_file_bytes = 96 * 1024 * 1024

_typed_code_max_profile_bytes = 131_072

_typed_code_chunk_bytes = 262_144

_typed_code_stat_fields = (
    "st_dev", "st_ino", "st_mode", "st_uid", "st_gid", "st_nlink",
    "st_size", "st_mtime_ns", "st_ctime_ns",
)

def _typed_code_stamp(info):
    return tuple(getattr(info, field) for field in _typed_code_stat_fields)

def _typed_code_canonical(value):
    """Bound literal JSON before encoding independently supplied profile data."""
    pending, cells, text_bytes = [(value, 0)], 0, 0
    while pending:
        item, depth = pending.pop()
        cells += 1
        if cells > 8192 or depth > 24:
            raise ValueError("validator profile exceeds structural bound")
        if type(item) is dict:
            for key, child in item.items():
                if type(key) is not str:
                    raise ValueError("literal validator profile keys required")
                pending.extend(((key, depth + 1), (child, depth + 1)))
        elif type(item) is list:
            pending.extend((child, depth + 1) for child in item)
        elif type(item) is str:
            text_bytes += len(item.encode("utf-8"))
            if text_bytes > _typed_code_max_profile_bytes:
                raise ValueError("validator profile exceeds text bound")
        elif type(item) is int:
            if item.bit_length() > 64:
                raise ValueError("validator profile integer exceeds bound")
        elif item is not None and type(item) is not bool:
            raise ValueError("literal bounded validator profile JSON required")
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True,
                         separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(encoded) > _typed_code_max_profile_bytes:
        raise ValueError("validator profile exceeds encoded bound")
    return encoded

def _typed_code_protected(info, uid, *, directory):
    predicate = stat.S_ISDIR if directory else stat.S_ISREG
    if (not predicate(info.st_mode) or info.st_uid != uid
            or stat.S_IMODE(info.st_mode) & 0o022
            or (not directory and info.st_nlink != 1)):
        raise ValueError("protected single-owner validator input required")

def _typed_code_path(value):
    if (not isinstance(value, Path) or not value.is_absolute()
            or str(value) != os.path.normpath(str(value))):
        raise ValueError("absolute canonical validator input Path required")
    return value

def _typed_code_open_protected_path(path, uid, *, directory):
    """Pin every ancestor via dir-fd/no-follow rather than resolve per file."""
    path = _typed_code_path(path)
    descriptor = os.open("/", os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
    try:
        _typed_code_protected(os.fstat(descriptor), uid, directory=True)
        for index, part in enumerate(path.parts[1:]):
            final = index == len(path.parts) - 2
            is_dir = directory or not final
            flags = os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK
            if is_dir:
                flags |= os.O_DIRECTORY
            child = os.open(part, flags, dir_fd=descriptor)
            try:
                _typed_code_protected(os.fstat(child), uid, directory=is_dir)
            except BaseException:
                os.close(child)
                raise
            os.close(descriptor)
            descriptor = child
        result, descriptor = descriptor, None
        return result
    finally:
        if descriptor is not None:
            os.close(descriptor)

def _typed_code_read_hash(fd, before, budget):
    if not 0 <= before.st_size <= _typed_code_max_file_bytes:
        raise ValueError("validator code file exceeds byte bound")
    budget["files"] += 1
    budget["bytes"] += before.st_size
    if budget["files"] > _typed_code_max_files or budget["bytes"] > _typed_code_max_total_bytes:
        raise ValueError("validator code population exceeds bound")
    digest, amount = hashlib.sha256(), 0
    while True:
        chunk = os.read(fd, min(_typed_code_chunk_bytes, _typed_code_max_file_bytes - amount + 1))
        if not chunk:
            break
        amount += len(chunk)
        if amount > before.st_size:
            raise ValueError("validator code file grew during capture")
        digest.update(chunk)
    if amount != before.st_size or _typed_code_stamp(os.fstat(fd)) != _typed_code_stamp(before):
        raise ValueError("validator code file changed during capture")
    return digest.hexdigest()

def _typed_code_tree(path, uid, budget, *, reject_bytecode):
    """Hash all regular files; reject special files/symlinks and tree races.

    Per-directory mode/owner rows are included in the tree commitment. Device,
    inode, mtime and ctime are race guards, not cache-key substitutes for bytes.
    """
    root_fd = _typed_code_open_protected_path(path, uid, directory=True)
    rows, relative_files = [], set()

    def descend(fd, relative, depth):
        if depth > 64:
            raise ValueError("validator code directory depth exceeds bound")
        before = os.fstat(fd)
        _typed_code_protected(before, uid, directory=True)
        budget["directories"] += 1
        if budget["directories"] > _typed_code_max_directories:
            raise ValueError("validator code directory population exceeds bound")
        rows.append(["directory", relative, stat.S_IMODE(before.st_mode), before.st_uid])
        # Disjoint from file names, these markers retain the directories already
        # committed above, including empty namespaces, without changing the key.
        relative_files.add(relative + "/")
        with os.scandir(fd) as entries:
            names = []
            for entry in entries:
                if len(names) >= _typed_code_max_files + _typed_code_max_directories:
                    raise ValueError("validator code directory entries exceed bound")
                name = entry.name
                if type(name) is not str or name in {".", ".."} or "/" in name or "\0" in name:
                    raise ValueError("literal validator directory entry required")
                names.append(name)
        if len(set(names)) != len(names):
            raise ValueError("duplicate validator directory entry")
        for name in sorted(names):
            child_relative = relative + "/" + name if relative else name
            info = os.stat(name, dir_fd=fd, follow_symlinks=False)
            if stat.S_ISDIR(info.st_mode):
                child = os.open(name, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=fd)
                try:
                    if _typed_code_stamp(os.fstat(child)) != _typed_code_stamp(info):
                        raise ValueError("validator code directory changed before open")
                    descend(child, child_relative, depth + 1)
                    if _typed_code_stamp(os.stat(name, dir_fd=fd, follow_symlinks=False)) != _typed_code_stamp(info):
                        raise ValueError("validator code directory changed after traversal")
                finally:
                    os.close(child)
                continue
            _typed_code_protected(info, uid, directory=False)
            if reject_bytecode and (name.endswith((".pyc", ".pyo")) or "__pycache__" in child_relative.split("/")):
                raise ValueError("native source tree contains unadmitted Python bytecode")
            child = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=fd)
            try:
                opened = os.fstat(child)
                _typed_code_protected(opened, uid, directory=False)
                if _typed_code_stamp(opened) != _typed_code_stamp(info):
                    raise ValueError("validator code file changed before open")
                digest = _typed_code_read_hash(child, opened, budget)
                if _typed_code_stamp(os.stat(name, dir_fd=fd, follow_symlinks=False)) != _typed_code_stamp(info):
                    raise ValueError("validator code file replaced after capture")
            finally:
                os.close(child)
            rows.append(["file", child_relative, info.st_size,
                         stat.S_IMODE(info.st_mode), info.st_uid, digest])
            relative_files.add(child_relative)
        if _typed_code_stamp(os.fstat(fd)) != _typed_code_stamp(before):
            raise ValueError("validator code directory changed during traversal")

    before_count, before_bytes = budget["files"], budget["bytes"]
    try:
        descend(root_fd, "", 0)
        # Reopen the root canonically to detect an ancestor/root replacement.
        closing = _typed_code_open_protected_path(path, uid, directory=True)
        try:
            if _typed_code_stamp(os.fstat(closing)) != _typed_code_stamp(os.fstat(root_fd)):
                raise ValueError("validator code root changed during capture")
        finally:
            os.close(closing)
    finally:
        os.close(root_fd)
    # This finite inventory can exceed the profile-specific JSON bound.
    encoded = json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode()
    return {"path": str(path), "files": budget["files"] - before_count,
            "bytes": budget["bytes"] - before_bytes,
            "population_sha256": hashlib.sha256(encoded).hexdigest()}, relative_files

def _typed_code_file(path, uid, budget):
    fd = _typed_code_open_protected_path(path, uid, directory=False)
    try:
        before = os.fstat(fd)
        digest = _typed_code_read_hash(fd, before, budget)
        closing = _typed_code_open_protected_path(path, uid, directory=False)
        try:
            if _typed_code_stamp(os.fstat(closing)) != _typed_code_stamp(before):
                raise ValueError("validator runtime file replaced during capture")
        finally:
            os.close(closing)
    finally:
        os.close(fd)
    return {"path": str(path), "bytes": before.st_size, "mode": stat.S_IMODE(before.st_mode),
            "uid": before.st_uid, "sha256": digest}

def _typed_code_namespace(name, module, prefix, root, membership):
    """Admit only the standard single-root namespace in the fresh native tree."""
    # Unsupported import implementations disable this optional cache path;
    # they must not prevent the ordinary native wrapper from importing.
    from importlib._bootstrap_external import _NamespacePath
    from importlib.machinery import ModuleSpec, NamespaceLoader, PathFinder, all_suffixes
    spec = getattr(module, "__spec__", None)
    locations = getattr(module, "__path__", None)
    loader = getattr(module, "__loader__", None)
    if (prefix not in _typed_code_native_packages
            or type(spec) is not ModuleSpec or type(loader) is not NamespaceLoader
            or spec.loader is not loader or spec.name != name
            or getattr(module, "__name__", None) != name
            or getattr(module, "__package__", None) != name or spec.parent != name
            or spec.origin is not None or getattr(module, "__file__", None) is not None
            or spec.has_location or spec.cached is not None
            or type(locations) is not _NamespacePath
            or spec.submodule_search_locations is not locations
            or getattr(loader, "_path", None) is not locations
            or getattr(locations, "_name", None) != name):
        raise ValueError("loaded validator namespace has no standard native provenance")
    # Check provenance before iterating: _NamespacePath can recalculate through
    # a callback when its parent search path changes.
    finder = getattr(locations, "_path_finder", None)
    expected_finder = PathFinder._get_spec
    stored_paths = getattr(locations, "_path", None)
    if (type(finder) is not type(expected_finder)
            or getattr(finder, "__self__", None) is not expected_finder.__self__
            or getattr(finder, "__func__", None) is not expected_finder.__func__
            or type(stored_paths) is not list or len(stored_paths) != 1):
        raise ValueError("loaded validator namespace search provenance differs")
    suffix = name.split(".")[len(prefix.split(".")):]
    if any(not part.isidentifier() for part in suffix):
        raise ValueError("loaded validator namespace name is noncanonical")
    directory = root.joinpath(*suffix)
    paths = tuple(locations)
    if (len(paths) != 1 or type(paths[0]) is not str
            or paths[0] != str(directory) or directory.resolve(strict=True) != directory):
        raise ValueError("loaded validator namespace search path differs")
    relative = "/".join(suffix)
    if relative + "/" not in membership:
        raise ValueError("loaded validator namespace is absent from fresh code inventory")
    base = relative + "/" if relative else ""
    if any(base + "__init__" + extension in membership for extension in all_suffixes()):
        raise ValueError("loaded validator namespace masks a regular native package")


def _typed_code_origins(trees, files):
    """Validate loaded namespaces against independently supplied exact roots.

    Module count/origins are observations, deliberately excluded from the
    deterministic cache key: publisher and observer import different subsets.
    No unknown namespace root or foreign file beneath a checked name is allowed.
    """
    checked = 0
    for name, module in tuple(sys.modules.items()):
        if module is None or type(name) is not str:
            continue
        prefix = next((key for key in trees if name == key or name.startswith(key + ".")), None)
        if prefix is None:
            if name not in files:
                continue
            expected, membership = files[name], None
        else:
            expected, membership = trees[prefix]
        origin = getattr(getattr(module, "__spec__", None), "origin", None)
        actual = getattr(module, "__file__", None)
        if membership is not None and actual is None and origin is None:
            try:
                _typed_code_namespace(name, module, prefix, expected, membership)
            except (AttributeError, TypeError) as exc:
                # Optional private import machinery may differ or be malformed.
                # Preserve ordinary native fallback, including closing refusal
                # when the opening fence actually used an authenticated hit.
                raise ValueError("loaded validator namespace inspection is unsupported") from exc
            checked += 1
            continue
        if type(actual) is not str or type(origin) is not str or actual != origin:
            raise ValueError("loaded validator module has no exact native file origin")
        path = Path(actual)
        if not path.is_absolute() or path.resolve(strict=True) != path:
            raise ValueError("loaded validator module origin is noncanonical")
        if membership is None:
            if path != expected:
                raise ValueError("loaded validator dependency file origin differs")
        else:
            try:
                relative = str(path.relative_to(expected))
            except ValueError as exc:
                raise ValueError("loaded validator module belongs to a foreign package root") from exc
            if relative not in membership:
                raise ValueError("loaded validator module is absent from fresh code inventory")
            locations = getattr(module, "__path__", None)
            if locations is not None and tuple(locations) != (str(path.parent),):
                raise ValueError("loaded validator package search path differs")
        checked += 1
    return checked

def _typed_code_fingerprint_native_validator_code(*, package_roots, dependency_roots, dependency_files,
                                      interpreter_path, trusted_uid, profile):
    """Minimal native/CID proposal fingerprint for immediate integration.

    All roots/files are explicit independently supplied configuration. No lazy
    imports or package discovery occur here. Missing dependencies disqualify a
    cache hit; the caller can take its ordinary native-verification path.

    This binds the full three native trees, the complete CID Python/data/bytecode
    closure, actual interpreter bytes/version/cache-tag/optimization and the
    independently expected profile. It deliberately DOES NOT assert fresh
    qualification of all stdlib, cryptography, shared-library or Python runtime
    state. Bind an independently authenticated immutable platform/dependency
    profile as well; use the wider helper if such inputs can change in place.
    Bytecode is rejected in native trees and hashed in dependency trees. Native
    loaded-code/startup fences and monkeypatch caveats above still apply.
    """
    if type(trusted_uid) is not int or trusted_uid < 0:
        raise ValueError("independently expected literal code owner UID required")
    if sys.pycache_prefix is not None:
        raise ValueError("native source-only cache eligibility requires no alternate pycache")
    for roots in (package_roots, dependency_roots, dependency_files):
        if type(roots) is not dict or any(type(name) is not str or not name for name in roots):
            raise ValueError("explicit independently admitted validator input maps required")
        for path in roots.values():
            _typed_code_path(path)
    if set(package_roots) != _typed_code_native_packages or not _typed_code_minimal_cid_dependency_trees <= set(dependency_roots):
        raise ValueError("complete native and CID package roots required")
    if "typing_extensions" not in dependency_files:
        raise ValueError("complete CID typing dependency file required")
    if type(profile) is not dict or not profile:
        raise ValueError("independently authenticated validator/platform profile required")
    profile_bytes = _typed_code_canonical(profile)
    _typed_code_path(interpreter_path)
    if interpreter_path != Path(sys.executable).resolve(strict=True):
        raise ValueError("actual interpreter differs from independently expected canonical executable")
    paths = [interpreter_path, *(path for roots in (package_roots, dependency_roots, dependency_files)
                                for path in roots.values())]
    if len(set(paths)) != len(paths):
        raise ValueError("validator code input aliases are not admitted")
    budget = {"files": 0, "directories": 0, "bytes": 0}
    tree_bindings, checked_trees, file_bindings = {}, {}, {}
    for category, roots in (("native", package_roots), ("dependency", dependency_roots)):
        for name, path in sorted(roots.items()):
            binding, members = _typed_code_tree(path, trusted_uid, budget, reject_bytecode=category == "native")
            tree_bindings[category + ":" + name] = binding
            checked_trees[name] = (path, members)
    for name, path in sorted(dependency_files.items()):
        file_bindings["dependency:" + name] = _typed_code_file(path, trusted_uid, budget)
    file_bindings["runtime:python"] = _typed_code_file(interpreter_path, trusted_uid, budget)
    checked = _typed_code_origins(checked_trees, dependency_files)
    binding = {
        "schema": "supervisor-native-cid-validator-code-binding@1",
        "profile_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        "trusted_uid": trusted_uid, "trees": tree_bindings, "files": file_bindings,
        "interpreter": {"implementation": sys.implementation.name,
                        "cache_tag": sys.implementation.cache_tag, "version": sys.version,
                        "version_info": list(sys.version_info), "byteorder": sys.byteorder,
                        "optimization": sys.flags.optimize},
        "population_files": budget["files"], "population_bytes": budget["bytes"],
    }
    return {
        "binding": binding,
        "binding_sha256": hashlib.sha256(_typed_code_canonical(binding)).hexdigest(),
        "observations": {"loaded_module_origins_checked": checked,
                         "source_fence_before_imports_required": True,
                         "whole_python_environment_qualified": False},
    }


def _typed_validator_code():
    """Fingerprint protected native/CID sources from actual canonical imports.

    Reuse requires root-owned sources/ancestors immutable to the non-root
    supervisor, plus the existing admitted startup boundary. Editable hosts
    fall back to native validation. Full package populations bind
    future validator dependencies conservatively; CID dependencies include their
    data and bytecode. Native bytecode or foreign import origins disable reuse.
    This is not a portable proof certificate or a whole-platform attestation.
    """
    if os.geteuid() == 0:
        raise ValueError("typed cache requires code immutable to the local owner")
    # The cache is qualified for a Linux owner without filesystem-bypass
    # capabilities. Root-owned sources/ancestors prevent native bytecode writes
    # even when the protected Python -I entry ignores PYTHONDONTWRITEBYTECODE.
    with open("/proc/self/status", encoding="ascii") as status_file:
        status = status_file.read(65_537)
    if len(status) > 65_536:
        raise ValueError("bounded owner capability status required")
    capabilities = {}
    for row in status.splitlines():
        name, separator, value = row.partition(":")
        if name in {"CapEff", "CapPrm", "CapAmb"} and separator:
            if name in capabilities:
                raise ValueError("distinct owner capability fields required")
            capabilities[name] = int(value.strip(), 16)
    if set(capabilities) != {"CapEff", "CapPrm", "CapAmb"} or any(capabilities.values()):
        raise ValueError("unprivileged owner required for immutable validator reuse")
    packages = {}
    for name in sorted(_typed_code_native_packages):
        module = sys.modules.get(name)
        if module is None or type(getattr(module, "__file__", None)) is not str:
            raise ValueError("canonical loaded native package required for cache eligibility")
        packages[name] = Path(module.__file__).parent
    dependencies = {}
    for name in sorted(_typed_code_minimal_cid_dependency_trees):
        module = importlib.import_module(name)
        filename = getattr(module, "__file__", None)
        if type(filename) is not str:
            raise ValueError("canonical CID dependency file required for cache eligibility")
        dependencies[name] = Path(filename).parent
    typing = importlib.import_module("typing_extensions")
    if type(getattr(typing, "__file__", None)) is not str:
        raise ValueError("canonical CID typing dependency file required")
    return _typed_code_fingerprint_native_validator_code(
        package_roots=packages, dependency_roots=dependencies,
        dependency_files={"typing_extensions": Path(typing.__file__)},
        interpreter_path=Path(sys.executable).resolve(strict=True), trusted_uid=0,
        profile={"schema": "supervisor-local-typed-validator-profile@1",
                 "trust_scope": "protected_owner_and_native_runtime",
                 "source_profile": "complete_native_packages_and_cid_dependencies@1"},
    )["binding"]


def _typed_verification_binding(repository, artifact, parent_sha256, task_id, root_cid,
                                graph_root_id, seeds, scope, projected, selected_limit,
                                node_limit, edge_limit, report_limit, blocks):
    """Fresh exact-byte key; neither root CID alone nor metadata is sufficient."""
    info = repository.stat(follow_symlinks=False)
    rows = [[cid, len(raw), hashlib.sha256(raw).hexdigest()] for cid, raw in sorted(blocks.items())]
    try:
        validator_code = _typed_validator_code()
    except (OSError, ValueError, ImportError) as error:
        raise _TypedVerificationEligibilityError("optional typed validator capture unavailable") from error
    return {
        "schema": "supervisor-semantic-typed-verification-inputs@1",
        "repository": str(repository), "repository_device": info.st_dev,
        "repository_inode": info.st_ino, "owner_uid": os.geteuid(),
        "artifact": artifact, "artifact_sha256": parent_sha256,
        "task_id": task_id, "semantic_root_cid": root_cid,
        "graph_root_id": graph_root_id, "seed_symbol_ids": list(seeds),
        "graph_scope": scope, "allow_worker_projection": projected,
        "limits": {"max_selected_symbols": selected_limit, "max_graph_nodes": node_limit,
                   "max_graph_edges": edge_limit, "max_report_bytes": report_limit},
        "block_population": {"count": len(rows), "bytes": sum(row[1] for row in rows),
                             "sha256": hashlib.sha256(codec._json(rows).encode()).hexdigest()},
        "validator_code": validator_code,
    }
