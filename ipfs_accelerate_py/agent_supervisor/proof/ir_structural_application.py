"""Apply AST / knowledge-graph / vector-index logic to intermediate representations.

General supervisor module (not SCA-taskboard-specific). Companion to
:mod:`ir_logic_application` (Intent/Legal/Security/UI shared-IR).

These surfaces are supervisor-owned analysis intermediate representations:

* **AST** — :mod:`..analysis.analysis_ast_index` (body-free blob records + query)
* **Knowledge graph** — :mod:`..analysis.semantic_dependency_graph`
* **Vector index** — :mod:`..analysis.code_symbol_vector_index`

No remote embedding backends. No source-body parsing — work surfaces project
path-bound AST blob records from operation/path/symbol identity under any
domain (planner, doctor, symbolic repair, SCA, contract repair, generic).
"""

from __future__ import annotations

import hashlib
import math
import re
from typing import Any, Final, Mapping, Sequence


STRUCTURAL_IR_INTERFACE: Final = "IrStructuralApplication@1"
DEFAULT_STRUCTURAL_FAMILIES: Final[tuple[str, ...]] = (
    "ast",
    "knowledge_graph",
    "vector_index",
)
DEFAULT_VECTOR_DIMS: Final[int] = 8


def _slug(value: str, *, maximum: int = 48) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9._-]+", "-", (value or "unknown").strip())
    cleaned = cleaned.strip("-._") or "unknown"
    return cleaned[:maximum]


def _sha256_hex(payload: str | bytes) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _default_path_for_op(operation: str, path: str = "", *, domain: str = "agent_supervisor") -> str:
    if path:
        return path
    op_slug = _slug(operation)
    domain_slug = _slug(domain or "agent_supervisor")
    return f"agent_supervisor/work_surfaces/{domain_slug}/{op_slug}.py"


def _op_symbols(operation: str) -> tuple[str, ...]:
    """Derive body-free symbol names from a dotted/qualified operation."""
    op = operation or "unknown"
    leaf = op.rsplit(".", 1)[-1]
    leaf = re.sub(r"[^a-zA-Z0-9_]", "_", leaf) or "operation"
    if leaf[0].isdigit():
        leaf = f"op_{leaf}"
    register = "register_tool"
    mediate = "mcp_tools_call"
    return (leaf, register, mediate)


def project_ast_blob_record(
    *,
    operation: str,
    contract_id: str = "",
    finding_kind: str = "",
    path: str = "",
    domain: str = "agent_supervisor",
) -> dict[str, Any]:
    """Project one body-free AST intermediate record for a work surface."""
    path = _default_path_for_op(operation, path, domain=domain)
    symbols = _op_symbols(operation)
    calls = ("register_tool", "mcp_tools_call")
    imports = ("mcp", "ipfs_accelerate_py.agent_supervisor")
    payload = "|".join(
        [path, operation, contract_id, finding_kind, *symbols, *calls]
    )
    digest = _sha256_hex(payload)
    return {
        "path": path,
        "blob_identity": f"blob:{digest[:24]}",
        "source_sha256": digest,
        "qualified_symbols": symbols,
        "imports": imports,
        "calls": calls,
        "interfaces": ("mcp_server", "package_mcp_interop"),
        "symbol_lines": {s: (1 + i, 1 + i) for i, s in enumerate(symbols)},
        "language": "python",
        "operation": operation,
        "contract_id": contract_id,
        "finding_kind": finding_kind,
    }


def apply_ast_logic(
    *,
    operation: str,
    contract_id: str = "",
    finding_kind: str = "",
    path: str = "",
    symbol: str = "",
    domain: str = "agent_supervisor",
) -> dict[str, Any]:
    """Build AST index IR and query it for work operation symbols."""
    from ..analysis.analysis_ast_index import ASTEvidenceKind, build_analysis_ast_index

    record = project_ast_blob_record(
        operation=operation,
        contract_id=contract_id,
        finding_kind=finding_kind,
        path=path,
        domain=domain,
    )
    index = build_analysis_ast_index(path_records=[record])
    query_term = symbol or _op_symbols(operation)[0]
    queries: dict[str, Any] = {}
    for kind_name, kind, term in (
        ("symbol", ASTEvidenceKind.SYMBOL, query_term),
        ("call", ASTEvidenceKind.CALL, "register_tool"),
        ("path", ASTEvidenceKind.PATH, record["path"]),
    ):
        try:
            result = index.query(kind, term)
            evidence = getattr(result, "evidence", ()) or ()
            queries[kind_name] = {
                "query": term,
                "match_count": len(evidence),
                "truncated": bool(
                    getattr(getattr(result, "truncation", None), "truncated", False)
                ),
                "top": [
                    {
                        "path": getattr(item, "path", ""),
                        "symbol": getattr(item, "symbol", ""),
                        "value": getattr(item, "value", ""),
                        "score": getattr(item, "score", 0),
                        "relationship": getattr(item, "relationship", ""),
                    }
                    for item in evidence[:5]
                ],
            }
        except Exception as exc:  # noqa: BLE001
            queries[kind_name] = {
                "query": term,
                "error": f"{type(exc).__name__}: {exc}",
            }

    symbol_hits = int((queries.get("symbol") or {}).get("match_count") or 0)
    return {
        "family": "ast",
        "applied": True,
        "available": True,
        "status": "indexed",
        "ok": symbol_hits > 0 or bool(getattr(index, "path_records", ())),
        "grants_execution_authority": False,
        "role": "ast_intermediate_representation",
        "index_id": getattr(index, "index_id", None),
        "paths": list(getattr(index, "paths", ()) or ()),
        "path_count": len(getattr(index, "path_records", ()) or ()),
        "record": {
            "path": record["path"],
            "blob_identity": record["blob_identity"],
            "symbols": list(record["qualified_symbols"]),
            "calls": list(record["calls"]),
        },
        "queries": queries,
        "logic_applied": [
            "project_ast_blob_record",
            "build_analysis_ast_index",
            "AnalysisASTIndex.query(symbol/call/path)",
        ],
        "notes": [
            "AST IR is body-free (no source parse in applicator).",
            "Queries are snapshot-bound evidence references, not authority grants.",
        ],
        "_raw_index": index,
        "_raw_record": record,
    }


def _l2_vector(seed: str, dimensions: int = DEFAULT_VECTOR_DIMS) -> tuple[float, ...]:
    digest = hashlib.sha256(seed.encode("utf-8")).digest()
    # Expand deterministically if dims > 32
    raw: list[float] = []
    block = digest
    while len(raw) < dimensions:
        for b in block:
            raw.append((b / 255.0) * 2.0 - 1.0)
            if len(raw) >= dimensions:
                break
        block = hashlib.sha256(block).digest()
    norm = math.sqrt(sum(x * x for x in raw)) or 1.0
    return tuple(x / norm for x in raw[:dimensions])


def apply_vector_index_logic(
    *,
    operation: str,
    contract_id: str = "",
    finding_kind: str = "",
    path: str = "",
    symbol: str = "",
    ast_result: Mapping[str, Any] | None = None,
    dimensions: int = DEFAULT_VECTOR_DIMS,
    domain: str = "agent_supervisor",
) -> dict[str, Any]:
    """Build deterministic code-symbol vector index IR from AST sidecar records."""
    from ..analysis.code_symbol_vector_index import (
        build_code_symbol_vector_index,
        search_code_symbol_vector_index,
    )

    if ast_result is None or not ast_result.get("_raw_index"):
        ast_result = apply_ast_logic(
            operation=operation,
            contract_id=contract_id,
            finding_kind=finding_kind,
            path=path,
            symbol=symbol,
            domain=domain,
        )
    index = ast_result.get("_raw_index")
    if index is None:
        return {
            "family": "vector_index",
            "applied": False,
            "ok": False,
            "error": "ast_index_unavailable",
            "grants_execution_authority": False,
        }

    forest_id = f"forest:ir:{_slug(operation)}"
    tree_id = f"tree:ir:{_slug(operation)}"

    def _vector_for_row(row: Any) -> tuple[float, ...]:
        seed = str(
            getattr(row, "qualified_symbol", None)
            or getattr(row, "symbol", None)
            or operation
        )
        return _l2_vector(seed, dimensions)

    snapshot = build_code_symbol_vector_index(
        index,
        forest_id=forest_id,
        tree_id=tree_id,
        vectors=_vector_for_row,
        dimensions=dimensions,
        model_id="ir-deterministic-fixture@1",
        model_revision="1",
        producer_id="ir-structural-application@1",
    )
    query_seed = symbol or _op_symbols(operation)[0]
    query_vector = _l2_vector(query_seed, dimensions)
    search = search_code_symbol_vector_index(
        snapshot, query_vector, max_results=5
    )
    hits = getattr(search, "hits", ()) or ()
    hit_rows = []
    for hit in hits[:5]:
        row = getattr(hit, "row", None)
        hit_rows.append(
            {
                "rank": getattr(hit, "rank", None),
                "score": getattr(hit, "score", None),
                "symbol": getattr(row, "symbol", None) if row is not None else None,
                "qualified_symbol": getattr(row, "qualified_symbol", None)
                if row is not None
                else None,
                "path": getattr(row, "path", None) if row is not None else None,
                "semantic_authority": bool(
                    getattr(hit, "semantic_authority", False)
                ),
            }
        )

    row_count = len(getattr(snapshot, "rows", ()) or ())
    return {
        "family": "vector_index",
        "applied": True,
        "available": True,
        "status": "indexed",
        "ok": row_count > 0,
        "grants_execution_authority": False,
        "role": "vector_index_intermediate_representation",
        "index_id": getattr(snapshot, "index_id", None)
        or getattr(snapshot, "snapshot_id", None),
        "forest_id": forest_id,
        "tree_id": tree_id,
        "dimensions": dimensions,
        "model_id": "ir-deterministic-fixture@1",
        "row_count": row_count,
        "query": {"seed": query_seed, "hit_count": len(hits)},
        "hits": hit_rows,
        "ast_index_id": ast_result.get("index_id"),
        "logic_applied": [
            "build_analysis_ast_index",
            "build_code_symbol_vector_index(deterministic_fixture_vectors)",
            "search_code_symbol_vector_index",
        ],
        "notes": [
            "Vector IR uses admitted deterministic fixture embeddings only.",
            "Hits are non-authoritative retrieval candidates (semantic_authority=false).",
            "No remote embedding service is contacted.",
        ],
        "_raw_snapshot": snapshot,
    }


def apply_knowledge_graph_logic(
    *,
    operation: str,
    contract_id: str = "",
    finding_kind: str = "",
    path: str = "",
    symbol: str = "",
    candidate: Mapping[str, Any] | None = None,
    ast_result: Mapping[str, Any] | None = None,
    domain: str = "agent_supervisor",
) -> dict[str, Any]:
    """Build semantic dependency knowledge-graph IR for a work decision."""
    from ..analysis.semantic_dependency_graph import (
        SemanticAuthority,
        SemanticEdge,
        SemanticEdgeKind,
        SemanticNode,
        SemanticNodeKind,
        SemanticProvenance,
        SemanticTrust,
        build_semantic_dependency_graph,
        compute_mandatory_closure,
    )

    op = operation or "unknown"
    op_slug = _slug(op)
    root_id = f"decision-root:ir:{op_slug}"
    decision_id = f"decision:work:{op_slug}"
    plan_id = f"plan:ir:{op_slug}"
    path = _default_path_for_op(op, path, domain=domain)
    sym = symbol or _op_symbols(op)[0]

    def node(
        node_id: str,
        kind: SemanticNodeKind,
        *,
        provenance: SemanticProvenance = SemanticProvenance.PLANNER,
        trust: SemanticTrust = SemanticTrust.REVIEWED,
        authority: SemanticAuthority = SemanticAuthority.CONTEXT_ONLY,
        record: Mapping[str, Any] | None = None,
    ) -> SemanticNode:
        return SemanticNode(
            node_id=node_id,
            kind=kind,
            root_id=root_id,
            provenance=provenance,
            trust=trust,
            authority=authority,
            version="1",
            source_root_id=root_id,
            provenance_id=f"prov:{node_id}",
            record=dict(record or {}),
        )

    def edge(
        source: str,
        target: str,
        kind: SemanticEdgeKind,
        *,
        mandatory: bool = True,
        provenance: SemanticProvenance = SemanticProvenance.PLANNER,
    ) -> SemanticEdge:
        return SemanticEdge(
            source=source,
            target=target,
            kind=kind,
            root_id=root_id,
            provenance=provenance,
            trust=SemanticTrust.REVIEWED,
            authority=SemanticAuthority.CONTEXT_ONLY,
            version="1",
            provenance_id=f"prov:{source}->{target}",
            source_root_id=root_id,
            mandatory=mandatory,
            record={},
        )

    nodes: list[SemanticNode] = [
        node(
            decision_id,
            SemanticNodeKind.DECISION,
            record={
                "operation": op,
                "contract_id": contract_id,
                "finding_kind": finding_kind,
            },
        ),
        node(
            plan_id,
            SemanticNodeKind.PLAN,
            record={"plan_id": plan_id, "operation": op},
        ),
        node(
            f"action:repair:{op_slug}",
            SemanticNodeKind.ACTION,
            record={"action": "repair", "operation": op},
        ),
        node(
            f"action:mediate:{op_slug}",
            SemanticNodeKind.ACTION,
            record={"action": "mcp_tools_call", "operation": op},
        ),
        node(
            f"effect:register:{op_slug}",
            SemanticNodeKind.EFFECT,
            record={"effect": "register_tool", "operation": op},
        ),
        node(
            f"tool:mcp-server:{op_slug}",
            SemanticNodeKind.TOOL,
            record={"tool": "mcp-server"},
        ),
        node(
            f"ast:{path}",
            SemanticNodeKind.AST,
            provenance=SemanticProvenance.AST,
            record={"path": path},
        ),
        node(
            f"symbol:{sym}",
            SemanticNodeKind.SYMBOL,
            provenance=SemanticProvenance.AST,
            record={"symbol": sym, "operation": op},
        ),
        node(
            f"obligation:reindex:{op_slug}",
            SemanticNodeKind.OBLIGATION,
            record={"obligation": "reindex_after_repair"},
        ),
    ]

    # Attach plan actions when candidate graph present
    for action in (candidate or {}).get("actions") or []:
        aid = str(action.get("action_id") or "")
        if not aid:
            continue
        nodes.append(
            node(
                aid,
                SemanticNodeKind.ACTION,
                record=dict(action),
            )
        )

    edges: list[SemanticEdge] = [
        edge(decision_id, plan_id, SemanticEdgeKind.DEPENDS_ON),
        edge(plan_id, f"action:repair:{op_slug}", SemanticEdgeKind.DEPENDS_ON),
        edge(
            f"action:repair:{op_slug}",
            f"action:mediate:{op_slug}",
            SemanticEdgeKind.REQUIRES,
        ),
        edge(
            f"action:mediate:{op_slug}",
            f"tool:mcp-server:{op_slug}",
            SemanticEdgeKind.DEPENDS_ON,
        ),
        edge(
            f"action:repair:{op_slug}",
            f"effect:register:{op_slug}",
            SemanticEdgeKind.AFFECTS,
        ),
        edge(
            f"action:repair:{op_slug}",
            f"ast:{path}",
            SemanticEdgeKind.SOURCED_FROM,
            provenance=SemanticProvenance.AST,
        ),
        edge(
            f"ast:{path}",
            f"symbol:{sym}",
            SemanticEdgeKind.SOURCED_FROM,
            provenance=SemanticProvenance.AST,
        ),
        edge(
            decision_id,
            f"obligation:reindex:{op_slug}",
            SemanticEdgeKind.REQUIRES,
        ),
    ]

    graph = build_semantic_dependency_graph(
        root_id=root_id,
        nodes=nodes,
        edges=edges,
    )
    closure = compute_mandatory_closure(graph, decision_id)
    closure_nodes = list(getattr(closure, "node_ids", ()) or ())
    closure_edges = list(getattr(closure, "edge_ids", ()) or ())

    return {
        "family": "knowledge_graph",
        "applied": True,
        "available": True,
        "status": "built",
        "ok": len(getattr(graph, "nodes", ()) or ()) > 0,
        "grants_execution_authority": False,
        "role": "knowledge_graph_intermediate_representation",
        "root_id": root_id,
        "decision_id": decision_id,
        "node_count": len(getattr(graph, "nodes", ()) or ()),
        "edge_count": len(getattr(graph, "edges", ()) or ()),
        "closure": {
            "node_ids": closure_nodes[:32],
            "edge_ids": closure_edges[:32],
            "node_count": len(closure_nodes),
            "edge_count": len(closure_edges),
        },
        "ast_binding": {
            "path": path,
            "symbol": sym,
            "ast_index_id": (ast_result or {}).get("index_id"),
        },
        "logic_applied": [
            "project_decision_action_ast_nodes",
            "build_semantic_dependency_graph",
            "compute_mandatory_closure",
        ],
        "notes": [
            "Knowledge-graph IR binds decision/plan/action/AST/symbol nodes.",
            "Mandatory closure is context for planner/doctor, not execution authority.",
            "Intent/Legal/Security IR nodes can be projected via normalized_ir channel.",
        ],
        "_raw_graph": graph,
        "_raw_closure": closure,
    }


def apply_structural_logic(
    *,
    operation: str,
    contract_id: str = "",
    finding_kind: str = "",
    path: str = "",
    symbol: str = "",
    candidate: Mapping[str, Any] | None = None,
    families: Sequence[str] | None = None,
    domain: str = "agent_supervisor",
) -> dict[str, Any]:
    """Apply AST + knowledge graph + vector index logic to work-surface IR."""
    want = tuple(families or DEFAULT_STRUCTURAL_FAMILIES)
    out: dict[str, Any] = {}
    errors: list[str] = []
    ast_result: dict[str, Any] | None = None

    if "ast" in want:
        try:
            ast_result = apply_ast_logic(
                operation=operation,
                contract_id=contract_id,
                finding_kind=finding_kind,
                path=path,
                symbol=symbol,
                domain=domain,
            )
            out["ast"] = ast_result
        except Exception as exc:  # noqa: BLE001
            errors.append(f"ast: {type(exc).__name__}: {exc}")
            out["ast"] = {
                "family": "ast",
                "applied": False,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "grants_execution_authority": False,
            }

    if "vector_index" in want:
        try:
            out["vector_index"] = apply_vector_index_logic(
                operation=operation,
                contract_id=contract_id,
                finding_kind=finding_kind,
                path=path,
                symbol=symbol,
                ast_result=ast_result,
                domain=domain,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"vector_index: {type(exc).__name__}: {exc}")
            out["vector_index"] = {
                "family": "vector_index",
                "applied": False,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "grants_execution_authority": False,
            }

    if "knowledge_graph" in want:
        try:
            out["knowledge_graph"] = apply_knowledge_graph_logic(
                operation=operation,
                contract_id=contract_id,
                finding_kind=finding_kind,
                path=path,
                symbol=symbol,
                candidate=candidate,
                ast_result=ast_result,
                domain=domain,
            )
        except Exception as exc:  # noqa: BLE001
            errors.append(f"knowledge_graph: {type(exc).__name__}: {exc}")
            out["knowledge_graph"] = {
                "family": "knowledge_graph",
                "applied": False,
                "ok": False,
                "error": f"{type(exc).__name__}: {exc}",
                "grants_execution_authority": False,
            }

    family_ok = {name: bool(doc.get("ok")) for name, doc in out.items()}
    public = {
        name: {k: v for k, v in doc.items() if not k.startswith("_raw")}
        for name, doc in out.items()
    }
    return {
        "interface": STRUCTURAL_IR_INTERFACE,
        "families": public,
        "family_ok": family_ok,
        "passed": bool(family_ok) and all(family_ok.values()),
        "errors": errors,
        "gates": {
            "logic_applied_to_structural_ir": any(
                bool(d.get("applied")) for d in public.values()
            ),
            "no_false_execution_grants": all(
                not bool(d.get("grants_execution_authority")) for d in public.values()
            ),
        },
    }


__all__ = [
    "DEFAULT_STRUCTURAL_FAMILIES",
    "DEFAULT_VECTOR_DIMS",
    "STRUCTURAL_IR_INTERFACE",
    "apply_ast_logic",
    "apply_knowledge_graph_logic",
    "apply_structural_logic",
    "apply_vector_index_logic",
    "project_ast_blob_record",
]
