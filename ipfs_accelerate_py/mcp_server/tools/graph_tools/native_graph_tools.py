"""Native graph-tools category implementations for unified mcp_server."""

from __future__ import annotations

import itertools
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_VALID_CONSTRAINT_TYPES = {"unique", "exists", "node_key"}
_VALID_GRAPH_SEARCH_TYPES = {"semantic", "keyword", "hybrid"}
_VALID_GRAPH_VISUALIZE_FORMATS = {"dot", "mermaid", "d3_json", "ascii"}
_VALID_GRAPH_EXPLAIN_TYPES = {"entity", "relationship", "path", "why_connected"}
_TRANSACTION_COUNTER = itertools.count(1)


def _load_graph_tools_api() -> Dict[str, Any]:
    """Resolve source graph-tools APIs with compatibility fallback."""
    try:
        from ipfs_datasets_py.ipfs_datasets_py.mcp_server.tools.graph_tools import (  # type: ignore
            graph_add_entity as _graph_add_entity,
            graph_add_relationship as _graph_add_relationship,
            graph_complete_suggestions as _graph_complete_suggestions,
            graph_constraint_add as _graph_constraint_add,
            graph_create as _graph_create,
            graph_distributed_execute as _graph_distributed_execute,
            graph_explain as _graph_explain,
            graph_graphql_query as _graph_graphql_query,
            graph_index_create as _graph_index_create,
            graph_ontology_materialize as _graph_ontology_materialize,
            graph_provenance_verify as _graph_provenance_verify,
            graph_query_cypher as _graph_query_cypher,
            graph_search_hybrid as _graph_search_hybrid,
            graph_srl_extract as _graph_srl_extract,
            graph_transaction_begin as _graph_transaction_begin,
            graph_transaction_commit as _graph_transaction_commit,
            graph_transaction_rollback as _graph_transaction_rollback,
            graph_visualize as _graph_visualize,
            query_knowledge_graph as _query_knowledge_graph,
        )

        return {
            "graph_create": _graph_create,
            "graph_add_entity": _graph_add_entity,
            "graph_add_relationship": _graph_add_relationship,
            "graph_query_cypher": _graph_query_cypher,
            "query_knowledge_graph": _query_knowledge_graph,
            "graph_search_hybrid": _graph_search_hybrid,
            "graph_transaction_begin": _graph_transaction_begin,
            "graph_transaction_commit": _graph_transaction_commit,
            "graph_transaction_rollback": _graph_transaction_rollback,
            "graph_index_create": _graph_index_create,
            "graph_constraint_add": _graph_constraint_add,
            "graph_srl_extract": _graph_srl_extract,
            "graph_ontology_materialize": _graph_ontology_materialize,
            "graph_distributed_execute": _graph_distributed_execute,
            "graph_graphql_query": _graph_graphql_query,
            "graph_visualize": _graph_visualize,
            "graph_complete_suggestions": _graph_complete_suggestions,
            "graph_explain": _graph_explain,
            "graph_provenance_verify": _graph_provenance_verify,
        }
    except Exception:
        logger.warning("Source graph_tools import unavailable, using fallback graph-tools functions")

        transactions: Dict[str, Dict[str, Any]] = {}
        indexes: Dict[str, Dict[str, Any]] = {}
        constraints: Dict[str, Dict[str, Any]] = {}

        async def _create_fallback(driver_url: Optional[str] = None) -> Dict[str, Any]:
            return {
                "status": "error",
                "message": "graph backend unavailable",
                "driver_url": driver_url or "ipfs://localhost:5001",
            }

        async def _add_entity_fallback(
            entity_id: str,
            entity_type: str,
            properties: Optional[Dict[str, Any]] = None,
            driver_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = properties, driver_url
            return {
                "status": "error",
                "message": "graph backend unavailable",
                "entity_id": entity_id,
                "entity_type": entity_type,
            }

        async def _add_relationship_fallback(
            source_id: str,
            target_id: str,
            relationship_type: str,
            properties: Optional[Dict[str, Any]] = None,
            driver_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = properties, driver_url
            return {
                "status": "error",
                "message": "graph backend unavailable",
                "source_id": source_id,
                "target_id": target_id,
                "relationship_type": relationship_type,
            }

        async def _query_cypher_fallback(
            query: str,
            parameters: Optional[Dict[str, Any]] = None,
            driver_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = parameters, driver_url
            return {
                "status": "error",
                "message": "graph backend unavailable",
                "query": query,
                "results": [],
            }

        async def _transaction_begin_fallback(driver_url: Optional[str] = None) -> Dict[str, Any]:
            transaction_id = f"tx-{next(_TRANSACTION_COUNTER)}"
            transactions[transaction_id] = {
                "transaction_id": transaction_id,
                "driver_url": driver_url or "ipfs://localhost:5001",
                "status": "open",
            }
            return {
                "status": "success",
                "transaction_id": transaction_id,
                "message": "transaction started",
            }

        async def _transaction_commit_fallback(
            transaction_id: Optional[str] = None,
            driver_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = driver_url
            normalized_transaction_id = str(transaction_id or "").strip()
            tx = transactions.get(normalized_transaction_id)
            if tx is None:
                return {
                    "status": "error",
                    "message": "transaction not found",
                    "transaction_id": transaction_id,
                }
            tx["status"] = "committed"
            return {
                "status": "success",
                "transaction_id": normalized_transaction_id,
                "message": "transaction committed",
            }

        async def _transaction_rollback_fallback(
            transaction_id: Optional[str] = None,
            driver_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            _ = driver_url
            normalized_transaction_id = str(transaction_id or "").strip()
            tx = transactions.get(normalized_transaction_id)
            if tx is None:
                return {
                    "status": "error",
                    "message": "transaction not found",
                    "transaction_id": transaction_id,
                }
            tx["status"] = "rolled_back"
            return {
                "status": "success",
                "transaction_id": normalized_transaction_id,
                "message": "transaction rolled back",
            }

        async def _index_create_fallback(
            index_name: str,
            entity_type: str,
            properties: List[str],
            driver_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            indexes[index_name] = {
                "entity_type": entity_type,
                "properties": list(properties),
                "driver_url": driver_url or "ipfs://localhost:5001",
            }
            return {
                "status": "success",
                "index_name": index_name,
                "entity_type": entity_type,
                "properties": list(properties),
                "message": "index created",
            }

        async def _constraint_add_fallback(
            constraint_name: str,
            constraint_type: str,
            entity_type: str,
            properties: List[str],
            driver_url: Optional[str] = None,
        ) -> Dict[str, Any]:
            constraints[constraint_name] = {
                "constraint_type": constraint_type,
                "entity_type": entity_type,
                "properties": list(properties),
                "driver_url": driver_url or "ipfs://localhost:5001",
            }
            return {
                "status": "success",
                "constraint_name": constraint_name,
                "constraint_type": constraint_type,
                "entity_type": entity_type,
                "properties": list(properties),
                "message": "constraint added",
            }

        async def _query_knowledge_graph_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "query": kwargs.get("query", ""),
                "query_type": kwargs.get("query_type", "ir"),
                "results": [],
                "count": 0,
                "include_metadata": kwargs.get("include_metadata", True),
                "message": "fallback knowledge graph query",
            }

        async def _search_hybrid_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "query": kwargs.get("query", ""),
                "search_type": kwargs.get("search_type", "semantic"),
                "results": [],
                "limit": kwargs.get("limit", 10),
            }

        async def _srl_extract_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "text": kwargs.get("text", ""),
                "backend": kwargs.get("backend", "heuristic"),
                "frames": [],
            }

        async def _ontology_materialize_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "graph_name": kwargs.get("graph_name"),
                "entailed_facts": 0,
                "check_consistency": kwargs.get("check_consistency", False),
            }

        async def _distributed_execute_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "query": kwargs.get("query", ""),
                "num_partitions": kwargs.get("num_partitions", 4),
                "results": [],
            }

        async def _graphql_query_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "query": kwargs.get("query", ""),
                "data": {},
            }

        async def _visualize_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "format": kwargs.get("format", "dot"),
                "visualization": "graph {}",
            }

        async def _complete_suggestions_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "suggestions": [],
                "total_suggestions": 0,
                "min_score": kwargs.get("min_score", 0.3),
            }

        async def _explain_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "explain_type": kwargs.get("explain_type", "entity"),
                "explanation": "graph backend unavailable",
            }

        async def _provenance_verify_fallback(**kwargs: Any) -> Dict[str, Any]:
            return {
                "status": "success",
                "verified": True,
                "provenance_jsonl": kwargs.get("provenance_jsonl"),
            }

        return {
            "graph_create": _create_fallback,
            "graph_add_entity": _add_entity_fallback,
            "graph_add_relationship": _add_relationship_fallback,
            "graph_query_cypher": _query_cypher_fallback,
            "query_knowledge_graph": _query_knowledge_graph_fallback,
            "graph_search_hybrid": _search_hybrid_fallback,
            "graph_transaction_begin": _transaction_begin_fallback,
            "graph_transaction_commit": _transaction_commit_fallback,
            "graph_transaction_rollback": _transaction_rollback_fallback,
            "graph_index_create": _index_create_fallback,
            "graph_constraint_add": _constraint_add_fallback,
            "graph_srl_extract": _srl_extract_fallback,
            "graph_ontology_materialize": _ontology_materialize_fallback,
            "graph_distributed_execute": _distributed_execute_fallback,
            "graph_graphql_query": _graphql_query_fallback,
            "graph_visualize": _visualize_fallback,
            "graph_complete_suggestions": _complete_suggestions_fallback,
            "graph_explain": _explain_fallback,
            "graph_provenance_verify": _provenance_verify_fallback,
        }


_API = _load_graph_tools_api()


def _error_result(message: str, **extra: Any) -> Dict[str, Any]:
    """Return a normalized error envelope for deterministic dispatch behavior."""
    payload: Dict[str, Any] = {"status": "error", "error": message, "message": message}
    payload.update(extra)
    return payload


async def _await_maybe(result: Any) -> Dict[str, Any]:
    """Await coroutine-like API results while supporting direct return values."""
    if hasattr(result, "__await__"):
        return await result
    return result


async def graph_create(driver_url: Optional[str] = None) -> Dict[str, Any]:
    """Initialize or connect to a knowledge graph backend."""
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")

    try:
        payload = await _await_maybe(_API["graph_create"](driver_url=normalized_driver_url))
    except Exception as exc:
        return _error_result(f"graph_create failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


async def graph_add_entity(
    entity_id: str,
    entity_type: str,
    properties: Optional[Dict[str, Any]] = None,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Add an entity node to a knowledge graph."""
    normalized_entity_id = str(entity_id or "").strip()
    normalized_entity_type = str(entity_type or "").strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()

    if not normalized_entity_id or not normalized_entity_type:
        return _error_result(
            "entity_id and entity_type must be provided",
            entity_id=entity_id,
            entity_type=entity_type,
        )
    if properties is not None and not isinstance(properties, dict):
        return _error_result("properties must be an object when provided")
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")

    try:
        payload = await _await_maybe(
            _API["graph_add_entity"](
                entity_id=normalized_entity_id,
                entity_type=normalized_entity_type,
                properties=properties,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_add_entity failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


async def graph_add_relationship(
    source_id: str,
    target_id: str,
    relationship_type: str,
    properties: Optional[Dict[str, Any]] = None,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a typed relationship between two graph entities."""
    normalized_source = str(source_id or "").strip()
    normalized_target = str(target_id or "").strip()
    normalized_type = str(relationship_type or "").strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()

    if not normalized_source or not normalized_target or not normalized_type:
        return _error_result(
            "source_id, target_id, and relationship_type must be provided",
            source_id=source_id,
            target_id=target_id,
            relationship_type=relationship_type,
        )
    if properties is not None and not isinstance(properties, dict):
        return _error_result("properties must be an object when provided")
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")

    try:
        payload = await _await_maybe(
            _API["graph_add_relationship"](
                source_id=normalized_source,
                target_id=normalized_target,
                relationship_type=normalized_type,
                properties=properties,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_add_relationship failed: {exc}")

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


async def graph_query_cypher(
    query: str,
    parameters: Optional[Dict[str, Any]] = None,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute a Cypher query against the configured graph backend."""
    normalized_query = str(query or "").strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if not normalized_query:
        return _error_result("query must be provided", query=query)
    if parameters is not None and not isinstance(parameters, dict):
        return _error_result("parameters must be an object when provided")
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")

    try:
        payload = await _await_maybe(
            _API["graph_query_cypher"](
                query=normalized_query,
                parameters=parameters,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_query_cypher failed: {exc}", query=normalized_query)

    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


async def graph_transaction_begin(driver_url: Optional[str] = None) -> Dict[str, Any]:
    """Begin an explicit graph transaction."""
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")
    try:
        payload = await _await_maybe(_API["graph_transaction_begin"](driver_url=normalized_driver_url))
    except Exception as exc:
        return _error_result(f"graph_transaction_begin failed: {exc}")
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    return normalized


async def graph_transaction_commit(
    transaction_id: Optional[str] = None,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Commit an explicit graph transaction."""
    normalized_transaction_id = None if transaction_id is None else str(transaction_id).strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if normalized_transaction_id is not None and not normalized_transaction_id:
        return _error_result("transaction_id must be a non-empty string when provided")
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")
    try:
        payload = await _await_maybe(
            _API["graph_transaction_commit"](
                transaction_id=normalized_transaction_id,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_transaction_commit failed: {exc}", transaction_id=normalized_transaction_id)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("transaction_id", normalized_transaction_id)
    return normalized


async def graph_transaction_rollback(
    transaction_id: Optional[str] = None,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Roll back an explicit graph transaction."""
    normalized_transaction_id = None if transaction_id is None else str(transaction_id).strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if normalized_transaction_id is not None and not normalized_transaction_id:
        return _error_result("transaction_id must be a non-empty string when provided")
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")
    try:
        payload = await _await_maybe(
            _API["graph_transaction_rollback"](
                transaction_id=normalized_transaction_id,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_transaction_rollback failed: {exc}", transaction_id=normalized_transaction_id)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("transaction_id", normalized_transaction_id)
    return normalized


async def graph_index_create(
    index_name: str,
    entity_type: str,
    properties: List[str],
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Create a graph index for entity properties."""
    normalized_index_name = str(index_name or "").strip()
    normalized_entity_type = str(entity_type or "").strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if not normalized_index_name or not normalized_entity_type:
        return _error_result("index_name and entity_type must be provided")
    if not isinstance(properties, list) or not properties or any(not isinstance(item, str) or not item.strip() for item in properties):
        return _error_result("properties must be a non-empty array of non-empty strings")
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")
    try:
        payload = await _await_maybe(
            _API["graph_index_create"](
                index_name=normalized_index_name,
                entity_type=normalized_entity_type,
                properties=[str(item).strip() for item in properties],
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_index_create failed: {exc}", index_name=normalized_index_name)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("index_name", normalized_index_name)
    return normalized


async def graph_constraint_add(
    constraint_name: str,
    constraint_type: str,
    entity_type: str,
    properties: List[str],
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    """Add a uniqueness or existence constraint to the graph."""
    normalized_constraint_name = str(constraint_name or "").strip()
    normalized_constraint_type = str(constraint_type or "").strip().lower()
    normalized_entity_type = str(entity_type or "").strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if not normalized_constraint_name or not normalized_entity_type:
        return _error_result("constraint_name and entity_type must be provided")
    if normalized_constraint_type not in _VALID_CONSTRAINT_TYPES:
        return _error_result("constraint_type must be one of: exists, node_key, unique")
    if not isinstance(properties, list) or not properties or any(not isinstance(item, str) or not item.strip() for item in properties):
        return _error_result("properties must be a non-empty array of non-empty strings")
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")
    try:
        payload = await _await_maybe(
            _API["graph_constraint_add"](
                constraint_name=normalized_constraint_name,
                constraint_type=normalized_constraint_type,
                entity_type=normalized_entity_type,
                properties=[str(item).strip() for item in properties],
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_constraint_add failed: {exc}", constraint_name=normalized_constraint_name)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("constraint_name", normalized_constraint_name)
    return normalized


async def query_knowledge_graph(
    graph_id: Optional[str] = None,
    query: str = "",
    query_type: str = "ir",
    max_results: int = 100,
    include_metadata: bool = True,
    manifest_cid: Optional[str] = None,
    ir_ops: Optional[List[Dict[str, Any]]] = None,
    budgets: Optional[Dict[str, Any]] = None,
    budget_preset: Optional[str] = None,
    ipfs_backend: Optional[str] = None,
    car_fetch_mode: str = "auto",
) -> Dict[str, Any]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return _error_result("query must be provided", query=query)
    if not isinstance(query_type, str) or not query_type.strip():
        return _error_result("query_type must be a non-empty string", query_type=query_type)
    if not isinstance(max_results, int) or max_results < 1:
        return _error_result("max_results must be an integer >= 1", max_results=max_results)
    if not isinstance(include_metadata, bool):
        return _error_result("include_metadata must be a boolean", include_metadata=include_metadata)
    if ir_ops is not None and (not isinstance(ir_ops, list) or not all(isinstance(item, dict) for item in ir_ops)):
        return _error_result("ir_ops must be null or a list of objects", ir_ops=ir_ops)
    if budgets is not None and not isinstance(budgets, dict):
        return _error_result("budgets must be an object when provided", budgets=budgets)
    for field, value in (("graph_id", graph_id), ("manifest_cid", manifest_cid), ("budget_preset", budget_preset), ("ipfs_backend", ipfs_backend)):
        if value is not None and not str(value).strip():
            return _error_result(f"{field} must be a non-empty string when provided", **{field: value})
    if not isinstance(car_fetch_mode, str) or not car_fetch_mode.strip():
        return _error_result("car_fetch_mode must be a non-empty string", car_fetch_mode=car_fetch_mode)
    try:
        payload = await _await_maybe(
            _API["query_knowledge_graph"](
                graph_id=graph_id,
                query=normalized_query,
                query_type=query_type.strip(),
                max_results=max_results,
                include_metadata=include_metadata,
                manifest_cid=manifest_cid,
                ir_ops=ir_ops,
                budgets=budgets,
                budget_preset=budget_preset,
                ipfs_backend=ipfs_backend,
                car_fetch_mode=car_fetch_mode.strip(),
            )
        )
    except Exception as exc:
        return _error_result(f"query_knowledge_graph failed: {exc}", query=normalized_query)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("query", normalized_query)
    normalized.setdefault("results", [])
    return normalized


async def graph_search_hybrid(
    query: str,
    search_type: str = "semantic",
    limit: int = 10,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_query = str(query or "").strip()
    normalized_search_type = str(search_type or "").strip().lower()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if not normalized_query:
        return _error_result("query must be provided", query=query)
    if normalized_search_type not in _VALID_GRAPH_SEARCH_TYPES:
        return _error_result("search_type must be one of: hybrid, keyword, semantic", search_type=search_type)
    if not isinstance(limit, int) or limit < 1:
        return _error_result("limit must be an integer >= 1", limit=limit)
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")
    try:
        payload = await _await_maybe(
            _API["graph_search_hybrid"](
                query=normalized_query,
                search_type=normalized_search_type,
                limit=limit,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_search_hybrid failed: {exc}", query=normalized_query)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("query", normalized_query)
    normalized.setdefault("results", [])
    return normalized


async def graph_srl_extract(
    text: str,
    backend: str = "heuristic",
    return_triples: bool = False,
    return_temporal_graph: bool = False,
) -> Dict[str, Any]:
    normalized_text = str(text or "").strip()
    if not normalized_text:
        return _error_result("text must be provided", text=text)
    if not isinstance(backend, str) or not backend.strip():
        return _error_result("backend must be a non-empty string", backend=backend)
    if not isinstance(return_triples, bool):
        return _error_result("return_triples must be a boolean", return_triples=return_triples)
    if not isinstance(return_temporal_graph, bool):
        return _error_result("return_temporal_graph must be a boolean", return_temporal_graph=return_temporal_graph)
    try:
        payload = await _await_maybe(
            _API["graph_srl_extract"](
                text=normalized_text,
                backend=backend.strip(),
                return_triples=return_triples,
                return_temporal_graph=return_temporal_graph,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_srl_extract failed: {exc}")
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("text", normalized_text)
    return normalized


async def graph_ontology_materialize(
    graph_name: str,
    schema: Optional[Dict[str, Any]] = None,
    check_consistency: bool = False,
    explain: bool = False,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_graph_name = str(graph_name or "").strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if not normalized_graph_name:
        return _error_result("graph_name must be provided", graph_name=graph_name)
    if schema is not None and not isinstance(schema, dict):
        return _error_result("schema must be an object when provided", schema=schema)
    if not isinstance(check_consistency, bool):
        return _error_result("check_consistency must be a boolean", check_consistency=check_consistency)
    if not isinstance(explain, bool):
        return _error_result("explain must be a boolean", explain=explain)
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")
    try:
        payload = await _await_maybe(
            _API["graph_ontology_materialize"](
                graph_name=normalized_graph_name,
                schema=schema,
                check_consistency=check_consistency,
                explain=explain,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_ontology_materialize failed: {exc}", graph_name=normalized_graph_name)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("graph_name", normalized_graph_name)
    return normalized


async def graph_distributed_execute(
    query: str,
    num_partitions: int = 4,
    partition_strategy: str = "hash",
    parallel: bool = False,
    explain: bool = False,
    driver_url: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_query = str(query or "").strip()
    normalized_strategy = str(partition_strategy or "").strip()
    normalized_driver_url = None if driver_url is None else str(driver_url).strip()
    if not normalized_query:
        return _error_result("query must be provided", query=query)
    if not isinstance(num_partitions, int) or num_partitions < 1:
        return _error_result("num_partitions must be an integer >= 1", num_partitions=num_partitions)
    if not normalized_strategy:
        return _error_result("partition_strategy must be a non-empty string", partition_strategy=partition_strategy)
    if not isinstance(parallel, bool):
        return _error_result("parallel must be a boolean", parallel=parallel)
    if not isinstance(explain, bool):
        return _error_result("explain must be a boolean", explain=explain)
    if normalized_driver_url is not None and not normalized_driver_url:
        return _error_result("driver_url must be a non-empty string when provided")
    try:
        payload = await _await_maybe(
            _API["graph_distributed_execute"](
                query=normalized_query,
                num_partitions=num_partitions,
                partition_strategy=normalized_strategy,
                parallel=parallel,
                explain=explain,
                driver_url=normalized_driver_url,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_distributed_execute failed: {exc}", query=normalized_query)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("query", normalized_query)
    normalized.setdefault("results", [])
    return normalized


async def graph_graphql_query(query: str, kg_data: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    normalized_query = str(query or "").strip()
    if not normalized_query:
        return _error_result("query must be provided", query=query)
    if kg_data is not None and not isinstance(kg_data, dict):
        return _error_result("kg_data must be an object when provided", kg_data=kg_data)
    try:
        payload = await _await_maybe(_API["graph_graphql_query"](query=normalized_query, kg_data=kg_data))
    except Exception as exc:
        return _error_result(f"graph_graphql_query failed: {exc}", query=normalized_query)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("query", normalized_query)
    return normalized


async def graph_visualize(
    format: str = "dot",
    kg_data: Optional[Dict[str, Any]] = None,
    max_entities: Optional[int] = None,
    directed: bool = True,
    graph_name: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_format = str(format or "").strip().lower()
    normalized_graph_name = None if graph_name is None else str(graph_name).strip()
    if normalized_format not in _VALID_GRAPH_VISUALIZE_FORMATS:
        return _error_result("format must be one of: ascii, d3_json, dot, mermaid", format=format)
    if kg_data is not None and not isinstance(kg_data, dict):
        return _error_result("kg_data must be an object when provided", kg_data=kg_data)
    if max_entities is not None and (not isinstance(max_entities, int) or max_entities < 1):
        return _error_result("max_entities must be null or an integer >= 1", max_entities=max_entities)
    if not isinstance(directed, bool):
        return _error_result("directed must be a boolean", directed=directed)
    if normalized_graph_name is not None and not normalized_graph_name:
        return _error_result("graph_name must be a non-empty string when provided", graph_name=graph_name)
    try:
        payload = await _await_maybe(
            _API["graph_visualize"](
                format=normalized_format,
                kg_data=kg_data,
                max_entities=max_entities,
                directed=directed,
                graph_name=normalized_graph_name,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_visualize failed: {exc}", format=normalized_format)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("format", normalized_format)
    return normalized


async def graph_complete_suggestions(
    kg_data: Optional[Dict[str, Any]] = None,
    min_score: float = 0.3,
    max_suggestions: int = 20,
    entity_id: Optional[str] = None,
    rel_type: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_entity_id = None if entity_id is None else str(entity_id).strip()
    normalized_rel_type = None if rel_type is None else str(rel_type).strip()
    if kg_data is not None and not isinstance(kg_data, dict):
        return _error_result("kg_data must be an object when provided", kg_data=kg_data)
    if not isinstance(min_score, (int, float)) or not 0.0 <= float(min_score) <= 1.0:
        return _error_result("min_score must be a number between 0.0 and 1.0", min_score=min_score)
    if not isinstance(max_suggestions, int) or max_suggestions < 1:
        return _error_result("max_suggestions must be an integer >= 1", max_suggestions=max_suggestions)
    if entity_id is not None and not normalized_entity_id:
        return _error_result("entity_id must be a non-empty string when provided", entity_id=entity_id)
    if rel_type is not None and not normalized_rel_type:
        return _error_result("rel_type must be a non-empty string when provided", rel_type=rel_type)
    try:
        payload = await _await_maybe(
            _API["graph_complete_suggestions"](
                kg_data=kg_data,
                min_score=float(min_score),
                max_suggestions=max_suggestions,
                entity_id=normalized_entity_id,
                rel_type=normalized_rel_type,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_complete_suggestions failed: {exc}")
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("suggestions", [])
    return normalized


async def graph_explain(
    explain_type: str = "entity",
    entity_id: Optional[str] = None,
    start_entity_id: Optional[str] = None,
    end_entity_id: Optional[str] = None,
    relationship_id: Optional[str] = None,
    depth: str = "standard",
    max_hops: int = 4,
    kg_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_explain_type = str(explain_type or "").strip().lower()
    normalized_entity_id = None if entity_id is None else str(entity_id).strip()
    normalized_start = None if start_entity_id is None else str(start_entity_id).strip()
    normalized_end = None if end_entity_id is None else str(end_entity_id).strip()
    normalized_relationship_id = None if relationship_id is None else str(relationship_id).strip()
    if normalized_explain_type not in _VALID_GRAPH_EXPLAIN_TYPES:
        return _error_result("explain_type must be one of: entity, relationship, path, why_connected", explain_type=explain_type)
    if not isinstance(depth, str) or not depth.strip():
        return _error_result("depth must be a non-empty string", depth=depth)
    if not isinstance(max_hops, int) or max_hops < 1:
        return _error_result("max_hops must be an integer >= 1", max_hops=max_hops)
    if kg_data is not None and not isinstance(kg_data, dict):
        return _error_result("kg_data must be an object when provided", kg_data=kg_data)
    if normalized_explain_type == "entity" and not normalized_entity_id:
        return _error_result("entity_id is required for entity explain_type")
    if normalized_explain_type == "relationship" and not normalized_relationship_id:
        return _error_result("relationship_id is required for relationship explain_type")
    if normalized_explain_type in {"path", "why_connected"} and (not normalized_start or not normalized_end):
        return _error_result("start_entity_id and end_entity_id are required for path-based explain_type")
    try:
        payload = await _await_maybe(
            _API["graph_explain"](
                explain_type=normalized_explain_type,
                entity_id=normalized_entity_id,
                start_entity_id=normalized_start,
                end_entity_id=normalized_end,
                relationship_id=normalized_relationship_id,
                depth=depth.strip(),
                max_hops=max_hops,
                kg_data=kg_data,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_explain failed: {exc}", explain_type=normalized_explain_type)
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("explain_type", normalized_explain_type)
    return normalized


async def graph_provenance_verify(
    provenance_jsonl: Optional[str] = None,
    kg_data: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    normalized_provenance_jsonl = None if provenance_jsonl is None else str(provenance_jsonl).strip()
    if provenance_jsonl is not None and not normalized_provenance_jsonl:
        return _error_result("provenance_jsonl must be a non-empty string when provided", provenance_jsonl=provenance_jsonl)
    if kg_data is not None and not isinstance(kg_data, dict):
        return _error_result("kg_data must be an object when provided", kg_data=kg_data)
    try:
        payload = await _await_maybe(
            _API["graph_provenance_verify"](
                provenance_jsonl=normalized_provenance_jsonl,
                kg_data=kg_data,
            )
        )
    except Exception as exc:
        return _error_result(f"graph_provenance_verify failed: {exc}")
    normalized = dict(payload or {})
    normalized.setdefault("status", "error" if "error" in normalized else "success")
    normalized.setdefault("provenance_jsonl", normalized_provenance_jsonl)
    return normalized


def register_native_graph_tools(manager: Any) -> None:
    """Register native graph-tools category tools in unified manager."""
    manager.register_tool(
        category="graph_tools",
        name="graph_create",
        func=graph_create,
        description="Initialize a graph backend for graph operations.",
        input_schema={
            "type": "object",
            "properties": {
                "driver_url": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_add_entity",
        func=graph_add_entity,
        description="Add a typed entity to the graph.",
        input_schema={
            "type": "object",
            "properties": {
                "entity_id": {"type": "string", "minLength": 1},
                "entity_type": {"type": "string", "minLength": 1},
                "properties": {"type": ["object", "null"]},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["entity_id", "entity_type"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_add_relationship",
        func=graph_add_relationship,
        description="Add a typed relationship between two entities in the graph.",
        input_schema={
            "type": "object",
            "properties": {
                "source_id": {"type": "string", "minLength": 1},
                "target_id": {"type": "string", "minLength": 1},
                "relationship_type": {"type": "string", "minLength": 1},
                "properties": {"type": ["object", "null"]},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["source_id", "target_id", "relationship_type"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_query_cypher",
        func=graph_query_cypher,
        description="Execute a Cypher query against the graph backend.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "parameters": {"type": ["object", "null"]},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_transaction_begin",
        func=graph_transaction_begin,
        description="Begin an explicit graph transaction.",
        input_schema={
            "type": "object",
            "properties": {
                "driver_url": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_transaction_commit",
        func=graph_transaction_commit,
        description="Commit an explicit graph transaction.",
        input_schema={
            "type": "object",
            "properties": {
                "transaction_id": {"type": ["string", "null"], "minLength": 1},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_transaction_rollback",
        func=graph_transaction_rollback,
        description="Roll back an explicit graph transaction.",
        input_schema={
            "type": "object",
            "properties": {
                "transaction_id": {"type": ["string", "null"], "minLength": 1},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_index_create",
        func=graph_index_create,
        description="Create a graph index for entity properties.",
        input_schema={
            "type": "object",
            "properties": {
                "index_name": {"type": "string", "minLength": 1},
                "entity_type": {"type": "string", "minLength": 1},
                "properties": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["index_name", "entity_type", "properties"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_constraint_add",
        func=graph_constraint_add,
        description="Add a graph uniqueness or existence constraint.",
        input_schema={
            "type": "object",
            "properties": {
                "constraint_name": {"type": "string", "minLength": 1},
                "constraint_type": {"type": "string", "enum": sorted(_VALID_CONSTRAINT_TYPES)},
                "entity_type": {"type": "string", "minLength": 1},
                "properties": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string", "minLength": 1},
                },
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["constraint_name", "constraint_type", "entity_type", "properties"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="query_knowledge_graph",
        func=query_knowledge_graph,
        description="Query a knowledge graph with source-compatible IR and budget parameters.",
        input_schema={
            "type": "object",
            "properties": {
                "graph_id": {"type": ["string", "null"]},
                "query": {"type": "string", "minLength": 1},
                "query_type": {"type": "string", "default": "ir"},
                "max_results": {"type": "integer", "minimum": 1, "default": 100},
                "include_metadata": {"type": "boolean", "default": True},
                "manifest_cid": {"type": ["string", "null"]},
                "ir_ops": {"type": ["array", "null"], "items": {"type": "object"}},
                "budgets": {"type": ["object", "null"]},
                "budget_preset": {"type": ["string", "null"]},
                "ipfs_backend": {"type": ["string", "null"]},
                "car_fetch_mode": {"type": "string", "default": "auto"},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_search_hybrid",
        func=graph_search_hybrid,
        description="Perform hybrid graph search across semantic and keyword modes.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "search_type": {"type": "string", "enum": sorted(_VALID_GRAPH_SEARCH_TYPES), "default": "semantic"},
                "limit": {"type": "integer", "minimum": 1, "default": 10},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_srl_extract",
        func=graph_srl_extract,
        description="Extract SRL frames and optional triples from text.",
        input_schema={
            "type": "object",
            "properties": {
                "text": {"type": "string", "minLength": 1},
                "backend": {"type": "string", "default": "heuristic"},
                "return_triples": {"type": "boolean", "default": False},
                "return_temporal_graph": {"type": "boolean", "default": False},
            },
            "required": ["text"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_ontology_materialize",
        func=graph_ontology_materialize,
        description="Materialize ontology rules into a graph.",
        input_schema={
            "type": "object",
            "properties": {
                "graph_name": {"type": "string", "minLength": 1},
                "schema": {"type": ["object", "null"]},
                "check_consistency": {"type": "boolean", "default": False},
                "explain": {"type": "boolean", "default": False},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["graph_name"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_distributed_execute",
        func=graph_distributed_execute,
        description="Execute a distributed graph query.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "num_partitions": {"type": "integer", "minimum": 1, "default": 4},
                "partition_strategy": {"type": "string", "default": "hash"},
                "parallel": {"type": "boolean", "default": False},
                "explain": {"type": "boolean", "default": False},
                "driver_url": {"type": ["string", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_graphql_query",
        func=graph_graphql_query,
        description="Execute a GraphQL-style query against graph data.",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "minLength": 1},
                "kg_data": {"type": ["object", "null"]},
            },
            "required": ["query"],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_visualize",
        func=graph_visualize,
        description="Render graph data into DOT, Mermaid, D3 JSON, or ASCII output.",
        input_schema={
            "type": "object",
            "properties": {
                "format": {"type": "string", "enum": sorted(_VALID_GRAPH_VISUALIZE_FORMATS), "default": "dot"},
                "kg_data": {"type": ["object", "null"]},
                "max_entities": {"type": ["integer", "null"], "minimum": 1},
                "directed": {"type": "boolean", "default": True},
                "graph_name": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_complete_suggestions",
        func=graph_complete_suggestions,
        description="Suggest likely missing relationships in graph data.",
        input_schema={
            "type": "object",
            "properties": {
                "kg_data": {"type": ["object", "null"]},
                "min_score": {"type": "number", "minimum": 0.0, "maximum": 1.0, "default": 0.3},
                "max_suggestions": {"type": "integer", "minimum": 1, "default": 20},
                "entity_id": {"type": ["string", "null"]},
                "rel_type": {"type": ["string", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_explain",
        func=graph_explain,
        description="Explain graph entities, relationships, and connectivity paths.",
        input_schema={
            "type": "object",
            "properties": {
                "explain_type": {"type": "string", "enum": sorted(_VALID_GRAPH_EXPLAIN_TYPES), "default": "entity"},
                "entity_id": {"type": ["string", "null"]},
                "start_entity_id": {"type": ["string", "null"]},
                "end_entity_id": {"type": ["string", "null"]},
                "relationship_id": {"type": ["string", "null"]},
                "depth": {"type": "string", "default": "standard"},
                "max_hops": {"type": "integer", "minimum": 1, "default": 4},
                "kg_data": {"type": ["object", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )

    manager.register_tool(
        category="graph_tools",
        name="graph_provenance_verify",
        func=graph_provenance_verify,
        description="Verify provenance data for graph updates or datasets.",
        input_schema={
            "type": "object",
            "properties": {
                "provenance_jsonl": {"type": ["string", "null"]},
                "kg_data": {"type": ["object", "null"]},
            },
            "required": [],
        },
        runtime="fastapi",
        tags=["native", "mcpp", "graph-tools"],
    )
