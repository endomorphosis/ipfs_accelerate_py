#!/usr/bin/env python3
"""UNI-204 graph import compatibility tests."""

from __future__ import annotations

from ipfs_accelerate_py.mcp_server.tools.graph_tools import (
    graph_add_entity,
    graph_add_relationship,
    graph_complete_suggestions,
    graph_constraint_add,
    graph_create,
    graph_distributed_execute,
    graph_explain,
    graph_graphql_query,
    graph_index_create,
    graph_ontology_materialize,
    graph_provenance_verify,
    graph_query_cypher,
    graph_search_hybrid,
    graph_srl_extract,
    graph_transaction_begin,
    graph_transaction_commit,
    graph_transaction_rollback,
    graph_visualize,
    query_knowledge_graph,
)
from ipfs_accelerate_py.mcp_server.tools.graph_tools import native_graph_tools


def test_graph_package_exports_supported_native_functions() -> None:
    assert query_knowledge_graph is native_graph_tools.query_knowledge_graph
    assert graph_create is native_graph_tools.graph_create
    assert graph_add_entity is native_graph_tools.graph_add_entity
    assert graph_add_relationship is native_graph_tools.graph_add_relationship
    assert graph_query_cypher is native_graph_tools.graph_query_cypher
    assert graph_search_hybrid is native_graph_tools.graph_search_hybrid
    assert graph_transaction_begin is native_graph_tools.graph_transaction_begin
    assert graph_transaction_commit is native_graph_tools.graph_transaction_commit
    assert graph_transaction_rollback is native_graph_tools.graph_transaction_rollback
    assert graph_index_create is native_graph_tools.graph_index_create
    assert graph_constraint_add is native_graph_tools.graph_constraint_add
    assert graph_srl_extract is native_graph_tools.graph_srl_extract
    assert graph_ontology_materialize is native_graph_tools.graph_ontology_materialize
    assert graph_distributed_execute is native_graph_tools.graph_distributed_execute
    assert graph_graphql_query is native_graph_tools.graph_graphql_query
    assert graph_visualize is native_graph_tools.graph_visualize
    assert graph_complete_suggestions is native_graph_tools.graph_complete_suggestions
    assert graph_explain is native_graph_tools.graph_explain
    assert graph_provenance_verify is native_graph_tools.graph_provenance_verify
