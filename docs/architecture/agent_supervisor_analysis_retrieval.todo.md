# Agent Supervisor Analysis Retrieval Task Board

This is one conflict-minimized lane in the first analysis-offload tranche.
Shared exports and orchestration wiring are deferred to the integration
tranche.

## ANALYSIS-GRAPH-001 Add bounded multi-signal GraphRAG retrieval

- Status: todo
- Completion: manual
- Priority: P0
- Track: analysis-retrieval
- Depends on:
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis_retrieval.py, test/api/test_agent_supervisor_analysis_retrieval.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_retrieval.py -q
- Board namespace: agent-supervisor-analysis-offload-v1
- Parallel lane: analysis-retrieval
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_retrieval.py, test/api/test_agent_supervisor_analysis_retrieval.py
- Acceptance: First inspect code_evidence_graph.py, objective_graph.py, goal_coverage.py, proof_scope_index.py, todo_vector_index.py, and artifact_store.py. Implement a deterministic bounded retrieval layer that can fuse lexical, vector-when-available, AST-symbol, dependency-neighborhood, goal-coverage, and proof-gap signals without requiring every optional backend. Results must include stable evidence references, per-signal scores, a ranking explanation, backend health, and truncation metadata; unavailable or unhealthy signals must degrade explicitly rather than silently changing semantics. Prevent source bodies, decoded model text, nested artifact graphs, and other large payloads from entering result records. Add tests for ranking stability, signal fusion, optional-backend degradation, provenance, deduplication, and count/byte limits. Keep this module standalone for the parallel tranche and do not edit agent_supervisor/__init__.py, audit_scanner.py, or shared contracts.
