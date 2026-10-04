# Agent Supervisor AST Evidence Task Board

This is one conflict-minimized lane in the first analysis-offload tranche.
Shared exports and orchestration wiring are deferred to the integration
tranche.

## ANALYSIS-AST-001 Build the canonical incremental AST evidence index

- Status: todo
- Completion: manual
- Priority: P0
- Track: ast-evidence
- Depends on:
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis_ast_index.py, test/api/test_agent_supervisor_analysis_ast_index.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_ast_index.py -q
- Board namespace: agent-supervisor-analysis-offload-v1
- Parallel lane: ast-evidence
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_ast_index.py, test/api/test_agent_supervisor_analysis_ast_index.py
- Acceptance: First inspect conflict_graph.ASTBlobRecord, index_ast_blob_records, objective_graph.collect_ast_dataset_records, code_evidence_graph.py, and proof_scope_index.py. Build a deterministic, incrementally reusable index over the existing canonical AST records; do not add a second parser or a competing AST record schema. Provide bounded queries for paths, symbols, definitions, imports, calls, references, and objective terms, with source/blob identities and explicit truncation metadata. Reuse unchanged blob records, invalidate changed blobs, and return compact evidence references and ranking explanations rather than full AST or source payloads. Add focused tests for deterministic indexing, unchanged-record reuse, changed-record invalidation, symbol and relationship lookup, stable ranking, and strict result/byte bounds. Keep this module standalone for the parallel tranche and do not edit agent_supervisor/__init__.py, audit_scanner.py, or shared contracts.
