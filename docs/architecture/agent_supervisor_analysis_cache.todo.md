# Agent Supervisor Analysis Cache Task Board

This is one conflict-minimized lane in the first analysis-offload tranche.
Shared exports and orchestration wiring are deferred to the integration
tranche.

## ANALYSIS-CACHE-001 Implement the content-addressed analysis cache

- Status: todo
- Completion: manual
- Priority: P0
- Track: analysis-cache
- Depends on:
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis_cache.py, test/api/test_agent_supervisor_analysis_cache.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_cache.py -q
- Board namespace: agent-supervisor-analysis-offload-v1
- Parallel lane: analysis-cache
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_cache.py, test/api/test_agent_supervisor_analysis_cache.py
- Acceptance: First inspect artifact_store.py, formal_verification_cache.py, duckdb_state.py, task_identity.py, and the repository's atomic-write and checkout-lock helpers. Implement a bounded content-addressed cache for analysis-stage receipts keyed by repository tree identity, objective revision, analyzer and schema versions, configuration digest, query digest, and policy digest. Cache records must store compact receipts and artifact references rather than source text, decoded model output, AST bodies, or recursively nested artifact graphs. Distinguish successful, partial, failed, timed-out, and inconclusive entries; negative or inconclusive entries require bounded TTLs and can never count as completion evidence. Provide deterministic serialization, corruption-safe reads, atomic concurrent writes, explicit hit/miss/invalidation reasons, and configurable entry/count/byte retention bounds. Add focused tests for exact-key hits, every invalidation dimension, stale negative entries, corruption recovery, concurrent writers, and persistence-size bounds. Keep this module standalone for the parallel tranche and do not edit agent_supervisor/__init__.py, audit_scanner.py, or shared contracts.
