# Agent Supervisor Analysis Contracts Task Board

This is one conflict-minimized lane in the first analysis-offload tranche.
Shared exports and orchestration wiring are deferred to the integration
tranche.

## ANALYSIS-CONTRACT-001 Define bounded analysis evidence contracts

- Status: completed
- Completion: manual
- Priority: P0
- Track: analysis-contracts
- Depends on:
- Outputs: ipfs_accelerate_py/agent_supervisor/analysis_contracts.py, test/api/test_agent_supervisor_analysis_contracts.py
- Validation: python -m pytest test/api/test_agent_supervisor_analysis_contracts.py -q
- Board namespace: agent-supervisor-analysis-offload-v1
- Parallel lane: analysis-contracts
- Predicted files: ipfs_accelerate_py/agent_supervisor/analysis_contracts.py, test/api/test_agent_supervisor_analysis_contracts.py
- Acceptance: First inspect analyzer_health.py, scan_receipts.py, artifact_store.py, formal_planning_contracts.py, and task_proposal_router.py and reuse their established identity and serialization conventions. Add immutable typed contracts for bounded analysis evidence packets, stage receipts, candidate proposals, provenance references, confidence/novelty/cost fields, and explicit conclusive versus inconclusive outcomes. Serialization and content identities must be deterministic. Contracts must support artifact references instead of embedding unbounded source, AST, proof, prompt, or model-response bodies; enforce configurable count and byte limits; and make it impossible for failed, partial, stale, or negatively cached analysis to masquerade as completion evidence. Add focused round-trip, determinism, bounds, and invalid-state tests. Do not edit agent_supervisor/__init__.py or wire the contracts into audit_scanner.py in this task.
