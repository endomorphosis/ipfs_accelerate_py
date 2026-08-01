# LFV-037 Objective Goal Gap

Date: 2026-07-29
Fingerprint: 2ceefa799561e3f4c0da017d663ac78654a3596e
Goal id: LFV-G072
Goal title: Adapt the agent supervisor to the canonical logic platform
Objective heap: docs/architecture/logic_formal_verification_expansion.objectives.md
Priority: P1
Track: integration
Status: todo
Schedulable: true
Review only: false
Parent goals: LFV-G000
Graph depth: 1
Objective heap index: 36
Bundle: logic-formal-verification/supervisor
Parallel lane: logic-formal-verification/supervisor
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Replace overlapping supervisor family/capability/provider vocabularies with thin canonical adapters while retaining scheduling, resource, isolation, routing, cache, and evidence behavior.
AST query: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, test/api/test_agent_supervisor_canonical_logic.py
Conflict policy: Single owner for supervisor registry/router compatibility edits; do not move orchestration/resource code into datasets or duplicate semantic contracts.
Predicted files: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_operation_registry.py, ipfs_accelerate_py/agent_supervisor/proof/logic_translation_validation.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py, ipfs_accelerate_py/agent_supervisor/proof/prover_matrix_registry.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_capabilities.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_provider.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_canonical_logic.py
AST symbols: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, test/api/test_agent_supervisor_canonical_logic.py
Interfaces: SupervisorCanonicalLogicAdapter@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/21ee2f42a85d6d74bc5aa2dd98801d75d6df87b1741fb5488759f5dd78651ca1
Acceptance subset: Analysis families, property kinds, translation forms, matrix entries, capability probes, providers, routes, resources, caches, and receipts map losslessly, supervisor-local facades remain compatible, datasets imports are lazy, cross-repo current-revision checks pass.
Preconditions: objective goal LFV-G072 is schedulable
Effects: satisfy evidence requirement: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, satisfy evidence requirement: test/api/test_agent_supervisor_canonical_logic.py
Evidence subset: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, test/api/test_agent_supervisor_canonical_logic.py
Dependencies: LFV-G015, LFV-G043, LFV-G044, LFV-G045, LFV-G046, LFV-G047, LFV-G048, LFV-G050, LFV-G062, LFV-G070
Resource class: cpu-medium
Token class: medium
Estimated tokens: 0
Resources: cpu-medium
Merge fate: objective/LFV-G072
Rejection reasons: none (accepted)

## Goal

Replace overlapping supervisor family/capability/provider vocabularies with thin canonical adapters while retaining scheduling, resource, isolation, routing, cache, and evidence behavior.

## Missing Evidence

- ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py
- test/api/test_agent_supervisor_canonical_logic.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
