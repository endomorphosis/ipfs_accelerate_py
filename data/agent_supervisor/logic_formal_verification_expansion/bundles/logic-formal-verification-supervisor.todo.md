# Objective Bundle: logic-formal-verification/supervisor

Source todo: docs/architecture/logic_formal_verification_expansion.todo.md
Purpose: bundle objective-generated tasks so parallel daemons can work one lane at a time.
Conflict policy: keep edits inside this bundle when possible; use the LLM merge resolver for semantic conflicts.

## LFV-037 Close objective gap: Adapt the agent supervisor to the canonical logic platform

- Status: todo
- Completion: manual
- Is schedulable: true
- Review only: false
- Priority: P1
- Track: integration
- Depends on: LFV-008, LFV-035, LFV-023, LFV-021, LFV-027, LFV-024, LFV-026, LFV-032, LFV-025, LFV-036
- Outputs: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_operation_registry.py, ipfs_accelerate_py/agent_supervisor/proof/logic_translation_validation.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py, ipfs_accelerate_py/agent_supervisor/proof/prover_matrix_registry.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_capabilities.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_provider.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_canonical_logic.py
- Validation: python -m pytest test/api/test_agent_supervisor_canonical_logic.py -q
- Evidence inputs: data/agent_supervisor/discovery
- Discovery evidence: /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/logic_formal_verification_expansion/discovery/2026-07-29-lfv-037-objective-gap-2ceefa799561.md
- Bundle: logic-formal-verification/supervisor
- Bundle shard: data/agent_supervisor/logic_formal_verification_expansion/bundles/logic-formal-verification-supervisor.todo.md
- Bundle strategy: explicit
- Graph parents: LFV-G000
- Graph depth: 1
- Objective heap index: 36
- Parallel lane: logic-formal-verification/supervisor
- Conflict policy: Single owner for supervisor registry/router compatibility edits; do not move orchestration/resource code into datasets or duplicate semantic contracts.
- Predicted files: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, ipfs_accelerate_py/agent_supervisor/analysis/analysis_operation_registry.py, ipfs_accelerate_py/agent_supervisor/proof/logic_translation_validation.py, ipfs_accelerate_py/agent_supervisor/proof/multi_prover_router.py, ipfs_accelerate_py/agent_supervisor/proof/prover_matrix_registry.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_capabilities.py, ipfs_accelerate_py/agent_supervisor/proof/formal_verification_provider.py, ipfs_accelerate_py/agent_supervisor/integrations/ipfs_datasets_logic_provider.py, test/api/test_agent_supervisor_canonical_logic.py
- Changed paths:
- AST symbols: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, test/api/test_agent_supervisor_canonical_logic.py
- Interfaces: SupervisorCanonicalLogicAdapter@1
- Submodules: ipfs_datasets_py
- Generated artifacts:
- Allow concurrent with:
- Goal id: LFV-G072
- Completion authority: local
- External authority blockers:
- Canonical task key: task/v1/b967870d4081929a35432a2f034ebb34b7df4e523cdf49acc2a4d9cbca396604
- Canonical task CID: baguqeeraxftyodkaqgjjunkdfixqgtv3gs356tsshtputlgcutm4xsrzmyca
- Semantic identity: objective-evidence-obligation/v1/21ee2f42a85d6d74bc5aa2dd98801d75d6df87b1741fb5488759f5dd78651ca1
- Acceptance subset: Analysis families, property kinds, translation forms, matrix entries, capability probes, providers, routes, resources, caches, and receipts map losslessly, supervisor-local facades remain compatible, datasets imports are lazy, cross-repo current-revision checks pass.
- Preconditions: objective goal LFV-G072 is schedulable
- Effects: satisfy evidence requirement: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, satisfy evidence requirement: test/api/test_agent_supervisor_canonical_logic.py
- Evidence subset: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, test/api/test_agent_supervisor_canonical_logic.py
- Resource class: cpu-medium
- Token class: medium
- Estimated tokens: 0
- Resources: cpu-medium
- Merge fate: objective/LFV-G072
- Rejection reasons: none (accepted)
- Evidence obligation key: objective-evidence-obligation/v1/21ee2f42a85d6d74bc5aa2dd98801d75d6df87b1741fb5488759f5dd78651ca1
- Missing evidence: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, test/api/test_agent_supervisor_canonical_logic.py
- Embedding query: Replace overlapping supervisor family/capability/provider vocabularies with thin canonical adapters while retaining scheduling, resource, isolation, routing, cache, and evidence behavior.
- AST query: ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, test/api/test_agent_supervisor_canonical_logic.py
- Surplus group: objective/LFV-G072
- Merge key: 7f6a0d8dcaa32f0b
- Merge family: objective/LFV-G072
- Merge role: aggregate
- Work item count: 2
- Work scope: goal_subgoal_multi_evidence_batch
- Goal packet:
- Goal packet role:
- Goal packet goals:
- Goal packet task count: 0
- Goal packet work item count: 0
- Completion goal bindings: {}
- Completion task bindings:
- Candidate kind: aggregate
- Todo vector key: b69502db591a80f1
- Acceptance: Objective scan filed this gap for LFV-G072. Use evidence in /home/barberb/lift_coding/data/logic_software_verification_program/repo/data/agent_supervisor/logic_formal_verification_expansion/discovery/2026-07-29-lfv-037-objective-gap-2ceefa799561.md, add code/tests/docs or child goals that prove the missing evidence terms are covered (ipfs_accelerate_py/agent_supervisor/canonical_logic_adapter.py, test/api/test_agent_supervisor_canonical_logic.py), and keep the supervisor-fed backlog aligned with the objective heap.  Refine the objective heap if the gap needs smaller child goals.
