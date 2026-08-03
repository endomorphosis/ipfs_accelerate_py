# FVT-027 Objective Goal Gap

Date: 2026-07-30
Fingerprint: aa091bd5c56bb39cc2f38b51e0e587a5131fd427
Goal id: FVT-G036
Goal title: Integrate the goal-directed tactician with existing utilities
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: tactician-integration
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 26
Bundle: formal-verification-tactician/proof-search
Parallel lane: formal-verification-tactician/proof-search
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Compose formalization, retrieval, proof scheduler, proof-carrying planner, Hammer/kernels, Leanstral, SymAI, autoencoder, legal evidence adapter, caches, corpus, ZKP receipt binding, and supervisor admission behind one restartable tactician.
AST query: ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py, test/api/test_goal_directed_tactician_integration.py
Conflict policy: Own the parent orchestration facade and integration test; import canonical datasets contracts through the existing provider boundary and do not duplicate semantics in the supervisor.
Predicted files: ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py, test/api/test_goal_directed_tactician_integration.py
AST symbols: ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py, test/api/test_goal_directed_tactician_integration.py
Interfaces: GoalDirectedProofTactician@1
Submodules: none
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/7d90eb37c590fc9b0c82b5a0f68e89d8d513a4c6723471af8e70980e3adac08f
Acceptance subset: Exact cache keys include tree/target/assumptions/provider/version/policy/bounds, model and cache evidence cannot bypass validation, proof-carrying execution is resumable, ZKP binds an existing trusted receipt without increasing its assurance, legal compatibility remains intact.
Preconditions: objective goal FVT-G036 is schedulable
Effects: satisfy evidence requirement: ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py, satisfy evidence requirement: test/api/test_goal_directed_tactician_integration.py
Evidence subset: ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py, test/api/test_goal_directed_tactician_integration.py
Dependencies: FVT-G025, FVT-G035
Resource class: cpu-proof-orchestrate
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-orchestrate
Merge fate: objective/FVT-G036
Rejection reasons: none (accepted)

## Goal

Compose formalization, retrieval, proof scheduler, proof-carrying planner, Hammer/kernels, Leanstral, SymAI, autoencoder, legal evidence adapter, caches, corpus, ZKP receipt binding, and supervisor admission behind one restartable tactician.

## Missing Evidence

- ipfs_accelerate_py/agent_supervisor/proof/goal_directed_tactician.py
- test/api/test_goal_directed_tactician_integration.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
