# FVT-008 Objective Goal Gap

Date: 2026-07-30
Fingerprint: a9f0c33ea8d6e4d7bdba03e367f5f8170efcf12f
Goal id: FVT-G040
Goal title: Implement oracle-preserving semantic counterexample minimizers
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P0
Track: counterexample-minimization
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 7
Bundle: formal-verification-tactician/counterexamples
Parallel lane: formal-verification-tactician/counterexamples
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Replace unconditional Boolean minimization with backend-specific, budgeted, oracle-preserving model/core/trace/attack/hypertrace/kernel reducers and truthful reduction guarantees.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/minimization.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_semantic_minimization.py
Conflict policy: Own semantic reducer protocols/implementations and tests; retain normalization/bounding as a distinct lower guarantee and never stamp `minimized` merely because output is short.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/minimization.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_semantic_minimization.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/minimization.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_semantic_minimization.py
Interfaces: SemanticCounterexampleMinimizer@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/115e120af8b1ebf9592cdfbf7aef83f0632a579640e9ae03c6909688bde64c03
Acceptance subset: SMT projection/don't-cares and subset cores, shortest prefix/lasso/event slice, protocol dependency slice, and earliest hypertrace divergence recheck the violation after every accepted removal, receipts record oracle, algorithm/version, budget, reduction log, and actual guarantee including exhaustion.
Preconditions: objective goal FVT-G040 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/minimization.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_semantic_minimization.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/minimization.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_semantic_minimization.py
Dependencies: FVT-G007, FVT-G020
Resource class: cpu-proof-minimize
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-minimize
Merge fate: objective/FVT-G040
Rejection reasons: none (accepted)

## Goal

Replace unconditional Boolean minimization with backend-specific, budgeted, oracle-preserving model/core/trace/attack/hypertrace/kernel reducers and truthful reduction guarantees.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/minimization.py
- ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_semantic_minimization.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
