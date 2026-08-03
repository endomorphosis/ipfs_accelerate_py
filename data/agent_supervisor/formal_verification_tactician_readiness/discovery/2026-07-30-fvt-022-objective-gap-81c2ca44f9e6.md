# FVT-022 Objective Goal Gap

Date: 2026-07-30
Fingerprint: 81c2ca44f9e6220544f521e25689920ed2f3bc96
Goal id: FVT-G043
Goal title: Deduplicate semantic witnesses and quarantine disagreement
Objective heap: docs/architecture/formal_verification_tactician_readiness.objectives.md
Priority: P1
Track: counterexample-equivalence
Status: todo
Schedulable: true
Review only: false
Parent goals: FVT-G000
Graph depth: 1
Objective heap index: 21
Bundle: formal-verification-tactician/counterexamples
Parallel lane: formal-verification-tactician/counterexamples
Bundle strategy: explicit
Goal packet: none
Goal packet role: none
Goal packet goals: none
Goal packet task count: 0
Goal packet work item count: 0
Evidence methods: none
Embedding query: Define property-specific semantic witness equivalence, diversity/coverage selection, cross-provider differential replay, and explicit disagreement quarantine.
AST query: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/equivalence.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_equivalence.py
Conflict policy: Own equivalence/diversity/differential tests; do not use hashes alone as semantic equivalence or discard contradictory evidence.
Predicted files: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/equivalence.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_equivalence.py
AST symbols: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/equivalence.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_equivalence.py
Interfaces: CounterexampleSemanticEquivalence@1
Submodules: ipfs_datasets_py
Generated artifacts: none
Allow concurrent with: none
Semantic identity: objective-evidence-obligation/v1/fb23395c46c669923c7bc7b0d3d123d9a5dc5fb6bd44b33a8767ff4c62fc82e1
Acceptance subset: Syntactic variants of one witness deduplicate only under a reviewed semantic relation, materially different causal paths remain diverse, cross-provider disagreement is retained with both receipts and cannot raise authority or be reported as consensus.
Preconditions: objective goal FVT-G043 is schedulable
Effects: satisfy evidence requirement: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/equivalence.py, satisfy evidence requirement: ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_equivalence.py
Evidence subset: ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/equivalence.py, ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_equivalence.py
Dependencies: FVT-G012, FVT-G041
Resource class: cpu-proof-differential
Token class: medium
Estimated tokens: 0
Resources: cpu-proof-differential
Merge fate: objective/FVT-G043
Rejection reasons: none (accepted)

## Goal

Define property-specific semantic witness equivalence, diversity/coverage selection, cross-provider differential replay, and explicit disagreement quarantine.

## Missing Evidence

- ipfs_datasets_py/ipfs_datasets_py/logic/software_verification/counterexamples/equivalence.py
- ipfs_datasets_py/tests/integration/logic/software_verification/counterexamples/test_equivalence.py

## Present Evidence

- none found for this goal

## Suggested Handling

Close the missing evidence with focused code, tests, or documentation.
